import argparse
import pandas as pd
import numpy as np
import torch
from torch.distributions.chi2 import Chi2
import wandb
from statsmodels.stats.multitest import multipletests
import os
import logging

logger = logging.getLogger(__name__)

from .preprocess import process_data_OU, process_data_BM
from .optimize import Lq_optimize_torch_OU, Lq_optimize_torch_BM


def gene_expression_selection(
    tree_files, gene_files, regime_files, library_files, outfile, device, batch_size,
    max_iter, learning_rate, wandb_flag, window, tol, approx, nb, rnull, kkt, const,
    null_model="bm", prefix="sel", resume=False
):
    """
    Selection test: BM or fixed-alpha OU (H0, neutral) vs OU (H1, selection).
    Tests whether gene expression evolves under stabilizing selection (OU)
    or neutrally (BM) along the lineage tree.

    null_model: "bm" uses proper BM code path for H0 (default),
                "fixed_ou" uses OU with alpha fixed at ~0.
    """
    if wandb_flag:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name=wandb_flag)

    # Process data for OU (H1, and H0 if fixed_ou)
    logger.info("Preprocessing data (OU)")
    (
        tree_list, cells_list, df_list, diverge_list, share_list, epochs_list, beta_list,
        diverge_list_torch, share_list_torch_ou, epochs_list_torch,
        beta_list_torch, regime_list, library_list
    ) = process_data_OU(tree_files, gene_files, regime_files, library_files, rnull, device)

    # Process data for BM (H0) if using BM null
    if null_model == "bm":
        logger.info("Preprocessing data (BM)")
        (
            _, _, _, share_list_torch_bm, _
        ) = process_data_BM(tree_files, gene_files, library_files, device)

    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=torch.float32, device=device) for lib in library_list
    ]

    regimes = list(dict.fromkeys(x for sub in regime_list for x in sub))
    n_regimes = len(regimes)

    if batch_size > len(df_list[0].columns):
        batch_size = len(df_list[0].columns)

    n_genes = len(df_list[0].columns)
    n_batches = (n_genes + batch_size - 1) // batch_size

    # Checkpoint directory for per-batch results + q params
    ckpt_dir = outfile + ".ckpt"
    if resume and os.path.isdir(ckpt_dir):
        logger.info(f"Found checkpoint directory: {ckpt_dir}")
    elif resume:
        logger.info(f"No checkpoint directory found, starting from scratch")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Running selection test ({null_model} null vs OU): {n_genes} genes, {n_batches} batches of {batch_size}")
    for batch_start in range(0, n_genes, batch_size):
        batch_idx = batch_start // batch_size
        batch_end = min(batch_start + batch_size, n_genes)
        gene_names = df_list[0].columns[batch_start:batch_end]
        actual_batch = len(gene_names)

        # Check if batch already completed
        ckpt_file = os.path.join(ckpt_dir, f"batch_{batch_idx}.pt")
        if resume and os.path.exists(ckpt_file):
            logger.info(f"Batch {batch_idx+1}/{n_batches} already done, skipping")
            continue

        x_list = [df[gene_names].values.T for df in df_list]  # (batch, cells)
        x_tensor_list = [
            torch.tensor(x, dtype=torch.float32, device=device)
            for x in x_list
        ]

        if null_model == "bm":
            # --- H0: BM (neutral) ---
            s_init_bm = [
                x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
                for x in x_tensor_list
            ]
            q_params_bm = [
                torch.cat((x_tensor_list[i], s_init_bm[i]), dim=-1)
                for i in range(len(x_tensor_list))
            ]
            log_r_bm = torch.zeros((actual_batch,), dtype=torch.float32, device=device)
            bm_lambda_init = torch.ones((actual_batch,), dtype=torch.float32, device=device)
            init_params_h0 = q_params_bm + [log_r_bm] + [bm_lambda_init]

            h0_params, h0_loss = Lq_optimize_torch_BM(
                init_params_h0,
                1,  # mode=1: original tree (lambda=1)
                x_tensor_list,
                gene_names,
                share_list_torch_bm,
                max_iter,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                approx,
                nb,
                library_list_tensor,
                const,
            )

            # H0 q params: list of (batch, 2*n_cells) per clone
            h0_q = [p.detach().cpu() for p in h0_params[:-2]]

            # H0 result columns: r, mu, sigma
            h0_r = h0_params[-2].exp().unsqueeze(-1)  # (batch, 1)
            h0_bm = h0_params[-1]  # (batch, 3): lambda, mu, sigma
            h0_result = torch.cat((h0_r, h0_bm[:, 1:]), dim=-1)  # (batch, 3)

        else:
            # --- H0: OU with alpha fixed at ~0 (fixed_ou) ---
            x_tensor_ou_h0 = [x.unsqueeze(1) for x in x_tensor_list]
            s_init_h0 = [
                x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
                for x in x_tensor_ou_h0
            ]
            q_params_h0 = [
                torch.cat((x_tensor_ou_h0[i], s_init_h0[i]), dim=-1)
                for i in range(len(x_tensor_ou_h0))
            ]
            log_r_h0 = torch.zeros((actual_batch, 1), dtype=torch.float32, device=device)
            ou_params_h0 = torch.ones(
                (actual_batch, 1, n_regimes + 2), dtype=torch.float32, device=device
            )
            ou_params_h0[:, :, 0] = -20.0  # alpha ≈ 0
            init_params_h0 = q_params_h0 + [log_r_h0] + [ou_params_h0]

            h0_params, h0_loss = Lq_optimize_torch_OU(
                init_params_h0,
                1,
                x_tensor_ou_h0,
                gene_names,
                diverge_list_torch,
                share_list_torch_ou,
                epochs_list_torch,
                beta_list_torch,
                max_iter,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                approx,
                None,
                0,
                kkt,
                nb,
                library_list_tensor,
                const,
                fix_alpha=True,
            )
            h0_loss = h0_loss.squeeze(1)

            # H0 q params
            h0_q = [p.detach().cpu() for p in h0_params[:-2]]

            # H0 result columns: r, alpha(~0), sigma, theta
            h0_result = torch.cat(
                (h0_params[-2].exp().unsqueeze(-1),
                h0_params[-1][..., :1].exp(),
                h0_params[-1][..., 1:3]), dim=-1
            ).squeeze(1)

        # --- H1: OU with alpha free ---
        x_tensor_ou = [x.unsqueeze(1) for x in x_tensor_list]
        s_init_ou = [
            x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
            for x in x_tensor_ou
        ]
        q_params_ou = [
            torch.cat((x_tensor_ou[i], s_init_ou[i]), dim=-1)
            for i in range(len(x_tensor_ou))
        ]
        log_r_ou = torch.zeros((actual_batch, 1), dtype=torch.float32, device=device)
        ou_params_init = torch.ones(
            (actual_batch, 1, n_regimes + 2), dtype=torch.float32, device=device
        )
        init_params_h1 = q_params_ou + [log_r_ou] + [ou_params_init]

        h1_params, h1_loss = Lq_optimize_torch_OU(
            init_params_h1,
            1,
            x_tensor_ou,
            gene_names,
            diverge_list_torch,
            share_list_torch_ou,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            0,
            kkt,
            nb,
            library_list_tensor,
            const,
        )
        h1_loss = h1_loss.squeeze(1)

        # H1 q params
        h1_q = [p.detach().cpu() for p in h1_params[:-2]]

        # LRT: 2 * (H0_nll - H1_nll)
        lr_stat = 2 * (h0_loss - h1_loss)
        chi2_dist = Chi2(torch.tensor([1.0], device=device))
        lr_stat_safe = torch.nan_to_num(lr_stat, nan=0.0).clamp(min=0)
        p_value = 1.0 - chi2_dist.cdf(lr_stat_safe)

        # H1 result columns: r, alpha, sigma, theta
        h1_ou_result = torch.cat(
            (h1_params[-2].exp().unsqueeze(-1),
            h1_params[-1][..., :1].exp(),
            h1_params[-1][..., 1:3]), dim=-1
        ).squeeze(1)

        result = torch.cat((
            h0_result,
            h1_ou_result,
            h0_loss.unsqueeze(-1),
            h1_loss.unsqueeze(-1),
            lr_stat.unsqueeze(-1),
            p_value.unsqueeze(-1),
        ), dim=-1)

        # Save checkpoint: results + q params
        ckpt_data = {
            'genes': list(gene_names),
            'results': result.detach().cpu(),
            'h0_q': h0_q,
            'h1_q': h1_q,
        }
        torch.save(ckpt_data, ckpt_file)
        logger.info(f"Batch {batch_idx+1}/{n_batches} done ({batch_end}/{n_genes} genes)")

    # Load all batch checkpoints and combine
    results_list = []
    genes = []
    bm_q_list = []
    ou_q_list = []
    for batch_idx in range(n_batches):
        ckpt_file = os.path.join(ckpt_dir, f"batch_{batch_idx}.pt")
        ckpt_data = torch.load(ckpt_file, weights_only=False)
        results_list.append(ckpt_data['results'])
        genes.extend(ckpt_data['genes'])
        bm_q_list.append(ckpt_data['h0_q'])
        ou_q_list.append(ckpt_data['h1_q'])

    results = torch.cat(results_list, dim=0)

    # Build column names
    if null_model == "bm":
        h0_cols = ["h0_r", "h0_mu", "h0_sigma"]
    else:
        h0_cols = ["h0_r", "h0_alpha", "h0_sigma", "h0_theta"]
    h1_cols = ["h1_r", "h1_alpha", "h1_sigma", "h1_theta"]
    stat_cols = ["h0", "h1", "lr", "p"]
    columns = h0_cols + h1_cols + stat_cols

    results_df = pd.DataFrame(results.cpu().numpy(), columns=columns)
    results_df.insert(0, "gene", genes)
    results_df.insert(0, "ID", range(1, len(results_df) + 1))

    results_df["signif"], results_df["q"] = multipletests(results_df["p"], alpha=0.05, method="fdr_bh")[:2]
    results_df = results_df.sort_values("q")

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    results_df.to_csv(outfile, sep="\t", index=False)

    # Save H0 variational parameters (q_mean, q_std)
    bm_q_params = [torch.cat(q_batch, dim=0) for q_batch in zip(*bm_q_list)]
    for i in range(len(bm_q_params)):
        q_data = bm_q_params[i]
        if null_model == "fixed_ou":
            q_data = q_data.squeeze(1)
        df = pd.DataFrame(
            q_data.detach().cpu().numpy(),
            index=genes,
            columns=cells_list[i] * 2
        )
        df.to_csv(os.path.join(outdir, f"{prefix}_bm_q_params_{i}.tsv"), sep="\t")

    # Save OU variational parameters (q_mean, q_std)
    ou_q_params = [torch.cat(q_batch, dim=0) for q_batch in zip(*ou_q_list)]
    for i in range(len(ou_q_params)):
        df = pd.DataFrame(
            ou_q_params[i].squeeze(1).detach().cpu().numpy(),
            index=genes,
            columns=cells_list[i] * 2
        )
        df.to_csv(os.path.join(outdir, f"{prefix}_ou_q_params_{i}.tsv"), sep="\t")

    # Clean up checkpoint directory
    import shutil
    shutil.rmtree(ckpt_dir)
    logger.info(f"Results saved to {outfile}")


def run_selection_test():
    parser = argparse.ArgumentParser(description="Selection test: BM vs OU LRT for gene expression evolution")
    parser.add_argument("--tree", required=True, type=str, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, type=str, help="Path to cell by gene expression data (TSV)")
    parser.add_argument("--regime", required=True, type=str, help="Path to regime file (CSV)")
    parser.add_argument("--null", required=True, type=str, help="Regime for null hypothesis")
    parser.add_argument("--library", required=False, type=str, default=None, help="Path to library file (TSV)")
    parser.add_argument("--outfile", required=False, type=str, default="./selection.tsv", help="Path to save the results (TSV)")
    parser.add_argument("--prefix", required=False, type=str, default="sel", help="Prefix for BM/OU parameter output files")
    parser.add_argument("--batch", required=False, type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--iter", required=False, type=int, default=10000, help="Max optimization iterations")
    parser.add_argument("--lr", required=False, type=float, default=1e-1, help="Learning rate for optimization")
    parser.add_argument("--wandb", required=False, type=str, default=None, help="Wandb run name")
    parser.add_argument("--window", required=False, type=int, default=200, help="Window size for checking convergence")
    parser.add_argument("--tol", required=False, type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--approx", required=False, type=str, default="softplus_MC", help="Approximation method")
    parser.add_argument("--no_kkt", action="store_false", dest="kkt", help="Disable KKT constraint")
    parser.add_argument("--no_nb", required=False, action="store_false", help="Use poisson instead of negative binomial (default: use negative binomial)")
    parser.add_argument("--const", required=False, action="store_true", help="Include constant terms in likelihood")
    parser.add_argument("--null_model", required=False, type=str, default="bm", choices=["bm", "fixed_ou"],
                        help="Null model: 'bm' (proper BM, default) or 'fixed_ou' (OU with alpha fixed at ~0)")
    parser.add_argument("--log", type=str, default=None, help="Path to log file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log:
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tree = args.tree.split(",")
    expression = args.expression.split(",")
    regime = args.regime.split(",")
    if args.library is not None:
        library = args.library.split(",")
    else:
        library = [None] * len(tree)

    gene_expression_selection(
        tree, expression, regime, library, args.outfile, device=device,
        batch_size=args.batch, max_iter=args.iter, learning_rate=args.lr,
        wandb_flag=args.wandb, window=args.window, tol=args.tol,
        approx=args.approx, nb=args.no_nb, rnull=args.null, kkt=args.kkt,
        const=args.const, null_model=args.null_model, prefix=args.prefix,
        resume=args.resume
    )


if __name__ == "__main__":
    run_selection_test()
