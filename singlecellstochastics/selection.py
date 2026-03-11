import argparse
import pandas as pd
import numpy as np
import torch
from torch.distributions.chi2 import Chi2
import wandb
from statsmodels.stats.multitest import multipletests
import os

from .preprocess import process_data_BM, process_data_OU
from .optimize import Lq_optimize_torch_BM, Lq_optimize_torch_OU


def gene_expression_selection(
    tree_files, gene_files, regime_files, library_files, outfile, device, batch_size,
    max_iter, learning_rate, wandb_flag, window, tol, approx, nb, rnull, prior, kkt, const
):
    """
    Selection test: BM (H0, no selection) vs OU (H1, selection) LRT.
    Tests whether gene expression evolves under stabilizing selection (OU)
    or neutrally (BM) along the lineage tree.
    """
    if wandb_flag:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name=wandb_flag)

    # Process data for BM (H0)
    print("Preprocessing data (BM)")
    (
        tree_list, cells_list, df_list, share_list_torch, library_list
    ) = process_data_BM(tree_files, gene_files, library_files, device)

    # Process data for OU (H1)
    print("Preprocessing data (OU)")
    (
        _, _, _, diverge_list, share_list, epochs_list, beta_list,
        diverge_list_torch, share_list_torch_ou, epochs_list_torch,
        beta_list_torch, regime_list, _
    ) = process_data_OU(tree_files, gene_files, regime_files, library_files, rnull, device)

    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=torch.float32, device=device) for lib in library_list
    ]

    regimes = list(dict.fromkeys(x for sub in regime_list for x in sub))
    n_regimes = len(regimes)

    if batch_size > len(df_list[0].columns):
        batch_size = len(df_list[0].columns)

    results_list = []
    bm_q_list = []
    ou_q_list = []
    genes = []
    print("Running selection test (BM vs OU)")
    for batch_start in range(0, len(df_list[0].columns), batch_size):
        batch_end = min(batch_start + batch_size, len(df_list[0].columns))
        gene_names = df_list[0].columns[batch_start:batch_end]
        actual_batch = len(gene_names)
        genes.extend(gene_names)
        x_list = [df[gene_names].values.T for df in df_list]  # (batch, cells)
        x_tensor_list = [
            torch.tensor(x, dtype=torch.float32, device=device)
            for x in x_list
        ]

        # --- H0: BM model ---
        s_init_tensor = [
            x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
            for x in x_tensor_list
        ]
        q_params_init_bm = [
            torch.cat((x_tensor_list[i], s_init_tensor[i]), dim=-1)
            for i in range(len(x_tensor_list))
        ]
        log_r_bm = torch.zeros((actual_batch,), dtype=torch.float32, device=device)
        bm_params_init = torch.ones((actual_batch,), dtype=torch.float32, device=device) * (-2)
        init_params_bm = q_params_init_bm + [log_r_bm] + [bm_params_init]

        h0_params, h0_loss = Lq_optimize_torch_BM(
            init_params_bm,
            1,  # mode=1: use original tree
            x_tensor_list,
            gene_names,
            share_list_torch,
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

        # Save BM variational parameters for reconstruct_BM
        bm_q_list.append(h0_params[:-2])  # list of (batch, 2*n_cells) per clone

        # Save OU variational parameters for ou_diff H0 init
        ou_q_list.append(h1_params[:-2])  # list of (batch, 1, 2*n_cells) per clone

        # --- H1: OU model ---
        # Prepare OU input tensors with (batch, 1, cells) shape
        x_tensor_ou = [
            x.unsqueeze(1) for x in x_tensor_list
        ]  # list of (batch, 1, cells)

        s_init_ou = [
            x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
            for x in x_tensor_ou
        ]
        q_params_init_ou = [
            torch.cat((x_tensor_ou[i], s_init_ou[i]), dim=-1)
            for i in range(len(x_tensor_ou))
        ]
        log_r_ou = torch.zeros((actual_batch, 1), dtype=torch.float32, device=device)
        ou_params_init = torch.ones(
            (actual_batch, 1, n_regimes + 2), dtype=torch.float32, device=device
        )
        init_params_ou = q_params_init_ou + [log_r_ou] + [ou_params_init]

        h1_params, h1_loss = Lq_optimize_torch_OU(
            init_params_ou,
            1,  # mode=1: H0 (shared OU params across regimes)
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
            None,  # em
            prior,
            kkt,
            nb,
            library_list_tensor,
            const,
        )
        h1_loss = h1_loss.squeeze(1)  # (batch,)

        # LRT: 2 * (H0_nll - H1_nll)
        lr_stat = 2 * (h0_loss - h1_loss)
        # OU has 1 extra param vs BM (alpha; theta replaces mu), df=1
        chi2_dist = Chi2(torch.tensor([1.0], device=device))
        lr_stat_safe = torch.nan_to_num(lr_stat, nan=0.0).clamp(min=0)
        p_value = 1.0 - chi2_dist.cdf(lr_stat_safe)

        # Collect results
        # BM params: log_r, mu, sigma
        # OU params: log_r, alpha, sigma, theta0, ...
        bm_r = h0_params[-2].exp()
        bm_model = h0_params[-1]  # (batch, 3): lambda, mu, sigma
        ou_r = h1_params[-2].exp().squeeze(1)
        ou_model = h1_params[-1].squeeze(1)  # (batch, n_regimes+2): alpha, sigma, theta0, ...

        result = torch.cat((
            bm_r.unsqueeze(-1),
            bm_model,
            ou_r.unsqueeze(-1),
            ou_model,
            h0_loss.unsqueeze(-1),
            h1_loss.unsqueeze(-1),
            lr_stat.unsqueeze(-1),
            p_value.unsqueeze(-1),
        ), dim=-1)
        results_list.append(result.detach().cpu())

    results = torch.cat(results_list, dim=0)

    # Build column names
    bm_cols = ["h0_r", "h0_lambda", "h0_mu", "h0_sigma"]
    ou_cols = ["h1_r", "h1_alpha", "h1_sigma"] + [f"h1_theta{i}" for i in range(n_regimes)]
    stat_cols = ["h0", "h1", "lr", "p"]
    columns = bm_cols + ou_cols + stat_cols

    results_df = pd.DataFrame(results.cpu().numpy(), columns=columns)
    results_df.insert(0, "gene", genes)
    results_df.insert(0, "ID", range(1, len(results_df) + 1))

    results_df["signif"], results_df["q"] = multipletests(results_df["p"], alpha=0.05, method="fdr_bh")[:2]
    results_df = results_df.sort_values("q")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    results_df.to_csv(outfile, sep="\t", index=False)

    # Save BM variational parameters (q_mean, q_std) for reconstruct_BM
    bm_q_params = [torch.cat(q_batch, dim=0) for q_batch in zip(*bm_q_list)]
    outdir = os.path.dirname(outfile)
    for i in range(len(bm_q_params)):
        df = pd.DataFrame(
            bm_q_params[i].detach().cpu().numpy(),
            index=genes,
            columns=cells_list[i] * 2
        )
        df.to_csv(os.path.join(outdir, f"bm_q_params_{i}.tsv"), sep="\t")

    # Save OU variational parameters (q_mean, q_std) for ou_diff H0 init
    ou_q_params = [torch.cat(q_batch, dim=0) for q_batch in zip(*ou_q_list)]
    for i in range(len(ou_q_params)):
        # OU q params have shape (batch, 1, 2*n_cells); squeeze the N_sim=1 dim
        df = pd.DataFrame(
            ou_q_params[i].squeeze(1).detach().cpu().numpy(),
            index=genes,
            columns=cells_list[i] * 2
        )
        df.to_csv(os.path.join(outdir, f"ou_q_params_{i}.tsv"), sep="\t")


def run_selection_test():
    parser = argparse.ArgumentParser(description="Selection test: BM vs OU LRT for gene expression evolution")
    parser.add_argument("--tree", required=True, type=str, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, type=str, help="Path to cell by gene expression data (TSV)")
    parser.add_argument("--regime", required=True, type=str, help="Path to regime file (CSV)")
    parser.add_argument("--null", required=True, type=str, help="Regime for null hypothesis")
    parser.add_argument("--library", required=False, type=str, default=None, help="Path to library file (TSV)")
    parser.add_argument("--outfile", required=False, type=str, default="./selection.tsv", help="Path to save the results (TSV)")
    parser.add_argument("--model", required=False, type=str, default="nb", choices=["pois", "nb"],
                        help="Observation model: pois (Poisson) or nb (Negative Binomial, default)")
    parser.add_argument("--batch_size", required=False, type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max_iter", required=False, type=int, default=10000, help="Maximum number of optimization iterations")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-1, help="Learning rate for optimization")
    parser.add_argument("--wandb_flag", required=False, type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--window", required=False, type=int, default=200, help="Window size for checking convergence")
    parser.add_argument("--tol", required=False, type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--approx", required=False, type=str, default="softplus_MC", help="Approximation method")
    parser.add_argument("--prior", required=False, type=float, default=1.0, help="L2 prior on log alpha")
    parser.add_argument("--no_kkt", action="store_false", dest="kkt", help="Disable KKT constraint")
    parser.add_argument("--const", required=False, action="store_true", help="Include constant terms in likelihood")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tree = args.tree.split(",")
    expression = args.expression.split(",")
    regime = args.regime.split(",")
    if args.library is not None:
        library = args.library.split(",")
    else:
        library = [None] * len(tree)

    nb = (args.model == "nb")

    gene_expression_selection(
        tree, expression, regime, library, args.outfile, device=device,
        batch_size=args.batch_size, max_iter=args.max_iter, learning_rate=args.learning_rate,
        wandb_flag=args.wandb_flag, window=args.window, tol=args.tol,
        approx=args.approx, nb=nb, rnull=args.null, prior=args.prior,
        kkt=args.kkt, const=args.const
    )


if __name__ == "__main__":
    run_selection_test()
