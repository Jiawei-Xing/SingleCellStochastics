import argparse
import pandas as pd
import numpy as np
import torch
from torch.distributions.chi2 import Chi2
import wandb
from statsmodels.stats.multitest import multipletests
import os
import time
import logging

logger = logging.getLogger(__name__)

from .preprocess import process_data_BM
from .optimize import Lq_optimize_torch_BM, optimize_torch_BM

def gene_expression_plasticity(
    tree_files, gene_files, library_files, outfile, device, batch_size,
    max_iter, learning_rate, wandb_flag, window, tol, model="nb",
    approx="softplus_MC", const=False, resume=False
):
    if wandb_flag:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name=wandb_flag)

    # process data
    logger.info("Preprocessing data")
    (
        tree_list, cells_list, df_list, share_list_torch, library_list
    ) = process_data_BM(tree_files, gene_files, library_files, device)

    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=torch.float32, device=device) for lib in library_list
    ]  # list of (n_cells,)

    if batch_size > len(df_list[0].columns):
        batch_size = len(df_list[0].columns)

    n_genes = len(df_list[0].columns)
    n_batches = (n_genes + batch_size - 1) // batch_size

    # Checkpoint directory for per-batch results
    ckpt_dir = outfile + ".ckpt"
    if resume and os.path.isdir(ckpt_dir):
        logger.info(f"Found checkpoint directory: {ckpt_dir}")
    elif resume:
        logger.info(f"No checkpoint directory found, starting from scratch")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Running model: {n_genes} genes, {n_batches} batches of {batch_size}")
    # process gene expression data in batches
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

        x_list = [df[gene_names].values.T for df in df_list] # (batch, cells)
        x_tensor_list = [
            torch.tensor(x, dtype=torch.float32, device=device)
            for x in x_list
        ]

        bm_params_init = torch.ones(
            (actual_batch, ), dtype=torch.float32, device=device
        ) * (-2 if model in ("nb", "pois") else 1)

        if model in ("nb", "pois"):
            nb = (model == "nb")

            # init parameters for variational optimization
            s_init_tensor = [
                x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
                for x in x_tensor_list
            ]  # list of (actual_batch, n_cells)
            q_params_init = [
                torch.cat((x_tensor_list[i], s_init_tensor[i]), dim=-1)
                for i in range(len(x_tensor_list))
            ]

            log_r = torch.zeros(
                (actual_batch, ), dtype=torch.float32, device=device
            )  # init r=exp(0)=1 (moderate overdispersion)

            init_params = q_params_init + [log_r] + [bm_params_init]

            # optimize star tree
            star_params, star_loss = Lq_optimize_torch_BM(
                init_params,
                0,  # lambda = 0 for star tree
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

            # optimize lambda tree
            lambda_init_params = star_params[:-1] + [bm_params_init.clone().detach()]
            lambda_params, lambda_loss = Lq_optimize_torch_BM(
                lambda_init_params,
                2,  # mode = 2 for lambda tree
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

            # compute LRT statistic and p-value
            lr_stat = 2 * (star_loss - lambda_loss)
            chi2_dist = Chi2(torch.tensor([1.0], device=device))
            lr_stat_safe = torch.nan_to_num(lr_stat, nan=0.0).clamp(min=0)
            p_value = 1.0 - chi2_dist.cdf(lr_stat_safe)

            result = torch.cat((
                torch.cat((star_params[-2].exp().unsqueeze(-1), star_params[-1]), dim=-1),
                torch.cat((lambda_params[-2].exp().unsqueeze(-1), lambda_params[-1]), dim=-1),
                star_loss.unsqueeze(-1),
                lambda_loss.unsqueeze(-1),
                lr_stat.unsqueeze(-1),
                p_value.unsqueeze(-1)
            ), dim=-1)

        else:  # model == "bm"
            # optimize star tree
            star_params, star_loss = optimize_torch_BM(
                bm_params_init.clone(),
                0,  # lambda = 0 for star tree
                x_tensor_list,
                gene_names,
                share_list_torch,
                max_iter,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                const,
            )

            # optimize lambda tree
            lambda_params, lambda_loss = optimize_torch_BM(
                bm_params_init.clone(),
                2,  # mode = 2 for lambda tree
                x_tensor_list,
                gene_names,
                share_list_torch,
                max_iter,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                const,
            )

            # compute LRT statistic and p-value
            lr_stat = 2 * (star_loss - lambda_loss)
            chi2_dist = Chi2(torch.tensor([1.0], device=device))
            p_value = 1.0 - chi2_dist.cdf(lr_stat.clamp(min=0))

            result = torch.cat((
                star_params,
                lambda_params,
                star_loss.unsqueeze(-1),
                lambda_loss.unsqueeze(-1),
                lr_stat.unsqueeze(-1),
                p_value.unsqueeze(-1)
            ), dim=-1)

        # Save checkpoint: results + q params for this batch
        ckpt_data = {
            'genes': list(gene_names),
            'results': result.detach().cpu(),
        }
        torch.save(ckpt_data, ckpt_file)
        logger.info(f"Batch {batch_idx+1}/{n_batches} done ({batch_end}/{n_genes} genes)")

    # Load all batch checkpoints and combine
    results_list = []
    genes = []
    for batch_idx in range(n_batches):
        ckpt_file = os.path.join(ckpt_dir, f"batch_{batch_idx}.pt")
        ckpt_data = torch.load(ckpt_file, weights_only=False)
        results_list.append(ckpt_data['results'])
        genes.extend(ckpt_data['genes'])

    results = torch.cat(results_list, dim=0)

    # save results
    if model in ("nb", "pois"):
        columns = [
            "h0_r", "h0_lambda", "h0_mu", "h0_sigma",
            "h1_r", "h1_lambda", "h1_mu", "h1_sigma",
            "h0", "h1", "lr", "p"
        ]
    else:
        columns = [
            "h0_lambda", "h0_mu", "h0_sigma",
            "h1_lambda", "h1_mu", "h1_sigma",
            "h0", "h1", "lr", "p"
        ]

    results_df = pd.DataFrame(results.cpu().numpy(), columns=columns)
    results_df.insert(0, "gene", genes)
    results_df.insert(0, "ID", range(1, len(results_df) + 1))

    # Benjamini-Hochberg FDR correction
    results_df["signif"], results_df["q"] = multipletests(results_df["p"], alpha=0.05, method="fdr_bh")[:2]
    results_df = results_df.sort_values("q")  # sort by adjusted p-value

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    results_df.to_csv(outfile, sep="\t", index=False)

    # Clean up checkpoint directory
    import shutil
    shutil.rmtree(ckpt_dir)
    logger.info(f"Results saved to {outfile}")


def run_plasticity_test():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Testing gene expression correlation with tree by Pagel's lambda and LRT")
    parser.add_argument("--tree", required=True, type=str, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, type=str, help="Path to cell by gene expression data (TSV)")
    parser.add_argument("--library", required=False, type=str, default=None, help="Path to library file (TSV)")
    parser.add_argument("--outfile", required=False, type=str, default="./plasticity.tsv", help="Path to save the results (TSV)")
    parser.add_argument("--model", required=False, type=str, choices=["bm", "pois", "nb"], default="nb", help="Model type: 'bm' (pure Brownian Motion), 'pois' (OU-Poisson), or 'nb' (OU-Negative Binomial, default)")
    parser.add_argument("--batch", required=False, type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--iter", required=False, type=int, default=10000, help="Max optimization iterations")
    parser.add_argument("--lr", required=False, type=float, default=1e-1, help="Learning rate for optimization")
    parser.add_argument("--wandb", required=False, type=str, default=None, help="Wandb run name")
    parser.add_argument("--window", required=False, type=int, default=200, help="Window size for checking convergence")
    parser.add_argument("--tol", required=False, type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--approx", required=False, type=str, default="softplus_MC", help="How to approximate likelihood computation (nb/pois models only)")
    parser.add_argument("--const", required=False, action="store_true", help="Whether to keep BM parameters constant during optimization")
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
    if args.library is not None:
        library = args.library.split(",")
    else:
        library = [None] * len(tree)

    gene_expression_plasticity(
        tree, expression, library, args.outfile, device=device, batch_size=args.batch,
        max_iter=args.iter, learning_rate=args.lr, wandb_flag=args.wandb,
        window=args.window, tol=args.tol, model=args.model, approx=args.approx,
        const=args.const, resume=args.resume
    )

    elapsed = time.time() - start_time
    logger.info(f"Total running time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    run_plasticity_test()
