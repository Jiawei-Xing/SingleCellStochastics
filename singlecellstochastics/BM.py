import argparse
from sys import prefix
import pandas as pd
import numpy as np
import torch
from torch.distributions.chi2 import Chi2
import wandb
from statsmodels.stats.multitest import multipletests
import os

from .preprocess import process_data_BM
from .optimize import Lq_optimize_torch_BM

def gene_expression_plasticity(
    tree_files, gene_files, library_files, outfile, device, batch_size,
    max_iter, learning_rate, wandb_flag, window, tol, approx, nb, const
):
    if wandb_flag:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name=wandb_flag)

    # process data
    (
        tree_list, cells_list, df_list, share_list_torch, library_list
    ) = process_data_BM(tree_files, gene_files, library_files, device)

    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=torch.float32, device=device) for lib in library_list
    ]  # list of (n_cells,)
    
    if batch_size > len(df_list[0].columns):
        batch_size = len(df_list[0].columns)

    results_list = []
    q_list = []
    genes = []
    # process gene expression data in batches
    for batch_start in range(0, len(df_list[0].columns), batch_size):
        # gene expression batch
        batch_end = min(batch_start + batch_size, len(df_list[0].columns))
        gene_names = df_list[0].columns[batch_start:batch_end]
        genes.extend(gene_names)
        x_list = [df[gene_names].values.T for df in df_list] # (batch, cells)
        x_tensor_list = [
            torch.tensor(x, dtype=torch.float32, device=device) 
            for x in x_list
        ]

        # init parameters for optimization
        s_init_tensor = [
            x.std(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(x)
            for x in x_tensor_list
        ]  # list of (batch_size, n_cells)
        q_params_init = [
            torch.cat((x_tensor_list[i], s_init_tensor[i]), dim=-1) 
            for i in range(len(x_tensor_list))
        ]

        log_r = torch.ones(
            (batch_size, 1), dtype=torch.float32, device=device
        ) * 5  # (batch_size, 1), init from large r=exp(5) ~ 148.4

        bm_params_init = torch.ones(
            (batch_size, 3), dtype=torch.float32, device=device
        )  # (root_mean, pagel_lambda, sigma)

        init_params = q_params_init + [log_r] + [bm_params_init]

        # optimize star tree
        star_params, star_loss = Lq_optimize_torch_BM(    
            init_params,
            1,  # lambda = 1 for BM
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
            const
        )
        
        # save results for this batch
        result = torch.cat((
            torch.cat((star_params[-2].exp(), star_params[-1]), dim=-1),
            star_loss.unsqueeze(-1),
        ), dim=-1) # (batch, cols)
        results_list.append(result.detach().cpu())

        q_list.append(star_params[:-2])  # list of (batch, cells) for q parameters

    # concatenate all batch results
    results = torch.cat(results_list, dim=0)
    q_params = [torch.cat(q_batch, dim=0) for q_batch in zip(*q_list)]  # list of (n_genes, cells) for q parameters

    # save results    
    results_df = pd.DataFrame(
        results.cpu().numpy(),
        columns=[
            "h0_r", "h0_mu", "h0_lambda", "h0_sigma", "h0"
        ]
    )
    results_df.insert(0, "gene", genes)
    results_df.insert(0, "ID", range(1, len(results_df) + 1))

    # output variational parameters
    for i in range(len(q_params)):
        df = pd.DataFrame(
            q_params[i].cpu().numpy(), #(batch_size, n_cells*2)
            index=df_list[i].columns,
            columns=cells_list[i]*2
        )
        df.to_csv(f"bm_q_params_{i}.tsv", sep="\t")
    
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    results_df.to_csv(outfile, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing gene expression correlation with tree by Pagel's lambda and LRT")
    parser.add_argument("--tree", required=True, type=str, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, type=str, help="Path to cell by gene expression data (TSV)")
    parser.add_argument("--library", required=False, type=str, default=None, help="Path to library file (TSV)")
    parser.add_argument("--outfile", required=False, type=str, default="./BM.tsv", help="Path to save the results (TSV)")
    parser.add_argument("--batch_size", required=False, type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max_iter", required=False, type=int, default=10000, help="Maximum number of optimization iterations")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-1, help="Learning rate for optimization")
    parser.add_argument("--wandb_flag", required=False, type=str, default=None, help="Whether to log optimization process to Weights & Biases (provide a name for the run if True)")
    parser.add_argument("--window", required=False, type=int, default=200, help="Window size for checking convergence")
    parser.add_argument("--tol", required=False, type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--approx", required=False, type=str, default="softplus_MC", help="How to approximate likelihood computation")
    parser.add_argument("--no_nb", required=False, action="store_false", help="Use poisson instead of negative binomial (default: use negative binomial)")
    parser.add_argument("--const", required=False, action="store_true", help="Whether to keep BM parameters constant during optimization")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tree = args.tree.split(",")
    expression = args.expression.split(",")
    if args.library is not None:
        library = args.library.split(",")
    else:        
        library = [None] * len(tree)
    
    gene_expression_plasticity(
        tree, expression, library, args.outfile, device=device, batch_size=args.batch_size,
        max_iter=args.max_iter, learning_rate=args.learning_rate, wandb_flag=args.wandb_flag, window=args.window, tol=args.tol, approx=args.approx, nb=args.no_nb, const=args.const
    )
