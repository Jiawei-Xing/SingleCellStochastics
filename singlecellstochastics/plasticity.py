import argparse
import pandas as pd
import torch
from scipy.stats import chi2
import wandb

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

    # process gene expression data in batches
    for batch_start in range(0, len(df_list[0].columns), batch_size):
        # gene expression batch
        batch_end = min(batch_start + batch_size, len(df_list[0].columns))
        gene_names = df_list[0].columns[batch_start:batch_end]
        x_list = [df[gene_names].values.T for df in df_list] # (batch, cells)
        x_tensor_list = [
            torch.tensor(x, dtype=torch.float32, device=device) 
            for x in x_list
        ]

        # init parameters for optimization
        m_init_tensor = [
            torch.tensor(m, dtype=torch.float32, device=device) for m in x_list
        ]  # list of (batch_size, n_cells)
        s_init_tensor = [
            torch.tensor(s, dtype=torch.float32, device=device) for s in x_list
        ]  # list of (batch_size, n_cells)
        q_params_init = [
            torch.cat((m_init_tensor[i], s_init_tensor[i]), dim=-1) for i in range(len(m_init_tensor))
        ]

        log_r = torch.ones(
            (batch_size, 1), dtype=torch.float32, device=device
        ) * 5  # (batch_size, 1), init from large r=exp(5) ~ 148.4

        bm_params_init = torch.ones(
            (batch_size, 3), dtype=torch.float32, device=device
        )  # (root_mean, pagel_lambda, sigma)

        init_params = q_params_init + [log_r] + [bm_params_init]

        results = None
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
            const
        )

        # optimize lambda tree
        lambda_params, lambda_loss = Lq_optimize_torch_BM(    
            star_params,
            3,  # mode = 3 for lambda tree
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

        # compute LRT statistic and p-value
        lr_stat = 2 * (star_loss - lambda_loss)  # likelihood ratio statistic
        p_value = 1 - chi2.cdf(lr_stat, df=1)  # degrees of freedom = 1 for lambda vs star
        
        # save results for this batch
        result = torch.cat((
            torch.tensor(gene_names, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.cat((star_params[-2].exp(), star_params[-1]), dim=-1),
            torch.cat((lambda_params[-2].exp(), lambda_params[-1]), dim=-1),
            star_loss.unsqueeze(-1),
            lambda_loss.unsqueeze(-1),
            lr_stat.unsqueeze(-1),
            p_value.unsqueeze(-1)
        ), dim=-1) # (batch, cols)
        results = result if results is None else torch.cat((results, result), dim=0)

    # save results    
    results_df = pd.DataFrame(
        results.cpu().numpy(),
        columns=[
            "gene_name",
            "h0_r", "h0_mean", "h0_lambda", "h0_sigma",
            "h1_r", "h1_mean", "h1_lambda", "h1_sigma",
            "h0_loss", "h1_loss", "lr_stat", "p_value"
        ]
    )
    results_df.to_csv(outfile, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing gene expression correlation with tree by Pagel's lambda and LRT")
    parser.add_argument("--tree", required=True, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, help="Path to cell by gene expression data (TSV)")
    parser.add_argument("--library", required=True, help="Path to library file (TSV)")
    parser.add_argument("--outfile", required=False, default="plasticity.tsv", help="Path to save the results (TSV)")
    parser.add_argument("--batch_size", required=False, type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max_iter", required=False, type=int, default=10000, help="Maximum number of optimization iterations")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-1, help="Learning rate for optimization")
    parser.add_argument("--wandb_flag", required=False, type=str, default=None, help="Whether to log optimization process to Weights & Biases (provide a name for the run if True)")
    parser.add_argument("--window", required=False, type=int, default=200, help="Window size for checking convergence")
    parser.add_argument("--tol", required=False, type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--approx", required=False, type=str, default="softplus_MC", help="How to approximate likelihood computation")
    parser.add_argument("--no_nb", action="store_false", default=10, help="Use poisson instead of negative binomial (default: use negative binomial)")
    parser.add_argument("--const", required=False, action="store_true", help="Whether to keep BM parameters constant during optimization")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gene_expression_plasticity(
        args.tree, args.expression, args.library, args.outfile, device=device, batch_size=args.batch_size,
        max_iter=args.max_iter, learning_rate=args.learning_rate, wandb_flag=args.wandb_flag, window=args.window, tol=args.tol, approx=args.approx, nb=args.no_nb, const=args.const
    )
