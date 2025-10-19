import torch
import numpy as np
import pandas as pd
import argparse
import wandb
import os
import pickle

from .preprocess import process_data
from .lrt import likelihood_ratio_test
from .simulate import simulate_null_all, simulate_null_each
from .output import save_result, output_results

def run_ou_poisson():
    # seed
    torch.manual_seed(42)  # For GPU version
    np.random.seed(42)  # For CPU version

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # input parameters
    parser = argparse.ArgumentParser(
        description="Stochastic OUP model for gene expression evolution"
    )
    parser.add_argument(
        "--tree",
        type=str,
        required=True,
        help="Newick tree file (comma-separated if multiple clones)"
    )
    parser.add_argument(
        "--expr",
        type=str,
        required=True,
        help="Cell by gene count matrix (comma-separated if multiple clones)"
    )
    parser.add_argument(
        "--annot", type=str, default=None, help="Gene annotation file (optional)"
    )
    parser.add_argument(
        "--regime",
        type=str,
        required=True,
        help="Regime file (comma-separated if multiple clones)"
    )
    parser.add_argument(
        "--null", type=str, required=True, help="Regime for null hypothesis"
    )
    parser.add_argument(
        "--outdir", type=str, default="./", help="Output directory (default: ./)"
    )
    parser.add_argument(
        "--prefix", type=str, default="result", help="Prefix for output files (default: result)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Number of genes for batch processing. Must not be larger than the total number of genes. (default: 1000)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-1, help="Learning rate for Adam optimizer (default: 1e-1)"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=10000,
        help="Max number of iterations for optimization (default: 10,000)"
    )
    parser.add_argument(
        "--window", type=int, default=200, help="Number of iterations to check convergence (default: 200)"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--sim_all",
        type=int,
        default=None,
        help="Number of simulations for empirical null distribution (one distribution for all genes)"
    )
    parser.add_argument(
        "--sim_each",
        type=int,
        default=None,
        help="Number of simulations for empirical null distribution (one distribution for each gene)"
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Flag to enable using wandb with this name (default: None)"
    )
    parser.add_argument(
        "--approx",
        type=str,
        default="softplus_MC",
        help="Approximation method for Poisson likelihood expectation (default: softplus_MC)"
    )
    parser.add_argument(
        "--em_iter",
        type=int,
        default=0,
        help="Number of EM iterations (default: 0, optimize all params together)"
    )
    parser.add_argument(
        "--pseudo",
        type=float,
        default=0,
        help="Pseudo count for reverse softplus read counts as inital mean (default: 0, not reverse)"
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=1.0,
        help="L2 regularization strength for log alpha, aka precision of the Gaussian prior (default: 1.0)"
    )
    parser.add_argument(
        "--init", action="store_true", help="Initial run with OU model (default: False)"
    )
    parser.add_argument(
        "--no_kkt", action="store_false", help="Not use KKT condition to constrain OU parameters (default: True)"
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=0,
        help="Grid search range for alpha with fixed other parameters (default: 0, no grid search)"
    )
    args = parser.parse_args()

    tree_files = args.tree.split(",")
    gene_files = args.expr.split(",")
    regime_files = args.regime.split(",")
    batch_size = args.batch
    rnull = args.null
    learning_rate = args.lr
    max_iter = args.iter
    window = min(args.window, args.iter)
    tol = args.tol
    N_sim_all = args.sim_all
    N_sim_each = args.sim_each
    annot_file = args.annot
    output_dir = args.outdir
    prefix = args.prefix
    wandb_flag = args.wandb
    approx = args.approx
    em_iter = args.em_iter
    pseudo = args.pseudo
    prior = args.prior
    init = args.init
    kkt = args.no_kkt
    grid = args.grid

    if wandb_flag:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name=wandb_flag)

    results = {}
    results_empirical_each = {}
    results_empirical_all = {}
    ou_params_all = []
    lr_all = []

    # process data
    (
        tree_list,
        cells_list,
        df_list,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        diverge_list_torch,
        share_list_torch,
        epochs_list_torch,
        beta_list_torch,
        regime_list,
    ) = process_data(tree_files, gene_files, regime_files, rnull, device=device)

    regimes = list(
        dict.fromkeys(x for sub in regime_list for x in sub)
    )  # regimes by order
    n_regimes = len(regimes)

    # Make sure batch size is not larger than total gene number
    if batch_size > len(df_list[0].columns):
        batch_size = len(df_list[0].columns)

    # loop over gene batches
    for batch_start in range(0, len(df_list[0].columns), batch_size):
        # gene expression in tissues (batch)
        batch_genes = df_list[0].columns[batch_start : batch_start + batch_size]

        # gene names
        if annot_file:
            df_annot = pd.read_csv(annot_file, sep="\t", index_col=0, header=None)
            batch_gene_names = df_annot.loc[batch_genes, 0].tolist()
        else:
            batch_gene_names = batch_genes
        print(
            f"\ngene batch {batch_start}-{batch_start+len(batch_genes)-1}: {list(batch_gene_names)}",
            flush=True
        )

        # expression data
        x_original = [
            df[batch_genes].values.T for df in df_list
        ]  # list of (batch_size, n_cells)
        x_original = [
            np.expand_dims(x, axis=1) for x in x_original
        ]  # list of (batch_size, 1, n_cells)

        # likelihood ratio test (default)
        h0_params, h0_loss, h1_params, h1_loss = likelihood_ratio_test(
            #ou_params_h0, ou_loss_h0, ou_params_h1, ou_loss_h1 = \
            x_original,
            n_regimes,
            diverge_list,
            share_list,
            epochs_list,
            beta_list,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            batch_gene_names,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            em_iter,
            pseudo,
            batch_start,
            prior,
            init,
            kkt,
            grid
        )  # (batch_size, 1, ...)

        # save result
        results = save_result(batch_start, batch_size, batch_genes, \
            h0_params, h1_params, h0_loss, h1_loss, results, approx)

        # collect all ou params and lr
        ou_params_all.append(h0_params[:, 0, :])  # (batch_size, ...)
        lr = h0_loss - h1_loss  # (batch_size, 1)
        lr_all.append(lr[:, 0])  # (batch_size,)

        # empirical null distribution for each gene
        if N_sim_each:
            x_original = simulate_null_each(
                tree_list, h0_params, N_sim_each, cells_list
            )  # list of (batch_size, N_sim, n_cells)
            _, h0_loss_sim, _, h1_loss_sim = likelihood_ratio_test(
                x_original,
                n_regimes,
                diverge_list,
                share_list,
                epochs_list,
                beta_list,
                diverge_list_torch,
                share_list_torch,
                epochs_list_torch,
                beta_list_torch,
                batch_gene_names,
                max_iter,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                approx,
                em_iter,
                pseudo,
                batch_start,
                prior,
                init,
                kkt,
                grid
            )  # (batch_size, N_sim, ...)
            null_LRs = h0_loss_sim - h1_loss_sim  # (batch_size, N_sim)

            # for each gene in batch (using empirical)
            for i in range(batch_size):
                p_empirical = (sum(null_LRs[i, :] >= lr[i, 0]) + 1) / (
                    len(null_LRs[i, :]) + 1
                )
                result = results[batch_start + i][:-1] + [p_empirical]
                results_empirical_each[batch_start + i] = result

    # output results
    output_results(results, output_dir + prefix + "_chi-squared.tsv", regimes)
    
    # using empirical for each gene
    if N_sim_each:
        output_results(results_empirical_each, output_dir + prefix + "_empirical-each.tsv", regimes)

    # empirical null distribution for all genes
    if N_sim_all:
        print(f"\nUsing empirical null distribution for all genes ({N_sim_all} simulations)")
        # load cached parameters
        null_LRs = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            sim_all_cache_file = os.path.join(output_dir, f"sim_all_null_LRs_{N_sim_all}.pkl")

            # Try to load cached parameters
            if os.path.exists(sim_all_cache_file):
                try:
                    with open(sim_all_cache_file, 'rb') as f:
                        null_LRs = pickle.load(f)
                    print(f"\nLoaded cached null LRs from {sim_all_cache_file}")
                except Exception as e:
                    print(f"\nFailed to load cached parameters: {e}")

        # no cached sim_all
        if null_LRs is None:
            # simulate null for all genes
            ou_params_all = np.concatenate(ou_params_all, axis=0)  # (gene_num, ...)
            x_original = simulate_null_all(
                tree_list, ou_params_all, N_sim_all, cells_list
            )  # list of (N_sim_all, n_cells)

            # empirical null distribution for all genes
            x_original = [np.expand_dims(x, axis=1) for x in x_original]  # list of (N_sim_all, 1, n_cells)
            gene_names = ["sim_all"] * N_sim_all  # (N_sim_all,)
            _, h0_loss_sim, _, h1_loss_sim = likelihood_ratio_test(
                x_original,
                n_regimes,
                diverge_list,
                share_list,
                epochs_list,
                beta_list,
                diverge_list_torch,
                share_list_torch,
                epochs_list_torch,
                beta_list_torch,
                gene_names,
                max_iter,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                approx,
                em_iter,
                pseudo,
                batch_start,
                prior,
                init,
                kkt,
                grid
            )  # (N_sim_all, 1, ...)
            null_LRs = h0_loss_sim[:, 0] - h1_loss_sim[:, 0]  # (N_sim_all,)

            # save null LRs
            if output_dir is not None:
                with open(sim_all_cache_file, 'wb') as f:
                    pickle.dump(null_LRs, f)
        
        # LRT using empirical null distribution
        lr_all = np.concatenate(lr_all, axis=0)  # (gene_num,)
        for i in range(lr_all.shape[0]):
            p_empirical_all = (sum(null_LRs >= lr_all[i]) + 1) / (len(null_LRs) + 1)
            result = results[i][:-1] + [p_empirical_all]
            results_empirical_all[i] = result

        # output results
        output_results(results_empirical_all, output_dir + prefix + "_empirical-all.tsv", regimes)


if __name__ == "__main__":
    run_ou_poisson()
