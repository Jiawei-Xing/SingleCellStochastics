import torch
import numpy as np
import pandas as pd
import argparse
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import wandb

from .preprocess import process_data
from .lrt import likelihood_ratio_test
from .simulate import simulate_null_all, simulate_null_each


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
        help="Newick tree file (comma-separated if multiple clones)",
    )
    parser.add_argument(
        "--expr",
        type=str,
        required=True,
        help="Cell by gene count matrix (comma-separated if multiple clones)",
    )
    parser.add_argument(
        "--annot", type=str, default=None, help="Gene annotation file (optional)"
    )
    parser.add_argument(
        "--regime",
        type=str,
        required=True,
        help="Regime file (comma-separated if multiple clones)",
    )
    parser.add_argument(
        "--null", type=str, required=True, help="Regime for null hypothesis"
    )
    parser.add_argument("--output", type=str, default="output", help="Output file")
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Number of genes for batch processing. Will be adjusted to the number of genes if input value is larger.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=500,
        help="Max number of iterations for optimization",
    )
    parser.add_argument(
        "--sim1",
        type=int,
        default=None,
        help="Number of simulations for empirical null distribution (one distribution for all genes)",
    )
    parser.add_argument(
        "--sim2",
        type=int,
        default=None,
        help="Number of simulations for empirical null distribution (one distribution for each gene)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Flag to enable logging to wandb (default: False)",
    )
    args = parser.parse_args()

    tree_files = args.tree.split(",")
    gene_files = args.expr.split(",")
    regime_files = args.regime.split(",")
    batch_size = args.batch
    rnull = args.null
    learning_rate = args.lr
    max_iter = args.iter
    N_sim_all = args.sim1
    N_sim_each = args.sim2
    annot_file = args.annot
    output_file = args.output
    wandb_flag = args.wandb

    if wandb_flag:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name="OUP")

    results = {}
    results_empirical = {}
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
            flush=True,
        )

        # pseudocount and init params
        x_original = [
            df[batch_genes].values.T for df in df_list
        ]  # list of (batch_size, n_cells)
        x_original = [
            np.expand_dims(x, axis=1) for x in x_original
        ]  # list of (batch_size, 1, n_cells)

        # likelihood ratio test (default)
        h0_params, h0_loss, h1_params, h1_loss = likelihood_ratio_test(
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
            learning_rate=learning_rate,
            max_iter=max_iter,
            device=device,
            wandb_flag=wandb_flag,
        )  # (batch_size, 1, ...)
        lr = h0_loss - h1_loss  # substract -log likelihood
        p_value = 1 - chi2.cdf(lr.flatten(), n_regimes - 1)

        # for each gene in batch (chi-squared test)
        for i in range(batch_size):
            h0_theta = np.log1p(np.exp(h0_params[i, 0, -n_regimes]))
            h1_theta = np.log1p(np.exp(h1_params[i, 0, -n_regimes:]))
            #h0_theta = np.exp(h0_params[i, 0, -n_regimes])
            #h1_theta = np.exp(h1_params[i, 0, -n_regimes:])
            result = (
                [batch_start + i, batch_genes[i], h0_theta]
                + h1_theta.tolist()
                + [h0_loss[i, 0], h1_loss[i, 0], lr[i, 0], p_value[i]]
            )
            results[batch_start + i] = result
            print("default result: " + "\t".join(list(map(str, result))))

        # collect all ou params
        ou_params_all.append(h0_params[:, 0, :])  # (batch_size, ...)
        lr_all.append(lr[:, 0])  # (batch_size,)

        # empirical null distribution for each gene TODO: sim for each tree separately
        if N_sim_each:
            x_original = simulate_null_each(
                tree_list, h0_params, N_sim_each, cells
            )  # (batch_size, N_sim, n_cells)
            _, h0_loss_sim, _, h1_loss_sim = likelihood_ratio_test(
                x_original,
                batch_gene_names,
                learning_rate=learning_rate,
                max_iter=max_iter,
                device=device,
            )  # (batch_size, N_sim, ...)
            null_LRs = h0_loss_sim - h1_loss_sim  # (batch_size, N_sim)

            # for each gene in batch (using empirical)
            for i in range(batch_size):
                p_empirical = (sum(null_LRs[i, :] >= lr[i, 0]) + 1) / (
                    len(null_LRs[i, :]) + 1
                )
                result = results[batch_start + i][:-1] + [p_empirical]
                results_empirical[batch_start + i] = result
                print("sim_gene result: " + "\t".join(list(map(str, result))))

    # FDR by Benjamini-Hochberg procedure
    results = sorted(list(results.values()), key=lambda x: x[-1])
    p_values = [r[-1] for r in results]
    signif, q_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[:2]

    with open(output_file + "_chi-squared.tsv", "w") as f:
        f.write(
            f"ID\tgene\th0_theta\t"
            + "\t".join(["h1_theta" + r for r in regimes])
            + f"\th0\th1\tLR\tp\tq\tsignif\n"
        )
        for i in range(len(results)):
            output = "\t".join(list(map(str, results[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()

    # using empirical for each gene TODO: sim for each tree separately
    if N_sim_each:
        results_empirical = sorted(
            list(results_empirical.values()), key=lambda x: x[-1]
        )
        p_values = [r[-1] for r in results_empirical]
        signif, q_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[:2]

        with open(output_file + "_empirical_each.tsv", "w") as f:
            f.write(
                f"ID\tgene\th0_theta\t"
                + "\t".join(["h1_theta" + r for r in regimes])
                + f"\th0\th1\tLR\tp\tq\tsignif\n"
            )
            for i in range(len(results_empirical)):
                output = "\t".join(list(map(str, results_empirical[i])))
                f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
                f.flush()

    # # empirical null distribution for all genes TODO: sim for each tree separately TODO: output sims
    if N_sim_all:
        # simulate null for all genes
        ou_params_all = np.concatenate(ou_params_all, axis=0)  # (gene_num, ...)
        x_original = simulate_null_all(
            tree_list, ou_params_all, N_sim_all, cells
        )  # (N_sim_all, n_cells)

        # empirical null distribution for all genes
        x_original = np.expand_dims(
            x_original, axis=1
        )  # shape: (N_sim_all, 1, n_cells)
        gene_names = ["sim_all"] * N_sim_all  # (N_sim_all,)
        _, h0_loss_sim, _, h1_loss_sim = likelihood_ratio_test(
            x_original,
            gene_names,
            learning_rate=learning_rate,
            max_iter=max_iter,
            device=device,
        )  # (N_sim_all, 1, ...)
        null_LRs = h0_loss_sim[:, 0] - h1_loss_sim[:, 0]  # (N_sim_all,)

        # LRT using empirical null distribution
        lr_all = np.concatenate(lr_all, axis=0)  # (gene_num,)
        for i in range(lr_all.shape[0]):
            p_empirical_all = (sum(null_LRs >= lr_all[i]) + 1) / (len(null_LRs) + 1)
            result = results[i][:-1] + [p_empirical_all]
            results_empirical_all[i] = result

        results_empirical_all = sorted(
            list(results_empirical_all.values()), key=lambda x: x[-1]
        )
        p_values = [r[-1] for r in results_empirical_all]
        signif, q_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[:2]

        with open(output_file + "_empirical_all.tsv", "w") as f:
            f.write(
                f"ID\tgene\th0_theta\t"
                + "\t".join(["h1_theta" + r for r in regimes])
                + f"\th0\th1\tLR\tp\tq\tsignif\n"
            )
            for i in range(len(results_empirical_all)):
                output = "\t".join(list(map(str, results_empirical_all[i])))
                f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
                f.flush()


if __name__ == "__main__":
    run_ou_poisson()
