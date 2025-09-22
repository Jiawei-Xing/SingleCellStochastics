import sys
import io
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import torch
from Bio import Phylo

from .tree_utils import (
    read_tree,
    assign_nodes_to_regimes_from_file,
    assign_nodes_to_null_regimes,
    add_read_counts_to_tips,
)
from .input_output import load_read_count_tsv
from .optimization import adam_optimize_ou_parameters
from .stat_utils import calculate_lrt_and_pvalue
import contextlib


def process_gene(
    gene: str,
    read_count_data: dict,
    null_tree: Phylo.BaseTree.Tree,
    alt_tree: Phylo.BaseTree.Tree,
    origin_expression: torch.Tensor,
    null_regime: str,
    poisson_logl_mode: str,
    output_dir: str = None,
) -> None:
    """
    Process a single gene: fit null and alt OU models, compute LRT.

    Args:
        gene: Gene name (str).
        read_count_data: Dictionary tips to read counts (dict).
        null_tree: Biopython `Tree` object for the null hypothesis (all nodes in one regime).
        alt_tree: Biopython `Tree` object for the alternative hypothesis (nodes in multiple regimes).
        origin_expression: Expression value assumed at the origin of the experiment (torch.Tensor).
        null_regime: Regime label for all nodes under the null hypothesis (str).
        poisson_logl_mode: Mode for Poisson sampling, either "deterministic", "stochastic", or "variational".
        output_dir: Directory to save log files (str).

    Returns:
        None
    """
    log_path = f"{output_dir}/{gene}.txt"
    with open(log_path, "w") as f:
        f.write(f"\nProcessing gene {gene}")

    add_read_counts_to_tips(null_tree, read_count_data)
    add_read_counts_to_tips(alt_tree, read_count_data)

    alpha_init = torch.tensor(1.0, dtype=torch.float32)
    sigma_init = torch.tensor(1.0, dtype=torch.float32)
    theta_dict_init = {
        "0": torch.tensor(2.0, dtype=torch.float32),
        "1": torch.tensor(5.0, dtype=torch.float32),
    }
    
    # # Jiawei test run
    # alpha_init = torch.tensor(5.943188, dtype=torch.float32)
    # sigma_init = torch.tensor(20.95007494, dtype=torch.float32)
    # theta_dict_init = {
    #     "0": torch.tensor(0.9873305559158325, dtype=torch.float32),
    #     "1": torch.tensor(0.5626604557037354, dtype=torch.float32),
    # }
      
    theta_dict_init_null = {
        k: v.clone() for k, v in theta_dict_init.items() if k == null_regime
    }

    # Fit null model
    with open(log_path, "a") as f:
        f.write("\n\nFitting null model...\n")
    (
        null_optimal_negll,
        null_optimal_alpha,
        null_optima_sigma2,
        null_optimal_theta_dict,
    ) = adam_optimize_ou_parameters(
        null_tree,
        alpha_init,
        sigma_init,
        theta_dict_init_null,
        origin_expression,
        log_path,
        poisson_logl_mode
    )

    # Fit alt model
    with open(log_path, "a") as f:
        f.write("\n\nFitting alternative model...\n")
    (
        alt_optimal_negll,
        alt_optimal_alpha,
        alt_optima_sigma2,
        alt_optimal_theta_dict,
    ) = adam_optimize_ou_parameters(
        alt_tree,
        alpha_init,
        sigma_init,
        theta_dict_init,
        origin_expression,
        log_path,
        poisson_logl_mode
    )

    # Compute LRT statistic
    null_log_lik = -null_optimal_negll.item()
    alt_log_lik = -alt_optimal_negll.item()
    lrt_statistic, p_value = calculate_lrt_and_pvalue(null_log_lik, alt_log_lik)

    with open(log_path, "a") as f:
        f.write(f"\n\nLRT statistic = {lrt_statistic}\n")
        f.write(f"p-value = {p_value}\n")


def run_lrt():

    parser = argparse.ArgumentParser(
        "Stochastic simulation of gene expression evolution along lineage"
    )
    parser.add_argument(
        "--tree", type=str, required=True, help="File path of input tree"
    )
    parser.add_argument(
        "--regime",
        type=str,
        required=True,
        help="File path of input regime that defined which theta each node belongs to in the OU model",
    )
    parser.add_argument(
        "--expression_data",
        type=str,
        required=True,
        help="File path of input TSV expression data in cells x genes format",
    )
    parser.add_argument(
        "--origin_expression", type=int, default=2, help="Starting expression assumed at the origin of the experiment"
    )
    parser.add_argument(
        "--null_regime",
        type=str,
        default="0",
        help="Regime label for all nodes under the null hypothesis",
    )
    parser.add_argument(
        "--poisson_logl_mode",
        type=str,
        choices=["none", "deterministic", "stochastic", "variational"],
        default="variational",
        help="Mode for logl calculation if transformation/poisson is added after the OU model, either 'none' (logl for only the OU model), 'deterministic', 'stochastic', or 'variational'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./examples/output_results",
        help="Output directory for log files",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for parallel processing",
    )
    args = parser.parse_args()

    # Save some args locally
    null_regime = args.null_regime
    poisson_logl_mode = args.poisson_logl_mode
    output_dir = args.output_dir
    threads = args.threads

    # Setup tensors
    origin_expression = torch.tensor(float(args.origin_expression), dtype=torch.float32)

    # Read in the null hypothesis tree without regimes
    null_tree = read_tree(args.tree)
    assign_nodes_to_null_regimes(null_tree, null_regime=args.null_regime)
    
    # Print the root node branch length
    root = null_tree.root

    # Read in the alternative hypothesis tree with regimes
    alt_tree = read_tree(args.tree)
    assign_nodes_to_regimes_from_file(alt_tree, args.regime)

    read_count_data = load_read_count_tsv(args.expression_data)
    num_genes = len(read_count_data.keys())

    genes = read_count_data.keys()
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                process_gene,
                gene,
                read_count_data[gene],
                copy.deepcopy(null_tree),
                copy.deepcopy(alt_tree),
                origin_expression,
                null_regime,
                poisson_logl_mode,
                output_dir,
            )
            for gene in genes
        ]
        for future in as_completed(futures):
            future.result()
