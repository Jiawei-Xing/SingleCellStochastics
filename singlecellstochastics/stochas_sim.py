import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import math
from Bio import Phylo

from .ornstein_uhlenbeck import get_ou_expr_one_branch
from .brownian_motion import get_bm_expr_one_branch
from .tree_utils import (
    read_tree,
    assign_nodes_to_regimes_from_file,
    reset_all_nodes_expr,
    reset_all_nodes_read_counts,
)
from .poisson import get_poisson_sampled_read_counts
from .transform import clamp_latent_gene_expression_at_tips
from .input_output import write_read_counts


def get_latent_gene_expression_at_tips(
    tree: Phylo.BaseTree.Tree,
    test_regime: str = None,
    root_expr: float = None,
    optim: float = None,
    alpha: float = None,
    sigma2: float = None,
    background_model: str = "OU",
) -> None:
    """
    Recursively simulate latent gene expression values at the tips of a phylogenetic tree under the OU or BM process.

    Args:
        tree: A Biopython `Tree` object.
        test_regime (str, optional): Regime label for which to apply the test OU process.
        root_expr (float, optional): Expression value at the root node.
        optim (float, optional): Optimal expression value (theta) for the OU process.
        alpha (float, optional): Selective strength parameter for the OU process.
        sigma2 (float, optional): Variance parameter for the OU/BM process.
        background_model (str, optional): Model for background lineages ("OU" or "BM").

    Returns:
        None
    """

    def set_expr(node, parent_expr):
        # If at root, set its expression to the root expression
        if node == tree.root:
            new_expr = root_expr
        else:
            # Make sure node has a regime assigned
            if not hasattr(node, "regime"):
                raise ValueError(f"Node {node.name} does not have a regime assigned.")

            # Simulate expression based on regime and model
            if node.regime == test_regime:
                new_expr = get_ou_expr_one_branch(
                    parent_expr, optim, alpha, node.branch_length, sigma2
                )
            elif background_model == "OU":
                new_expr = get_ou_expr_one_branch(
                    parent_expr, root_expr, alpha, node.branch_length, sigma2
                )
            elif background_model == "BM":
                new_expr = get_bm_expr_one_branch(
                    parent_expr, node.branch_length, sigma2
                )
            else:
                raise ValueError("background_model input must be OU or BM")

        node.expr = new_expr

        # Recurse to children
        for child in node.clades:
            set_expr(child, new_expr)

    # Start recursion from the root
    set_expr(tree.root, None)


# simulate gene expression by BM or OU
def simulate(
    tree: Phylo.BaseTree.Tree,
    n_genes: int,
    root_expr: float,
    test_regime: str,
    optim: float = None,
    alpha: float = None,
    sigma2: float = None,
) -> tuple[list[str], dict[int, dict[str, int]]]:
    """
    Simulate gene expression data for multiple genes along a phylogenetic tree.

    Args:
        tree (Tree): A Biopython `Tree` object with assigned regimes.
        n_genes (int): Number of genes to simulate.
        root_expr (float): Expression value at the root node.
        test_regime (str): Regime label for which to apply the test OU process.
        optim (float, optional): Optimal expression value (theta) for the OU process.
        alpha (float, optional): Selective strength parameter for the OU process.
        sigma2 (float, optional): Variance parameter for the OU/BM process.

    Returns:
        tuple: A tuple containing:
            - cells (list): List of cell (tip) names.
            - read_counts (dict): A dictionary where keys are gene indices and values are dictionaries
                mapping cell names to read counts.
    """
    read_counts = {}
    plots = []
    for i in range(n_genes):

        print(f"Simulating gene {i+1}/{n_genes}")

        # Clean the tree from any previous genes
        reset_all_nodes_expr(tree)
        reset_all_nodes_read_counts(tree)

        # Simulate latent gene expression at tips
        get_latent_gene_expression_at_tips(
            tree=tree,
            test_regime=test_regime,
            root_expr=root_expr,
            optim=optim,
            alpha=alpha,
            sigma2=sigma2,
            background_model="OU",
        )

        # Transform negative latent gene expression at tips
        clamp_latent_gene_expression_at_tips(tree)

        # Poisson sample read counts at tips
        get_poisson_sampled_read_counts(tree)

        # Store read counts for this gene before resetting the tree for the next gene
        read_counts[i] = {node.name: node.read_count for node in tree.get_terminals()}

    return read_counts


def run_stochas_sim():
    parser = argparse.ArgumentParser(
        "Stochastic simulation of gene expression evolution along lineage"
    )
    parser.add_argument(
        "--tree", type=str, required=True, help="File path of input tree"
    )
    parser.add_argument(
        "--regime", type=str, required=True, help="File path of input regime"
    )
    parser.add_argument("--test", type=str, default="1", help="Regime for testing (OU)")
    parser.add_argument(
        "--root", type=int, default=2, help="Starting expression at the root"
    )
    parser.add_argument(
        "--n_genes", type=int, default=5, help="Number of genes to simulate"
    )
    parser.add_argument("--sigma2", type=float, default=1, help="Variance for BM or OU")
    parser.add_argument(
        "--optim", type=int, default=5, help="Optimal expression for OU"
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="Selective strength for OU"
    )
    parser.add_argument(
        "--out", type=str, default="examples/input_data", help="Output directory"
    )
    parser.add_argument("--label", type=str, default="test", help="Label for output")
    args = parser.parse_args()

    tree = read_tree(args.tree)
    assign_nodes_to_regimes_from_file(tree, args.regime)

    read_counts = simulate(
        tree, args.n_genes, args.root, args.test, args.optim, args.alpha, args.sigma2
    )

    cells = [node.name for node in tree.get_terminals()]
    write_read_counts(read_counts, cells, args.n_genes, args.out, args.label)


if __name__ == "__main__":
    run_stochas_sim()
