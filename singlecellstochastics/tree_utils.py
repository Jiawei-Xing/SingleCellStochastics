import csv
from typing import Dict, List, Tuple
from Bio import Phylo
import numpy as np


def read_tree(
    newick_file: str,
    normalize_branch_lengths: bool = True,
    name_unnamed_nodes: bool = False,
) -> Phylo.BaseTree.Tree:
    """
    Read a Newick tree, then optionally normalize branch lengths and assign internal node names.

    Args:
        newick_file (str): Path to the Newick-formatted tree file.
        normalize_branch_lengths (bool): If True, normalize branch lengths so that the longest root-to-tip path is 1.0.
        name_unnamed_nodes (bool): If True, assign unique names to unnamed internal nodes.

    Returns:
        Tree: A Biopython `Tree` object.

    """
    tree = Phylo.read(newick_file, "newick")
    total_length = max(tree.depths().values())

    # Normalize and name nodes
    i = 0
    for c in tree.find_clades():
        if normalize_branch_lengths:
            c.branch_length = (c.branch_length or 0) / total_length
        if c.name is None:
            if name_unnamed_nodes:
                c.name = f"node{i}"
                i += 1
            else:
                raise ValueError(
                    "All nodes must be named or set name_unnamed_nodes to True"
                )

    return tree


def assign_nodes_to_regimes_from_file(
    tree: Phylo.BaseTree.Tree, regime_file: str
) -> None:
    """
    Assign regime labels to nodes in the tree based on a CSV file mapping node names to regimes.
    Modifies in place.

    Args:
        tree (Tree): A Biopython `Tree` object.
        regime_file (str): Path to a CSV file with two columns: node name and regime label.

    Returns:
        None
    """
    # Directly match node names to regimes
    with open(regime_file) as f:
        reader = csv.reader(f)
        next(reader)
        regime_map = {row[0]: row[1] for row in reader}

    for clade in tree.find_clades():
        if clade.name in regime_map:
            clade.regime = str(regime_map[clade.name])
        else:
            raise ValueError(f"Clade {clade.name} not found in regime file")


def assign_nodes_to_null_regimes(tree: Phylo.BaseTree.Tree, null_regime="0") -> None:
    """
    Assign all nodes in the tree to a single null regime.
    Modifies in place.

    Args:
        tree (Tree): A Biopython `Tree` object.
        null_regime (str): The regime label to assign to all nodes.

    Returns:
        None
    """

    for clade in tree.find_clades():
        clade.regime = null_regime


def reset_all_nodes_expr(tree: Phylo.BaseTree.Tree) -> None:
    """
    Reset expression values for all nodes in the tree.

    Args:
        tree (Tree): A Biopython `Tree` object.

    Returns:
        None
    """
    for node in tree.find_clades():
        node.expr = None


def reset_all_nodes_read_counts(tree: Phylo.BaseTree.Tree) -> None:
    """
    Reset read counts for all nodes in the tree.

    Args:
        tree (Tree): A Biopython `Tree` object.

    Returns:
        None
    """
    for node in tree.find_clades():
        node.read_count = None


def add_read_counts_to_tips(tree: Phylo.BaseTree.Tree, read_count_dict: dict) -> None:
    """
    Adds read count values to tips in the tree.

    Args:
        tree (Tree): A Biopython `Tree` object.
        expr_dict (dict): A dictionary mapping tip names to expression values.

    Returns:
        None
    """
    for node in tree.get_terminals():
        if node.name in read_count_dict:
            node.read_count = read_count_dict.get(node.name, None)


def collect_tip_read_count_data(tree: Phylo.BaseTree.Tree) -> np.ndarray:
    """
    Extract read counts for all tips.

    Returns:
        y: array of observed read counts
    """
    y = np.array([tip.read_count for tip in tree.get_terminals()], dtype=float)
    return y
