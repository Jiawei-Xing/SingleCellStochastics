"""Name internal nodes in a VINE Newick tree for LAVOUS BM workflows."""

import argparse
from io import StringIO
import os

from Bio import Phylo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read the first Newick tree in a file, name unnamed internal nodes, and write it."
    )
    parser.add_argument("--newick", required=True, help="Input Newick file, typically VINE stdout.")
    parser.add_argument("--tree-out", required=True, help="Output Newick tree with named internal nodes.")
    return parser.parse_args()


def first_newick(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No Newick tree found in {path}")


def main():
    args = parse_args()
    tree = Phylo.read(StringIO(first_newick(args.newick)), "newick")
    idx = 0
    for clade in tree.find_clades(order="preorder"):
        if clade.clades and not clade.name:
            clade.name = f"internal_node{idx}"
            idx += 1

    os.makedirs(os.path.dirname(args.tree_out) or ".", exist_ok=True)
    Phylo.write(tree, args.tree_out, "newick")
    print(f"Saved clean tree -> {args.tree_out}")


if __name__ == "__main__":
    main()
