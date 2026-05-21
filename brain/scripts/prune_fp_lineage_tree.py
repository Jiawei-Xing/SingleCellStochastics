#!/usr/bin/env python
"""Prune a clone tree to homogeneous FP-lineage leaves."""

import argparse
import csv
import os

import pandas as pd
from Bio import Phylo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune a homogeneous clone tree to cell types in an FP-lineage map."
    )
    parser.add_argument("--dataset", required=True, help="Dataset label, e.g. E15_rep1.")
    parser.add_argument("--tree", required=True, help="Input Newick tree.")
    parser.add_argument("--metadata", required=True, help="Clone metadata TSV.")
    parser.add_argument(
        "--fp-map",
        default="scripts/fp_lineage_groups.tsv",
        help="TSV with columns cell_type and fp_regime.",
    )
    parser.add_argument("--outdir", default="data/fp_lineage", help="Output directory.")
    return parser.parse_args()


def read_fp_map(path):
    with open(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["cell_type"]: row["fp_regime"] for row in reader}


def name_internal_nodes(tree):
    idx = 0
    for clade in tree.find_clades(order="preorder"):
        if clade.clades:
            clade.name = f"internal_node{idx}"
            idx += 1


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fp_map = read_fp_map(args.fp_map)
    metadata = pd.read_csv(args.metadata, sep="\t")
    keep = metadata["dominant_cell_type"].isin(fp_map)
    metadata = metadata.loc[keep].copy()
    metadata["fp_regime"] = metadata["dominant_cell_type"].map(fp_map)
    if metadata.empty:
        raise ValueError(f"No FP-lineage leaves found in {args.metadata}")

    keep_leaves = set(metadata["leaf_id"].astype(str))
    tree = Phylo.read(args.tree, "newick")
    for leaf in list(tree.get_terminals()):
        if leaf.name not in keep_leaves:
            tree.prune(leaf)
    tree.ladderize(reverse=True)
    name_internal_nodes(tree)

    prefix = f"{args.dataset}_fp_lineage_homogeneous_clone"
    tree_path = os.path.join(args.outdir, f"{prefix}_vine_tree.nwk")
    samples_path = os.path.join(args.outdir, f"{prefix}_vine_samples.nwk")
    metadata_path = os.path.join(args.outdir, f"{prefix}_metadata.tsv")
    labels_path = os.path.join(args.outdir, f"{prefix}_statelabels.csv")
    manifest_path = os.path.join(args.outdir, "fp_lineage_homogeneous_manifest.tsv")

    Phylo.write(tree, tree_path, "newick")
    Phylo.write(tree, samples_path, "newick")
    metadata.to_csv(metadata_path, sep="\t", index=False)
    metadata[["leaf_id", "fp_regime"]].to_csv(
        labels_path, index=False, header=False
    )

    manifest = pd.DataFrame(
        [
            {
                "dataset": f"{args.dataset}_fp_lineage_homogeneous",
                "unit": "clone",
                "matrix": "",
                "statelabels": labels_path,
                "metadata": metadata_path,
                "state_map": "",
                "n_leaves": len(tree.get_terminals()),
                "n_sites": "",
            }
        ]
    )
    manifest.to_csv(manifest_path, sep="\t", index=False)

    counts = metadata["fp_regime"].value_counts().sort_index()
    print(f"Saved {tree_path}")
    print(f"Saved {metadata_path}")
    print(f"Saved {labels_path}")
    print(f"Saved {manifest_path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
