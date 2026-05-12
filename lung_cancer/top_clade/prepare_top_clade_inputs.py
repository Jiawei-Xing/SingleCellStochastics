#!/usr/bin/env python
"""Prepare LAVOUS inputs for a top-clade sensitivity run."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.Newick import Tree


DEFAULT_META = (
    "/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/"
    "test_real_data/Yang_KP-Tracer/KPTracer_meta.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default="data/3724_NT_All_vine_tree.nwk")
    parser.add_argument("--regime", default="data/3724_NT_All_vine_regime.csv")
    parser.add_argument("--read-counts", default="data/3724_NT_All_readcounts.tsv")
    parser.add_argument("--library", default="data/3724_NT_All_library.tsv")
    parser.add_argument("--metadata", default=DEFAULT_META)
    parser.add_argument("--outdir", default="top_clade/data")
    parser.add_argument("--prefix", default="3724_NT_TopClade")
    parser.add_argument(
        "--target-node",
        default="auto",
        help=(
            "Internal node to extract, or 'auto' to choose the largest mixed "
            "Pri/Met clade mostly in the top half of the circular plot."
        ),
    )
    parser.add_argument("--min-leaves", type=int, default=100)
    parser.add_argument("--min-each-regime", type=int, default=100)
    parser.add_argument("--min-top-frac", type=float, default=0.95)
    return parser.parse_args()


def count_leaves(clade):
    if clade.is_terminal():
        return 1
    return sum(count_leaves(child) for child in clade.clades)


def ladderize(clade):
    for child in clade.clades:
        ladderize(child)
    clade.clades.sort(key=count_leaves, reverse=True)


def get_leaves(clade):
    return [leaf.name for leaf in clade.get_terminals()]


def get_nodes(clade):
    return [node.name for node in clade.find_clades() if node.name is not None]


def find_clade_by_name(clade, name):
    for node in clade.find_clades(order="preorder"):
        if node.name == name:
            return node
    raise ValueError(f"Node {name!r} was not found in the tree.")


def choose_top_clade(tree, regime):
    """Choose a large mixed clade from the top of the reconstruction plot."""
    work = copy.deepcopy(tree)
    ladderize(work.root)

    leaves = get_leaves(work.root)
    angles = dict(
        zip(leaves, np.linspace(0.0, 2.0 * np.pi, len(leaves), endpoint=False))
    )

    rows = []
    for clade in work.find_clades(order="preorder"):
        if clade.is_terminal():
            continue
        clade_leaves = get_leaves(clade)
        if len(clade_leaves) < choose_top_clade.min_leaves:
            continue

        counts = regime.loc[clade_leaves].value_counts()
        t1 = int(counts.get("T1", 0))
        met = int(counts.get("Met", 0))
        top_frac = sum(0.0 <= angles[name] < np.pi for name in clade_leaves) / len(
            clade_leaves
        )
        if (
            t1 < choose_top_clade.min_each_regime
            or met < choose_top_clade.min_each_regime
            or top_frac < choose_top_clade.min_top_frac
        ):
            continue

        rows.append(
            {
                "node": clade.name,
                "n_leaves": len(clade_leaves),
                "top_fraction": top_frac,
                "T1": t1,
                "Met": met,
            }
        )

    if not rows:
        raise ValueError(
            "No top-clade candidate passed the auto-selection criteria. "
            "Lower --min-top-frac/--min-each-regime, or pass --target-node."
        )

    candidates = pd.DataFrame(rows).sort_values(
        ["n_leaves", "top_fraction"], ascending=[False, False]
    )
    choose_top_clade.candidates = candidates
    return str(candidates.iloc[0]["node"])


def write_summary(path, target_node, leaves, regime, metadata):
    lines = [
        f"target_node\t{target_node}",
        f"n_leaves\t{len(leaves)}",
        "",
        "regime_counts",
        regime.loc[leaves].value_counts().to_string(),
    ]

    if metadata is not None:
        present = metadata.index.intersection(leaves)
        md = metadata.loc[present]
        for column in [
            "Tumor",
            "SubTumor",
            "Lane",
            "Batch_Library",
            "Batch_Harvest",
            "MULTI",
        ]:
            if column in md:
                lines.extend(["", f"{column}_counts", md[column].value_counts().to_string()])

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    choose_top_clade.min_leaves = args.min_leaves
    choose_top_clade.min_each_regime = args.min_each_regime
    choose_top_clade.min_top_frac = args.min_top_frac

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tree = Phylo.read(args.tree, "newick")
    regime_df = pd.read_csv(args.regime, sep=None, engine="python")
    if not {"node_name", "regime"}.issubset(regime_df.columns):
        raise ValueError("Regime file must contain node_name and regime columns.")
    regime = regime_df.set_index("node_name")["regime"].astype(str)

    target_node = args.target_node
    if target_node == "auto":
        target_node = choose_top_clade(tree, regime)
        print("Auto-selected top clade candidates:")
        print(choose_top_clade.candidates.head(10).to_string(index=False))

    target = find_clade_by_name(tree.root, target_node)
    subtree_root = copy.deepcopy(target)
    subtree_root.branch_length = 0.0
    subtree = Tree(root=subtree_root, rooted=True)
    leaves = get_leaves(subtree.root)
    nodes = set(get_nodes(subtree.root))

    missing_regime = sorted(set(leaves) - set(regime.index))
    if missing_regime:
        raise ValueError(f"{len(missing_regime)} leaves are missing regime labels.")

    tree_path = outdir / f"{args.prefix}_vine_tree.nwk"
    regime_path = outdir / f"{args.prefix}_vine_regime.csv"
    counts_path = outdir / f"{args.prefix}_readcounts.tsv"
    library_path = outdir / f"{args.prefix}_library.tsv"
    leaves_path = outdir / f"{args.prefix}_leaves.txt"
    summary_path = outdir / f"{args.prefix}_summary.txt"
    metadata_path = outdir / f"{args.prefix}_metadata.csv"

    Phylo.write(subtree, tree_path, "newick")
    regime_df.loc[regime_df["node_name"].astype(str).isin(nodes)].to_csv(
        regime_path, index=False
    )

    read_counts = pd.read_csv(args.read_counts, sep="\t", index_col=0)
    missing_counts = sorted(set(leaves) - set(read_counts.index))
    if missing_counts:
        raise ValueError(f"{len(missing_counts)} leaves are missing read counts.")
    read_counts.loc[leaves].to_csv(counts_path, sep="\t")

    library = pd.read_csv(args.library, sep="\t", index_col=0, header=None)
    missing_library = sorted(set(leaves) - set(library.index))
    if missing_library:
        raise ValueError(f"{len(missing_library)} leaves are missing library factors.")
    library.loc[leaves].to_csv(library_path, sep="\t", header=False)

    metadata = None
    if args.metadata and Path(args.metadata).exists():
        metadata = pd.read_csv(args.metadata).rename(columns={"Unnamed: 0": "cell"})
        metadata = metadata.set_index("cell")
        md = metadata.loc[metadata.index.intersection(leaves)].copy()
        md["regime"] = regime.loc[md.index]
        md.to_csv(metadata_path)

    leaves_path.write_text("\n".join(leaves) + "\n")
    write_summary(summary_path, target_node, leaves, regime, metadata)

    print(f"Wrote {len(leaves)}-leaf subtree for {target_node}")
    print(f"  tree: {tree_path}")
    print(f"  regime: {regime_path}")
    print(f"  read counts: {counts_path}")
    print(f"  library: {library_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
