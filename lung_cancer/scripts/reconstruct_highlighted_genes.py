#!/usr/bin/env python
"""Reconstruct latent histories for volcano-highlighted Pri vs Met genes."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DIFF_RESULTS = Path("outputs/diff/3724_NT_All_diff_Pri-Met_empirical-all_10000.tsv")
TREE = Path("data/3724_NT_All_vine_tree.nwk")
REGIME = Path("data/3724_NT_All_vine_regime.csv")
Q_PARAMS = Path("outputs/diff/3724_NT_All_diff_Pri-Met_h1_q-mean-std_0.tsv")
OU_PARAMS = Path("outputs/diff/3724_NT_All_diff_Pri-Met_model-params.tsv")
READ_COUNTS = Path("data/3724_NT_All_readcounts.tsv")
LIBRARY = Path("data/3724_NT_All_library.tsv")
OUTPUT_DIR = Path("outputs/reconstruct/highlighted_genes")

DEFAULT_Q_THRESHOLD = 0.1
DEFAULT_EFFECT_THRESHOLD = 1.0
DEFAULT_LABELS_PER_SIDE = 6
DEFAULT_LABEL_EXCLUDE_REGEX = r"^(?:Rps|Rpl|Mrps|Mrpl|mt-)"
DEFAULT_TREE_LINEWIDTH = 0.6
DEFAULT_BAR_LINEWIDTH = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genes",
        nargs="+",
        help=(
            "Genes to reconstruct. If omitted, reconstructs the volcano-highlighted "
            "genes selected from --diff-results."
        ),
    )
    parser.add_argument("--diff-results", type=Path, default=DIFF_RESULTS)
    parser.add_argument("--tree", type=Path, default=TREE)
    parser.add_argument("--regime", type=Path, default=REGIME)
    parser.add_argument("--q-params", type=Path, default=Q_PARAMS)
    parser.add_argument("--ou-params", type=Path, default=OU_PARAMS)
    parser.add_argument("--read-counts", type=Path, default=READ_COUNTS)
    parser.add_argument("--library", type=Path, default=LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--q-threshold", type=float, default=DEFAULT_Q_THRESHOLD)
    parser.add_argument("--effect-threshold", type=float, default=DEFAULT_EFFECT_THRESHOLD)
    parser.add_argument(
        "--labels-per-side",
        type=int,
        default=DEFAULT_LABELS_PER_SIDE,
        help="Number of most extreme up/down genes to select when --genes is omitted.",
    )
    parser.add_argument(
        "--label-top-dot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also include the gene with the smallest p-value when --genes is omitted.",
    )
    parser.add_argument(
        "--label-exclude-regex",
        default=DEFAULT_LABEL_EXCLUDE_REGEX,
        help=(
            "Case-insensitive gene-name regex to exclude from automatic highlighted "
            "gene selection. Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print reconstruction commands without running them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs that already exist.",
    )
    parser.add_argument(
        "--no-outer",
        action="store_true",
        help="Do not draw read-count bars outside the reconstruction tree.",
    )
    parser.add_argument(
        "--tree-linewidth",
        type=float,
        default=DEFAULT_TREE_LINEWIDTH,
        help="Line width for reconstructed tree branches.",
    )
    parser.add_argument(
        "--bar-linewidth",
        type=float,
        default=DEFAULT_BAR_LINEWIDTH,
        help="Line width for outer read-count bars.",
    )
    return parser.parse_args()


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -50, 50)))


def highlighted_genes(args):
    df = pd.read_csv(args.diff_results, sep="\t")
    required = {"gene", "h1_theta_T1", "h1_theta_Met", "p", "q"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in {args.diff_results}: {missing}")

    df["delta_diff"] = softplus(df["h1_theta_Met"]) - softplus(df["h1_theta_T1"])
    df["neg_log_p"] = -np.log10(df["p"].clip(lower=1e-17))
    finite = np.isfinite(df[["delta_diff", "p", "q", "neg_log_p"]]).all(axis=1)
    df = df[finite].copy()

    sig = df["q"] < args.q_threshold
    large = df["delta_diff"].abs() >= args.effect_threshold
    if args.label_exclude_regex:
        label_candidates = ~df["gene"].astype(str).str.contains(
            re.compile(args.label_exclude_regex, re.IGNORECASE), na=False
        )
    else:
        label_candidates = pd.Series(True, index=df.index)

    label_sets = []
    if args.label_top_dot and not df.empty:
        label_sets.append(df.loc[label_candidates].nlargest(1, "neg_log_p"))
    if args.labels_per_side > 0:
        label_sets.extend(
            [
                df.loc[
                    label_candidates & sig & large & (df["delta_diff"] > 0)
                ].nlargest(
                    args.labels_per_side, "delta_diff"
                ),
                df.loc[
                    label_candidates & sig & large & (df["delta_diff"] < 0)
                ].nsmallest(
                    args.labels_per_side, "delta_diff"
                ),
            ]
        )

    if not label_sets:
        return []

    selected = pd.concat(label_sets, axis=0).drop_duplicates(
        subset=["gene"], keep="first"
    )
    return selected["gene"].astype(str).tolist()


def requested_genes(args):
    if args.genes:
        genes = []
        for value in args.genes:
            genes.extend(gene.strip() for gene in value.split(","))
        return [gene for gene in genes if gene]
    return highlighted_genes(args)


def assert_inputs_exist(args):
    paths = [
        args.tree,
        args.regime,
        args.q_params,
        args.ou_params,
        args.read_counts,
        args.library,
    ]
    if not args.genes:
        paths.append(args.diff_results)

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))


def assert_genes_available(genes, q_params):
    df = pd.read_csv(q_params, sep="\t", index_col=0, usecols=[0])
    available = set(df.index.astype(str))
    missing = [gene for gene in genes if gene not in available]
    if missing:
        raise ValueError(
            f"Gene(s) not found in {q_params}: {', '.join(missing)}"
        )


def safe_name(gene):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in gene)


def reconstruct_command(args, gene):
    stem = safe_name(gene)
    command = [
        sys.executable,
        "-m",
        "singlecellstochastics.reconstruct",
        "--tree",
        str(args.tree),
        "--q_params",
        str(args.q_params),
        "--read_counts",
        str(args.read_counts),
        "--library",
        str(args.library),
        "--gene",
        gene,
        "--model",
        "ou",
        "--regime",
        str(args.regime),
        "--ou",
        str(args.ou_params),
        "--out_tsv",
        str(args.output_dir / f"{stem}_history.tsv"),
        "--out_fig",
        str(args.output_dir / f"{stem}_history.png"),
        "--tree_linewidth",
        str(args.tree_linewidth),
        "--bar_linewidth",
        str(args.bar_linewidth),
    ]
    if args.no_outer:
        command.append("--no_outer")
    return command


def main():
    args = parse_args()
    assert_inputs_exist(args)

    genes = requested_genes(args)
    if not genes:
        raise ValueError("No genes selected for reconstruction.")

    assert_genes_available(genes, args.q_params)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reconstructing {len(genes)} gene(s): {', '.join(genes)}", flush=True)
    for gene in genes:
        out_fig = args.output_dir / f"{safe_name(gene)}_history.png"
        out_tsv = args.output_dir / f"{safe_name(gene)}_history.tsv"
        if not args.overwrite and out_fig.exists() and out_tsv.exists():
            print(f"Skipping {gene}: outputs already exist", flush=True)
            continue

        command = reconstruct_command(args, gene)
        print("+ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
