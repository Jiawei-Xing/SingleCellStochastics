#!/usr/bin/env python
"""Slide-ready GO figure for Pri-Met differential genes."""

import argparse
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_diff_fdr10 import clean_gene_list, read_diff_table, run_gprofiler


DEFAULT_INPUT = Path("outputs/diff/3724_NT_All_diff_Pri-Met_empirical-all_10000.tsv")
DEFAULT_OUTPUT_DIR = Path("outputs/diff/")
DEFAULT_Q_THRESHOLD = 0.05
DEFAULT_EFFECT_THRESHOLD = 0.5
DEFAULT_TERMS_PER_DIRECTION = 10
RIBOSOMAL_GENE_RE = re.compile(r"^(Rpl|Rps|Mrpl|Mrps)", re.IGNORECASE)
MITOCHONDRIAL_GENE_RE = re.compile(r"^mt-", re.IGNORECASE)
RIBOSOME_TERM_RE = re.compile(
    "|".join(
        [
            "ribosom",
            "translation",
            "translational",
            "40S",
            "60S",
            "peptide biosynthetic",
            "nonsense mediated decay",
            "SRP-dependent",
            "rRNA",
        ]
    ),
    re.IGNORECASE,
)
GENERIC_TERM_NAMES = {"WIKIPATHWAYS"}
P_FLOOR = 1e-300


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--q-threshold", type=float, default=DEFAULT_Q_THRESHOLD)
    parser.add_argument(
        "--effect-threshold", type=float, default=DEFAULT_EFFECT_THRESHOLD
    )
    parser.add_argument(
        "--terms-per-direction",
        type=int,
        default=DEFAULT_TERMS_PER_DIRECTION,
        help="Number of compact GO terms to display for each direction.",
    )
    parser.add_argument("--organism", default="mmusculus")
    parser.add_argument(
        "--go-sources",
        nargs="+",
        default=["GO:BP"],
    )
    parser.add_argument("--go-threshold", type=float, default=0.10)
    parser.add_argument("--gprofiler-timeout", type=int, default=60)
    parser.add_argument(
        "--keep-ribosomal-genes",
        action="store_true",
        help="Do not remove Rpl/Rps/Mrpl/Mrps genes from GO query sets.",
    )
    parser.add_argument(
        "--keep-mitochondrial-genes",
        action="store_true",
        help="Do not remove mt-* mitochondrial-encoded genes from GO query sets.",
    )
    parser.add_argument(
        "--hide-ribosome-terms",
        dest="keep_ribosome_terms",
        action="store_false",
        help="Hide ribosome/translation terms in the compact plot.",
    )
    parser.add_argument(
        "--keep-ribosome-terms",
        dest="keep_ribosome_terms",
        action="store_true",
        help="Keep ribosome/translation terms in the compact plot after gene filtering.",
    )
    parser.set_defaults(keep_ribosome_terms=True)
    return parser.parse_args()


def is_ribosomal_gene(gene):
    return bool(RIBOSOMAL_GENE_RE.search(str(gene)))


def is_mitochondrial_gene(gene):
    return bool(MITOCHONDRIAL_GENE_RE.search(str(gene)))


def source_tag(sources):
    return "_".join(source.lower().replace(":", "") for source in sources)


def selected_genes(diff_df, q_threshold, effect_threshold, direction):
    selected = diff_df.loc[
        (diff_df["q"] <= q_threshold)
        & (diff_df["abs_delta"] >= effect_threshold)
        & (diff_df["direction"] == direction)
    ]
    return clean_gene_list(selected["gene"])


def filter_genes(genes, keep_ribosomal_genes, keep_mitochondrial_genes):
    kept = []
    removed = {"ribosomal": [], "mitochondrial": []}
    for gene in genes:
        if not keep_ribosomal_genes and is_ribosomal_gene(gene):
            removed["ribosomal"].append(gene)
        elif not keep_mitochondrial_genes and is_mitochondrial_gene(gene):
            removed["mitochondrial"].append(gene)
        else:
            kept.append(gene)
    return kept, removed


def wrap_label(label, width=42):
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def compact_terms(enrich, direction, terms_per_direction, go_threshold, keep_terms):
    if enrich.empty:
        return enrich

    terms = enrich.copy()
    if not keep_terms:
        mask = terms["term_name"].fillna("").str.contains(RIBOSOME_TERM_RE)
        terms = terms.loc[~mask].copy()
    terms = terms.loc[~terms["term_name"].fillna("").isin(GENERIC_TERM_NAMES)].copy()

    if terms.empty:
        return terms

    sig = terms.loc[terms["adjusted_p_value"] <= go_threshold].copy()
    if sig.empty:
        sig = terms.copy()

    sig = sig.head(terms_per_direction).copy()
    sig["direction"] = direction
    sig["plot_label"] = sig.apply(
        lambda row: f"{row['source']} | {row['term_name']}", axis=1
    )
    sig["neg_log10_adj_p"] = -np.log10(
        sig["adjusted_p_value"].astype(float).clip(lower=P_FLOOR)
    )
    return sig


def save_compact_plot(plot_df, output_path, removed_counts, args):
    if plot_df.empty:
        raise ValueError("No enrichment terms available for compact slide plot")

    plot_df = plot_df.copy()
    direction_order = {"Met up": 0, "Met down": 1}
    plot_df["direction_order"] = plot_df["direction"].map(direction_order)
    plot_df = plot_df.sort_values(
        ["direction_order", "neg_log10_adj_p"], ascending=[True, False]
    ).reset_index(drop=True)
    plot_df["wrapped_label"] = plot_df["plot_label"].map(wrap_label)

    colors = plot_df["direction"].map({"Met up": "#b2182b", "Met down": "#2166ac"})
    sizes = 45 + 170 * (
        plot_df["intersection_size"].astype(float)
        / max(plot_df["intersection_size"].astype(float).max(), 1)
    )

    fig_height = max(3.2, 0.4 * len(plot_df) + 1.0)
    fig, ax = plt.subplots(figsize=(7.3, fig_height), dpi=300)
    y = np.arange(len(plot_df))

    for yi, xi, color in zip(y, plot_df["neg_log10_adj_p"], colors):
        ax.plot([0, xi], [yi, yi], color="#d0d0d0", linewidth=1.0, zorder=1)
        ax.scatter(xi, yi, s=sizes.iloc[yi], color=color, edgecolor="white", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["wrapped_label"], fontsize=7.5)
    ax.set_xlabel("$-\\log_{10}$(adjusted enrichment p)", fontsize=9)
    ax.set_ylabel("")
    filter_names = []
    if not args.keep_ribosomal_genes:
        filter_names.append("ribosomal")
    if not args.keep_mitochondrial_genes:
        filter_names.append("mt-encoded")
    filter_label = "/".join(filter_names)
    title = (
        f"Compact GO enrichment after {filter_label} gene filter\n"
        f"q <= {args.q_threshold:g}, |diff| >= {args.effect_threshold:g}"
    )
    if not filter_names:
        title = (
            f"Compact GO enrichment\n"
            f"q <= {args.q_threshold:g}, |diff| >= {args.effect_threshold:g}"
        )
    ax.set_title(title, fontsize=10, pad=8)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#b2182b",
            markeredgecolor="white",
            markersize=7,
            label="Met up",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#2166ac",
            markeredgecolor="white",
            markersize=7,
            label="Met down",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diff_df = read_diff_table(args.input)
    background_genes = clean_gene_list(diff_df["gene"])
    background_genes, background_removed = filter_genes(
        background_genes, args.keep_ribosomal_genes, args.keep_mitochondrial_genes
    )

    gp_args = SimpleNamespace(
        organism=args.organism,
        go_sources=args.go_sources,
        no_custom_background=False,
        gprofiler_timeout=args.gprofiler_timeout,
    )
    source_label = source_tag(args.go_sources)

    plot_parts = []
    removed_counts = {}
    for direction, label in [("Met up", "met_up"), ("Met down", "met_down")]:
        genes = selected_genes(
            diff_df, args.q_threshold, args.effect_threshold, direction
        )
        query_genes, removed = filter_genes(
            genes, args.keep_ribosomal_genes, args.keep_mitochondrial_genes
        )
        removed_counts[direction] = removed

        gene_prefix = (
            f"diff_fdr{args.q_threshold:g}_absdiff{args.effect_threshold:g}_{label}"
        )
        if not args.keep_ribosomal_genes:
            gene_prefix += "_nonribosomal"
        if not args.keep_mitochondrial_genes:
            gene_prefix += "_nonmitochondrial"

        (args.output_dir / f"{gene_prefix}_genes.txt").write_text(
            "\n".join(query_genes) + ("\n" if query_genes else "")
        )
        if removed["ribosomal"]:
            (args.output_dir / f"{gene_prefix}_removed_ribosomal_genes.txt").write_text(
                "\n".join(removed["ribosomal"]) + "\n"
            )
        if removed["mitochondrial"]:
            (
                args.output_dir / f"{gene_prefix}_removed_mitochondrial_genes.txt"
            ).write_text("\n".join(removed["mitochondrial"]) + "\n")

        enrich = run_gprofiler(query_genes, background_genes, gp_args)
        enrich_path = args.output_dir / f"go_{gene_prefix}_{source_label}.tsv"
        enrich.to_csv(enrich_path, sep="\t", index=False)

        plot_parts.append(
            compact_terms(
                enrich,
                direction,
                args.terms_per_direction,
                args.go_threshold,
                args.keep_ribosome_terms,
            )
        )

        print(
            f"{direction}: {len(genes)} selected genes, "
            f"{len(query_genes)} queried after gene filter "
            f"({len(removed['ribosomal'])} ribosomal, "
            f"{len(removed['mitochondrial'])} mt-encoded removed); "
            f"wrote {enrich_path}"
        )

    plot_df = pd.concat(plot_parts, ignore_index=True)
    suffix = "compact"
    if not args.keep_ribosomal_genes:
        suffix += "_nonribosomal"
    if not args.keep_mitochondrial_genes:
        suffix += "_nonmitochondrial"
    if not args.keep_ribosome_terms:
        suffix += "_no_translation_terms"
    output_path = args.output_dir / f"go_slide_{source_label}_{suffix}.png"
    save_compact_plot(plot_df, output_path, removed_counts, args)
    print(f"Wrote slide figure {output_path}")


if __name__ == "__main__":
    main()
