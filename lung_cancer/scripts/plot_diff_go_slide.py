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


DEFAULT_INPUT = Path("outputs/diff/3724_NT_All_diff_Pri-Met_empirical-all_10000.tsv")
DEFAULT_OUTPUT_DIR = Path("outputs/diff/")
DEFAULT_Q_THRESHOLD = 0.05
DEFAULT_EFFECT_THRESHOLD = 0.5
DEFAULT_TERMS_PER_DIRECTION = 10
MIN_MARKER_AREA = 45
MAX_MARKER_AREA = 215
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


def softplus(values):
    values = np.asarray(values, dtype=float)
    return np.log1p(np.exp(np.clip(values, -50, 50)))


def clean_gene_list(values):
    genes = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        gene = str(value).strip()
        if not gene or gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


def read_diff_table(path):
    df = pd.read_csv(path, sep="\t")
    required = {"gene", "h1_theta_T1", "h1_theta_Met", "p", "q", "lrt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df["pri_expr"] = softplus(df["h1_theta_T1"])
    df["met_expr"] = softplus(df["h1_theta_Met"])
    df["delta_met_minus_pri"] = df["met_expr"] - df["pri_expr"]
    df["abs_delta"] = df["delta_met_minus_pri"].abs()
    df["direction"] = np.where(df["delta_met_minus_pri"] >= 0, "Met up", "Met down")

    numeric = ["pri_expr", "met_expr", "delta_met_minus_pri", "p", "q", "lrt"]
    finite = np.isfinite(df[numeric]).all(axis=1)
    return df.loc[finite].copy()


def run_gprofiler(query_genes, background_genes, args):
    from analyze_diff_fdr10 import run_gprofiler as run_gprofiler_impl

    return run_gprofiler_impl(query_genes, background_genes, args)


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
        "--use-existing-enrichment",
        action="store_true",
        help="Load existing GO enrichment TSVs from the output directory if present.",
    )
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


def marker_sizes(intersection_sizes, max_intersection):
    max_intersection = max(float(max_intersection), 1.0)
    values = pd.Series(intersection_sizes, dtype=float)
    return MIN_MARKER_AREA + (MAX_MARKER_AREA - MIN_MARKER_AREA) * (
        values / max_intersection
    )


def nice_legend_value(value):
    if value <= 0:
        return 1
    magnitude = 10 ** np.floor(np.log10(value))
    residual = value / magnitude
    if residual < 1.5:
        nice = 1
    elif residual < 3.5:
        nice = 2
    elif residual < 7.5:
        nice = 5
    else:
        nice = 10
    return int(max(1, nice * magnitude))


def size_legend_values(intersection_sizes):
    values = pd.Series(intersection_sizes, dtype=float).dropna()
    if values.empty:
        return []
    max_value = values.max()
    candidates = [max_value / 4, max_value / 2, max_value]
    return sorted({nice_legend_value(value) for value in candidates})


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
    direction_colors = {"Met up": "#b2182b", "Met down": "#2166ac"}
    plot_df["direction_order"] = plot_df["direction"].map(direction_order)
    plot_df = plot_df.sort_values(
        ["direction_order", "neg_log10_adj_p"], ascending=[True, False]
    ).reset_index(drop=True)
    plot_df["wrapped_label"] = plot_df["plot_label"].map(
        lambda label: wrap_label(label, width=34)
    )

    max_intersection = max(plot_df["intersection_size"].astype(float).max(), 1)

    filter_names = []
    if not args.keep_ribosomal_genes:
        filter_names.append("ribosomal")
    if not args.keep_mitochondrial_genes:
        filter_names.append("mt-encoded")
    title_prefix = "Compact GO enrichment"
    if args.terms_per_direction != DEFAULT_TERMS_PER_DIRECTION:
        title_prefix = f"Top {args.terms_per_direction} up/down GO enrichment"
    filter_label = "/".join(filter_names)
    title = (
        f"{title_prefix} after {filter_label} gene filter\n"
        f"q <= {args.q_threshold:g}, |diff| >= {args.effect_threshold:g}"
    )
    if not filter_names:
        title = (
            f"{title_prefix}\n"
            f"q <= {args.q_threshold:g}, |diff| >= {args.effect_threshold:g}"
        )

    max_x = max(plot_df["neg_log10_adj_p"].astype(float).max(), 1.0) * 1.08
    fig_height = max(4.2, 0.45 * args.terms_per_direction + 1.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, fig_height),
        dpi=300,
        sharex=True,
        gridspec_kw={"wspace": 0.95},
    )

    for ax, direction in zip(axes, ["Met up", "Met down"]):
        panel_df = (
            plot_df.loc[plot_df["direction"] == direction]
            .sort_values("neg_log10_adj_p", ascending=False)
            .head(args.terms_per_direction)
            .reset_index(drop=True)
        )
        y = np.arange(len(panel_df))
        sizes = marker_sizes(panel_df["intersection_size"], max_intersection)
        color = direction_colors[direction]

        for yi, row, size in zip(y, panel_df.itertuples(), sizes):
            xi = row.neg_log10_adj_p
            ax.plot([0, xi], [yi, yi], color="#d0d0d0", linewidth=1.0, zorder=1)
            ax.scatter(
                xi,
                yi,
                s=size,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

        ax.set_title(direction, fontsize=10, color=color, pad=6)
        ax.set_yticks(y)
        ax.set_yticklabels(panel_df["wrapped_label"], fontsize=9)
        ax.set_xlim(0, max_x)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("")
        ax.set_xlabel("$-\\log_{10}$(adjusted enrichment p)", fontsize=9)

    size_handles = []
    for value in size_legend_values(plot_df["intersection_size"]):
        area = marker_sizes([value], max_intersection).iloc[0]
        size_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#808080",
                markeredgecolor="white",
                markersize=np.sqrt(area),
                label=f"{value:g}",
            )
        )
    if size_handles:
        fig.legend(
            handles=size_handles,
            title="Intersection genes",
            frameon=False,
            loc="center right",
            bbox_to_anchor=(0.985, 0.5),
            fontsize=9,
            title_fontsize=9,
            labelspacing=1.1,
            handletextpad=1.2,
        )
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 0.91, 0.93])
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

        enrich_path = args.output_dir / f"go_{gene_prefix}_{source_label}.tsv"
        if args.use_existing_enrichment and enrich_path.exists():
            enrich = pd.read_csv(enrich_path, sep="\t")
            enrich_action = "loaded"
        else:
            enrich = run_gprofiler(query_genes, background_genes, gp_args)
            enrich.to_csv(enrich_path, sep="\t", index=False)
            enrich_action = "wrote"

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
            f"{enrich_action} {enrich_path}"
        )

    plot_df = pd.concat(plot_parts, ignore_index=True)
    suffix = "compact"
    if args.terms_per_direction != DEFAULT_TERMS_PER_DIRECTION:
        suffix += f"_top{args.terms_per_direction}"
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
