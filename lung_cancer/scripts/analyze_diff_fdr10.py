#!/usr/bin/env python
"""GO enrichment and summary plots for LaVOUS Pri-Met differential results."""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


DEFAULT_INPUT = Path("outputs/diff/3724_NT_All_diff_Pri-Met_empirical-all_10000.tsv")
DEFAULT_OUTPUT_DIR = Path("outputs/diff/fdr10_analysis")
DEFAULT_Q_THRESHOLD = 0.10
DEFAULT_EFFECT_THRESHOLD = 1.0
DEFAULT_TOP_N = 25
P_FLOOR = 1e-300
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=DEFAULT_Q_THRESHOLD,
        help="FDR cutoff used to select differential genes.",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=DEFAULT_EFFECT_THRESHOLD,
        help=(
            "Minimum absolute delta model-implied expression used to select "
            "genes, matching scripts/plot_volcano.py."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top genes or GO terms to show per plot.",
    )
    parser.add_argument(
        "--go-source",
        choices=("gprofiler", "gmt", "none"),
        default="gprofiler",
        help=(
            "GO enrichment backend. Use 'gprofiler' for the live API, 'gmt' for "
            "offline over-representation analysis, or 'none' for plots only."
        ),
    )
    parser.add_argument(
        "--organism",
        default="mmusculus",
        help="g:Profiler organism code, for example mmusculus or hsapiens.",
    )
    parser.add_argument(
        "--gmt",
        type=Path,
        help="GMT gene set file for offline enrichment when --go-source gmt.",
    )
    parser.add_argument(
        "--go-sources",
        nargs="+",
        default=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG", "WP"],
        help="g:Profiler annotation sources to query.",
    )
    parser.add_argument(
        "--go-threshold",
        type=float,
        default=0.10,
        help="Adjusted enrichment p-value cutoff for GO plots.",
    )
    parser.add_argument(
        "--no-custom-background",
        action="store_true",
        help="Do not use all tested genes as the g:Profiler enrichment background.",
    )
    parser.add_argument(
        "--gprofiler-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each g:Profiler request.",
    )
    return parser.parse_args()


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
    df["neg_log10_p"] = -np.log10(df["p"].clip(lower=P_FLOOR))
    df["direction"] = np.where(df["delta_met_minus_pri"] >= 0, "Met up", "Met down")

    numeric = ["pri_expr", "met_expr", "delta_met_minus_pri", "p", "q", "lrt"]
    finite = np.isfinite(df[numeric]).all(axis=1)
    return df.loc[finite].copy()


def write_gene_outputs(df, selected, output_dir, q_threshold, effect_threshold):
    prefix = output_dir / f"diff_fdr{q_threshold:g}_absdiff{effect_threshold:g}"
    cols = [
        "ID",
        "gene",
        "direction",
        "delta_met_minus_pri",
        "pri_expr",
        "met_expr",
        "lrt",
        "p",
        "q",
    ]
    cols = [col for col in cols if col in selected.columns]

    selected.sort_values(["q", "p", "abs_delta"], ascending=[True, True, False]).to_csv(
        prefix.with_name(prefix.name + "_genes.tsv"),
        sep="\t",
        index=False,
        columns=cols,
    )
    df[["gene"]].drop_duplicates().to_csv(
        prefix.with_name(prefix.name + "_background_genes.txt"),
        index=False,
        header=False,
    )

    for label, subset in {
        "all": selected,
        "met_up": selected.loc[selected["delta_met_minus_pri"] > 0],
        "met_down": selected.loc[selected["delta_met_minus_pri"] < 0],
    }.items():
        genes = clean_gene_list(subset["gene"])
        (prefix.with_name(prefix.name + f"_{label}_genes.txt")).write_text(
            "\n".join(genes) + ("\n" if genes else "")
        )

    return prefix


def save_delta_histogram(df, selected, output_dir, q_threshold, effect_threshold):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=200)
    bins = np.linspace(
        np.nanpercentile(df["delta_met_minus_pri"], 1),
        np.nanpercentile(df["delta_met_minus_pri"], 99),
        60,
    )
    ax.hist(
        df["delta_met_minus_pri"],
        bins=bins,
        color="#d9d9d9",
        edgecolor="white",
        linewidth=0.4,
        label="All tested",
    )
    ax.hist(
        selected["delta_met_minus_pri"],
        bins=bins,
        color="#4c78a8",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        label=f"q <= {q_threshold:g}, |diff| >= {effect_threshold:g}",
    )
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.axvline(effect_threshold, color="#333333", linestyle=":", linewidth=1.0)
    ax.axvline(-effect_threshold, color="#333333", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Delta model-implied expression, Met - Pri")
    ax.set_ylabel("Genes")
    ax.set_title("Differential effect distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "delta_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_theta_scatter(df, selected, output_dir, q_threshold, effect_threshold, top_n):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=200)
    ax.scatter(
        df["pri_expr"],
        df["met_expr"],
        s=8,
        alpha=0.35,
        color="#9e9e9e",
        edgecolors="none",
        label="All tested",
    )
    up = selected["delta_met_minus_pri"] > 0
    down = selected["delta_met_minus_pri"] < 0
    ax.scatter(
        selected.loc[down, "pri_expr"],
        selected.loc[down, "met_expr"],
        s=10,
        alpha=0.75,
        color="#2166ac",
        edgecolors="none",
        label=f"Met down, q <= {q_threshold:g}, |diff| >= {effect_threshold:g}",
    )
    ax.scatter(
        selected.loc[up, "pri_expr"],
        selected.loc[up, "met_expr"],
        s=10,
        alpha=0.75,
        color="#b2182b",
        edgecolors="none",
        label=f"Met up, q <= {q_threshold:g}, |diff| >= {effect_threshold:g}",
    )

    hi = max(df["pri_expr"].quantile(0.995), df["met_expr"].quantile(0.995))
    ax.plot([0, hi], [0, hi], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlim(left=0, right=hi)
    ax.set_ylim(bottom=0, top=hi)
    ax.set_xlabel("Primary model-implied expression")
    ax.set_ylabel("Metastasis model-implied expression")
    ax.set_title("Pri-Met expression shift")
    ax.legend(frameon=False, fontsize=8)

    label_df = selected.nlargest(max(0, min(top_n, len(selected))), "abs_delta")
    for _, row in label_df.iterrows():
        ax.annotate(
            row["gene"],
            (min(row["pri_expr"], hi), min(row["met_expr"], hi)),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
            fontstyle="italic",
            color="#333333",
        )

    fig.tight_layout()
    path = output_dir / "pri_met_expression_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_top_gene_barplot(selected, output_dir, top_n):
    if selected.empty:
        return None

    up = selected.loc[selected["delta_met_minus_pri"] > 0].nlargest(
        top_n, "delta_met_minus_pri"
    )
    down = selected.loc[selected["delta_met_minus_pri"] < 0].nsmallest(
        top_n, "delta_met_minus_pri"
    )
    plot_df = pd.concat([down, up], axis=0)
    if plot_df.empty:
        return None

    colors = np.where(plot_df["delta_met_minus_pri"] > 0, "#b2182b", "#2166ac")
    height = max(4.5, 0.22 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, height), dpi=200)
    ax.barh(plot_df["gene"], plot_df["delta_met_minus_pri"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Delta model-implied expression, Met - Pri")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} Met-up and Met-down genes")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    path = output_dir / "top_met_up_down_genes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_volcano(df, selected, output_dir, q_threshold, effect_threshold, top_n):
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)
    passes_q = df["q"] <= q_threshold
    passes_effect = df["abs_delta"] >= effect_threshold
    small_effect = passes_q & ~passes_effect
    ns = ~passes_q
    up = passes_q & passes_effect & (df["delta_met_minus_pri"] > 0)
    down = passes_q & passes_effect & (df["delta_met_minus_pri"] < 0)

    ax.scatter(
        df.loc[ns, "delta_met_minus_pri"],
        df.loc[ns, "neg_log10_p"],
        s=8,
        alpha=0.3,
        color="#bdbdbd",
        edgecolors="none",
        label="Not selected",
    )
    ax.scatter(
        df.loc[small_effect, "delta_met_minus_pri"],
        df.loc[small_effect, "neg_log10_p"],
        s=8,
        alpha=0.35,
        color="#f4a582",
        edgecolors="none",
        label="q pass, small effect",
    )
    ax.scatter(
        df.loc[down, "delta_met_minus_pri"],
        df.loc[down, "neg_log10_p"],
        s=10,
        alpha=0.75,
        color="#2166ac",
        edgecolors="none",
        label="Met down",
    )
    ax.scatter(
        df.loc[up, "delta_met_minus_pri"],
        df.loc[up, "neg_log10_p"],
        s=10,
        alpha=0.75,
        color="#b2182b",
        edgecolors="none",
        label="Met up",
    )

    p_cutoff = df.loc[passes_q, "p"].max() if passes_q.any() else np.nan
    if pd.notna(p_cutoff):
        ax.axhline(
            -math.log10(max(p_cutoff, P_FLOOR)),
            color="#333333",
            linestyle="--",
            linewidth=1,
            label=f"q <= {q_threshold:g}",
        )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axvline(effect_threshold, color="#333333", linestyle=":", linewidth=0.8)
    ax.axvline(-effect_threshold, color="#333333", linestyle=":", linewidth=0.8)

    label_df = selected.nlargest(max(0, min(top_n, len(selected))), "abs_delta")
    for _, row in label_df.iterrows():
        ax.annotate(
            row["gene"],
            (row["delta_met_minus_pri"], row["neg_log10_p"]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
            fontstyle="italic",
            color="#333333",
        )

    ax.set_xlabel("Delta model-implied expression, Met - Pri")
    ax.set_ylabel("$-\\log_{10}(p)$")
    ax.set_title("Pri-Met differential test")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "volcano_fdr10.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_gmt(path):
    gene_sets = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[0]
            description = fields[1]
            genes = clean_gene_list(fields[2:])
            gene_sets[name] = {"description": description, "genes": set(genes)}
    return gene_sets


def fdr_bh(p_values):
    if not p_values:
        return []
    return multipletests(p_values, method="fdr_bh")[1]


def flatten_intersections(values):
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        return [str(values)]

    flattened = []
    for value in values:
        flattened.extend(flatten_intersections(value))
    return flattened


def gprofiler_intersection_genes(row, data):
    intersections = row.get("intersections")
    if not intersections:
        return []

    query_name = row.get("query")
    query_meta = (
        data.get("meta", {})
        .get("genes_metadata", {})
        .get("query", {})
        .get(query_name, {})
    )
    mapped_genes = list(query_meta.get("mapping", {}).keys())
    if mapped_genes and len(mapped_genes) == len(intersections):
        return [
            mapped_genes[index]
            for index, evidence in enumerate(intersections)
            if evidence
        ]

    return flatten_intersections(intersections)


def run_gmt_ora(query_genes, background_genes, gmt_path):
    query = set(query_genes) & set(background_genes)
    background = set(background_genes)
    gene_sets = parse_gmt(gmt_path)
    rows = []

    for term, record in gene_sets.items():
        term_genes = record["genes"] & background
        overlap = term_genes & query
        if not overlap:
            continue

        population = len(background)
        successes = len(term_genes)
        draws = len(query)
        hits = len(overlap)
        p_value = hypergeom.sf(hits - 1, population, successes, draws)
        rows.append(
            {
                "source": "GMT",
                "term_id": term,
                "term_name": term,
                "description": record["description"],
                "p_value": p_value,
                "term_size": successes,
                "query_size": draws,
                "intersection_size": hits,
                "precision": hits / draws if draws else np.nan,
                "recall": hits / successes if successes else np.nan,
                "intersection_genes": ",".join(sorted(overlap)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["adjusted_p_value"] = fdr_bh(result["p_value"].tolist())
    return result.sort_values(["adjusted_p_value", "p_value", "term_name"])


def run_gprofiler(query_genes, background_genes, args):
    payload = {
        "organism": args.organism,
        "query": query_genes,
        "sources": args.go_sources,
        "user_threshold": 1.0,
        "all_results": True,
        "no_evidences": False,
    }
    if not args.no_custom_background:
        payload["domain_scope"] = "custom"
        payload["background"] = background_genes

    request = urllib.request.Request(
        GPROFILER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.gprofiler_timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    rows = []
    for row in data.get("result", []):
        rows.append(
            {
                "source": row.get("source"),
                "term_id": row.get("native"),
                "term_name": row.get("name"),
                "p_value": row.get("p_value"),
                "adjusted_p_value": row.get("p_value"),
                "term_size": row.get("term_size"),
                "query_size": row.get("query_size"),
                "intersection_size": row.get("intersection_size"),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "intersection_genes": ",".join(gprofiler_intersection_genes(row, data)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["adjusted_p_value", "p_value", "term_name"])


def save_enrichment_plot(enrich, output_path, title, threshold, top_n):
    if enrich.empty:
        return None

    plot_df = enrich.loc[enrich["adjusted_p_value"] <= threshold].copy()
    if plot_df.empty:
        plot_df = enrich.head(top_n).copy()
    else:
        plot_df = plot_df.head(top_n)

    if plot_df.empty:
        return None

    plot_df["plot_name"] = plot_df["term_name"].fillna(plot_df["term_id"]).astype(str)
    plot_df["plot_name"] = plot_df["plot_name"].str.slice(0, 70)
    plot_df["neg_log10_adj_p"] = -np.log10(
        plot_df["adjusted_p_value"].astype(float).clip(lower=P_FLOOR)
    )
    plot_df = plot_df.iloc[::-1]

    height = max(4.0, 0.32 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, height), dpi=200)
    sizes = 30 + 160 * (
        plot_df["intersection_size"].astype(float)
        / max(plot_df["intersection_size"].astype(float).max(), 1)
    )
    scatter = ax.scatter(
        plot_df["neg_log10_adj_p"],
        plot_df["plot_name"],
        s=sizes,
        c=plot_df["precision"].astype(float),
        cmap="viridis",
        edgecolors="#333333",
        linewidth=0.3,
    )
    ax.set_xlabel("$-\\log_{10}$(adjusted enrichment p)")
    ax.set_ylabel("")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Gene ratio")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_enrichment(label, genes, background_genes, args):
    if not genes:
        print(f"Skipping {label} enrichment: no genes")
        return None

    if args.go_source == "none":
        return None

    if args.go_source == "gmt":
        if args.gmt is None:
            raise ValueError("--gmt is required when --go-source gmt")
        result = run_gmt_ora(genes, background_genes, args.gmt)
    else:
        try:
            result = run_gprofiler(genes, background_genes, args)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"Skipping {label} g:Profiler enrichment: {exc}", file=sys.stderr)
            return None

    out_tsv = args.output_dir / f"go_{label}.tsv"
    result.to_csv(out_tsv, sep="\t", index=False)
    plot_path = save_enrichment_plot(
        result,
        args.output_dir / f"go_{label}_top_terms.png",
        f"{label.replace('_', ' ').title()} enrichment",
        args.go_threshold,
        args.top_n,
    )
    return out_tsv, plot_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_diff_table(args.input)
    sig = df.loc[df["q"] <= args.q_threshold].copy()
    selected = sig.loc[sig["abs_delta"] >= args.effect_threshold].copy()
    up = selected.loc[selected["delta_met_minus_pri"] > 0]
    down = selected.loc[selected["delta_met_minus_pri"] < 0]
    background_genes = clean_gene_list(df["gene"])

    prefix = write_gene_outputs(
        df, selected, args.output_dir, args.q_threshold, args.effect_threshold
    )
    plot_paths = [
        save_delta_histogram(
            df, selected, args.output_dir, args.q_threshold, args.effect_threshold
        ),
        save_theta_scatter(
            df,
            selected,
            args.output_dir,
            args.q_threshold,
            args.effect_threshold,
            args.top_n,
        ),
        save_top_gene_barplot(selected, args.output_dir, args.top_n),
        save_volcano(
            df,
            selected,
            args.output_dir,
            args.q_threshold,
            args.effect_threshold,
            args.top_n,
        ),
    ]
    plot_paths = [path for path in plot_paths if path is not None]

    go_tag = f"fdr{args.q_threshold:g}_absdiff{args.effect_threshold:g}"
    gene_sets = {
        f"all_{go_tag}": clean_gene_list(selected["gene"]),
        f"met_up_{go_tag}": clean_gene_list(up["gene"]),
        f"met_down_{go_tag}": clean_gene_list(down["gene"]),
    }
    enrichment_outputs = []
    for label, genes in gene_sets.items():
        result = run_enrichment(label, genes, background_genes, args)
        if result is not None:
            enrichment_outputs.append(result)

    print(f"Read {len(df):,} tested genes from {args.input}")
    print(
        f"Selected {len(selected):,} genes at q <= {args.q_threshold:g} and "
        f"|diff| >= {args.effect_threshold:g}: "
        f"{len(up):,} Met-up, {len(down):,} Met-down"
    )
    print(f"Wrote gene tables with prefix {prefix}")
    for path in plot_paths:
        print(f"Wrote plot {path}")
    for tsv_path, plot_path in enrichment_outputs:
        print(f"Wrote enrichment table {tsv_path}")
        if plot_path is not None:
            print(f"Wrote enrichment plot {plot_path}")


if __name__ == "__main__":
    main()
