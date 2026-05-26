#!/usr/bin/env python
"""Plot gene expression correlations with DA and GLU pseudotime paths."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread


DEFAULT_TRAJECTORY_DIR = Path("outputs/e15_rep1_fp_trajectory_velocity")
DEFAULT_LAVOUS_FILE = Path(
    "outputs/e15_rep1_fp_lineage_vine/calibration/"
    "E15_rep1_fp_lineage_homogeneous_clone_vine_diff_pro_da_glu_"
    "sim_each2000_empirical-each.tsv"
)
DEFAULT_PREFIX = "e15_rep1_fp_cells"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help="Directory containing trajectory counts, genes, and obs outputs.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Trajectory output filename prefix.",
    )
    parser.add_argument(
        "--lavous-file",
        type=Path,
        default=DEFAULT_LAVOUS_FILE,
        help="LAVOUS calibration TSV containing gene-level significance.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix for PNG/PDF/TSV. Defaults inside trajectory-dir.",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Library-size target for log1p normalized expression.",
    )
    parser.add_argument(
        "--label-top",
        type=int,
        default=35,
        help="Number of significant genes to label, ranked by q then |DA-GLU|.",
    )
    return parser.parse_args()


def read_inputs(trajectory_dir, prefix, lavous_file):
    counts_path = trajectory_dir / f"{prefix}_counts.mtx"
    genes_path = trajectory_dir / f"{prefix}_genes.tsv"
    obs_path = trajectory_dir / f"{prefix}_trajectory_obs.tsv"

    for path in (counts_path, genes_path, obs_path, lavous_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    counts = mmread(str(counts_path)).tocsr().astype(np.float64)
    genes = pd.read_csv(genes_path, sep="\t")
    obs = pd.read_csv(obs_path, sep="\t")
    lavous = pd.read_csv(lavous_file, sep="\t")

    if "gene_name" not in genes.columns:
        raise ValueError(f"{genes_path} must contain a gene_name column")
    if counts.shape != (obs.shape[0], genes.shape[0]):
        raise ValueError(
            "Count matrix shape does not match obs/genes: "
            f"{counts.shape} vs {obs.shape[0]} cells and {genes.shape[0]} genes"
        )
    for column in ("fp_branch", "dpt_pseudotime"):
        if column not in obs.columns:
            raise ValueError(f"{obs_path} must contain a {column} column")
    for column in ("gene", "p", "q", "signif"):
        if column not in lavous.columns:
            raise ValueError(f"{lavous_file} must contain a {column} column")

    return counts, genes, obs, lavous


def log_normalize_counts(counts, target_sum):
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum,
        total_counts,
        out=np.zeros_like(total_counts, dtype=np.float64),
        where=total_counts > 0,
    )
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def sparse_pearson_by_gene(expression, pseudotime):
    pseudotime = np.asarray(pseudotime, dtype=np.float64)
    valid = np.isfinite(pseudotime)
    expression = expression[valid]
    pseudotime = pseudotime[valid]

    n_cells = expression.shape[0]
    if n_cells < 3:
        return np.full(expression.shape[1], np.nan, dtype=np.float64)

    time_centered = pseudotime - pseudotime.mean()
    time_ss = np.sum(time_centered**2)
    if time_ss <= 0:
        return np.full(expression.shape[1], np.nan, dtype=np.float64)

    gene_sum = np.asarray(expression.sum(axis=0)).ravel()
    gene_sq_sum = np.asarray(expression.power(2).sum(axis=0)).ravel()
    gene_ss = gene_sq_sum - (gene_sum**2 / n_cells)
    numerator = np.asarray(expression.T.dot(time_centered)).ravel()
    denominator = np.sqrt(gene_ss * time_ss)

    corr = np.divide(
        numerator,
        denominator,
        out=np.full(expression.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(corr, -1.0, 1.0)


def build_correlation_table(expression, genes, obs, lavous):
    branches = obs["fp_branch"].astype(str)
    pseudotime = obs["dpt_pseudotime"].astype(float).to_numpy()
    glu_mask = branches.isin(["shared", "glu"]).to_numpy()
    da_mask = branches.isin(["shared", "da"]).to_numpy()

    table = pd.DataFrame(
        {
            "gene_index": np.arange(genes.shape[0], dtype=int),
            "gene": genes["gene_name"].astype(str).to_numpy(),
            "corr_pseudotime_glu": sparse_pearson_by_gene(
                expression[glu_mask], pseudotime[glu_mask]
            ),
            "corr_pseudotime_da": sparse_pearson_by_gene(
                expression[da_mask], pseudotime[da_mask]
            ),
        }
    )
    table["corr_da_minus_glu"] = (
        table["corr_pseudotime_da"] - table["corr_pseudotime_glu"]
    )

    sig = lavous[["gene", "p", "q", "signif", "lrt"]].copy()
    sig = sig.rename(
        columns={
            "p": "lavous_p",
            "q": "lavous_q",
            "signif": "lavous_signif",
            "lrt": "lavous_lrt",
        }
    )
    sig["lavous_signif"] = sig["lavous_signif"].astype(bool)
    table = table.merge(sig, on="gene", how="left")
    table["lavous_signif"] = table["lavous_signif"].fillna(False)
    return table


def plot_correlations(table, out_prefix, label_top):
    plot_table = table.dropna(subset=["corr_pseudotime_glu", "corr_pseudotime_da"])
    significant = plot_table["lavous_signif"].astype(bool)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(
        plot_table.loc[~significant, "corr_pseudotime_glu"],
        plot_table.loc[~significant, "corr_pseudotime_da"],
        s=9,
        color="#B9B9B9",
        alpha=0.38,
        linewidths=0,
        label="Other genes",
    )
    ax.scatter(
        plot_table.loc[significant, "corr_pseudotime_glu"],
        plot_table.loc[significant, "corr_pseudotime_da"],
        s=34,
        color="#C43B32",
        alpha=0.95,
        linewidths=0.45,
        edgecolors="white",
        label="LAVOUS significant",
        zorder=3,
    )

    ax.axhline(0, color="#6E6E6E", linewidth=0.7, alpha=0.55, zorder=0)
    ax.axvline(0, color="#6E6E6E", linewidth=0.7, alpha=0.55, zorder=0)
    ax.plot([-1, 1], [-1, 1], color="#4A4A4A", linewidth=0.9, alpha=0.65, zorder=0)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Correlation(expression, pseudotime): progenitor to Glu")
    ax.set_ylabel("Correlation(expression, pseudotime): progenitor to DA")
    ax.set_title("E15 rep1 FP pseudotime expression correlations")
    ax.legend(frameon=False, loc="lower right", markerscale=1.2)

    label_table = (
        plot_table.loc[significant]
        .assign(abs_delta=lambda df: df["corr_da_minus_glu"].abs())
        .sort_values(["lavous_q", "abs_delta"], ascending=[True, False])
        .head(max(label_top, 0))
    )
    for i, row in enumerate(label_table.itertuples(index=False)):
        dx = 5 if row.corr_pseudotime_glu <= row.corr_pseudotime_da else -5
        dy = 4 if i % 2 == 0 else -6
        ax.annotate(
            row.gene,
            (row.corr_pseudotime_glu, row.corr_pseudotime_da),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=6.2,
            color="#6B1F1A",
            arrowprops={
                "arrowstyle": "-",
                "color": "#A46A65",
                "linewidth": 0.35,
                "alpha": 0.6,
            },
            zorder=4,
        )

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = (
            args.trajectory_dir
            / f"{args.prefix}_pseudotime_glu_vs_da_expression_correlations"
        )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    counts, genes, obs, lavous = read_inputs(
        args.trajectory_dir, args.prefix, args.lavous_file
    )
    expression = log_normalize_counts(counts, args.target_sum)
    table = build_correlation_table(expression, genes, obs, lavous)
    table.to_csv(out_prefix.with_suffix(".tsv"), sep="\t", index=False)
    plot_correlations(table, out_prefix, args.label_top)

    n_plotted = table[["corr_pseudotime_glu", "corr_pseudotime_da"]].notna().all(axis=1).sum()
    n_sig = table["lavous_signif"].sum()
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")
    print(f"Wrote {out_prefix.with_suffix('.tsv')}")
    print(f"plotted_genes: {n_plotted}")
    print(f"lavous_significant_genes: {n_sig}")


if __name__ == "__main__":
    main()
