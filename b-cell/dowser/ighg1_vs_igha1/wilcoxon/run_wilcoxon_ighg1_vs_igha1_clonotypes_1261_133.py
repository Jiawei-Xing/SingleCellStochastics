#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR.parent
CLONES = ["Clonotype_1261", "Clonotype_133"]
OUT = SCRIPT_DIR / "ighg1_vs_igha1_wilcoxon_1261_133.tsv"


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qvals = np.empty(n, dtype=float)
    qvals[order] = np.minimum(ranked, 1.0)
    return qvals


def read_clone(clone: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    counts = pd.read_csv(DATA / "readcounts" / f"{clone}.readcounts.tsv", sep="\t")
    counts = counts.set_index("cell")

    regimes = pd.read_csv(DATA / "regimes" / f"{clone}.ighg1_vs_igha1.regime.csv")
    regimes = regimes[regimes["node_name"].isin(counts.index)]
    regimes = regimes.set_index("node_name").loc[counts.index, "regime"]
    regimes = regimes[regimes.isin(["IGHG1", "IGHA1"])]
    counts = counts.loc[regimes.index]

    libs = pd.read_csv(
        DATA / "library" / f"{clone}.library.tsv",
        sep="\t",
        header=None,
        names=["cell", "size_factor"],
    ).set_index("cell")["size_factor"]
    libs = libs.loc[counts.index].astype(float)

    normalized = counts.astype(float).div(libs, axis=0)
    expr = np.log1p(normalized)
    clones = pd.Series(clone, index=counts.index, name="clone")
    return counts, expr, regimes, clones


def main() -> None:
    annot = pd.read_csv(
        DATA / "metadata" / "gene_annotation.tsv",
        sep="\t",
        header=None,
        names=["gene_id", "symbol"],
    )
    symbol = annot.drop_duplicates("gene_id").set_index("gene_id")["symbol"]

    raw_parts = []
    expr_parts = []
    label_parts = []
    clone_parts = []
    for clone in CLONES:
        raw, expr, labels, clone_series = read_clone(clone)
        raw_parts.append(raw)
        expr_parts.append(expr)
        label_parts.append(labels)
        clone_parts.append(clone_series)

    raw_all = pd.concat(raw_parts, axis=0)
    expr_all = pd.concat(expr_parts, axis=0)
    labels = pd.concat(label_parts, axis=0)
    clones = pd.concat(clone_parts, axis=0)

    group_ighg1 = labels == "IGHG1"
    group_igha1 = labels == "IGHA1"
    if not group_ighg1.any() or not group_igha1.any():
        raise RuntimeError("Both IGHG1 and IGHA1 cells are required.")

    rows = []
    for gene_id in expr_all.columns:
        x = expr_all.loc[group_igha1, gene_id].to_numpy(dtype=float)
        y = expr_all.loc[group_ighg1, gene_id].to_numpy(dtype=float)
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            stat = np.nan
            pvalue = 1.0
        else:
            try:
                res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            except TypeError:
                res = mannwhitneyu(x, y, alternative="two-sided")
            stat = float(res.statistic)
            pvalue = float(res.pvalue)

        raw_x = raw_all.loc[group_igha1, gene_id].to_numpy(dtype=float)
        raw_y = raw_all.loc[group_ighg1, gene_id].to_numpy(dtype=float)
        mean_norm_x = np.expm1(x).mean()
        mean_norm_y = np.expm1(y).mean()
        rows.append(
            {
                "gene_id": gene_id,
                "symbol": symbol.get(gene_id, gene_id),
                "u_stat_igha1_vs_ighg1": stat,
                "pvalue": pvalue,
                "mean_log1p_igha1": x.mean(),
                "mean_log1p_ighg1": y.mean(),
                "diff_mean_log1p_igha1_minus_ighg1": x.mean() - y.mean(),
                "mean_norm_igha1": mean_norm_x,
                "mean_norm_ighg1": mean_norm_y,
                "log2fc_norm_igha1_vs_ighg1": np.log2((mean_norm_x + 1e-9) / (mean_norm_y + 1e-9)),
                "pct_detected_igha1": float((raw_x > 0).mean()),
                "pct_detected_ighg1": float((raw_y > 0).mean()),
            }
        )

    res = pd.DataFrame(rows)
    res["qvalue_bh"] = bh_fdr(res["pvalue"].to_numpy())
    res = res.sort_values(["qvalue_bh", "pvalue", "symbol"], kind="mergesort").reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    res.to_csv(OUT, sep="\t", index=False)

    counts_by_clone = pd.crosstab(clones, labels)
    print(f"wrote\t{OUT}")
    print("group_counts_by_clone")
    print(counts_by_clone.to_csv(sep="\t").strip())
    print("group_counts_total")
    print(labels.value_counts().to_csv(sep="\t", header=False).strip())
    print("top_25")
    cols = [
        "rank",
        "symbol",
        "gene_id",
        "pvalue",
        "qvalue_bh",
        "diff_mean_log1p_igha1_minus_ighg1",
        "log2fc_norm_igha1_vs_ighg1",
        "pct_detected_igha1",
        "pct_detected_ighg1",
    ]
    print(res.loc[:24, cols].to_csv(sep="\t", index=False).strip())
    print("markers")
    marker = res[res["symbol"].isin(["XBP1", "JCHAIN"])]
    print(marker[cols].to_csv(sep="\t", index=False).strip())


if __name__ == "__main__":
    main()
