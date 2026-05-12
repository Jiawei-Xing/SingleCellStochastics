#!/usr/bin/env python
"""Scatter plot of plasticity test results: Pagel's lambda vs -log10(p)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PLASTICITY_FILE = "outputs/heritability/3724_NT_All_heritability.tsv"
OUTPUT = "outputs/heritability/3724_NT_All_heritability.png"
FDR_THRESHOLD = 0.05
P_FLOOR = 1e-8

df = pd.read_csv(PLASTICITY_FILE, sep="\t")

# Compute -log10(p), clipping p=0 to a small value.
p_vals = df["p"].replace(0, P_FLOOR)
neg_log_p = np.clip(-np.log10(p_vals), None, 8)

lam = df["h1_lambda"]
n_sig = (df["q"] < FDR_THRESHOLD).sum()
n_total = len(df)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

ax.scatter(
    lam,
    neg_log_p,
    s=8,
    alpha=0.5,
    color="#4393c3",
    edgecolors="none",
    zorder=1,
)

# FDR 5% guide on a -log10(p) axis.
sig_p_cutoff = df.loc[df["q"] < FDR_THRESHOLD, "p"].max()
if pd.notna(sig_p_cutoff):
    fdr_y = -np.log10(max(sig_p_cutoff, P_FLOOR))
    ax.axhline(
        fdr_y,
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        zorder=2,
    )
    ax.text(
        0.98,
        fdr_y,
        "FDR 5%",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="red",
    )

ax.set_xlabel("Pagel's $\\lambda$", fontsize=13)
ax.set_ylabel(r"$-\log_{10}(p\text{-value})$", fontsize=13)

ax.tick_params(labelsize=11)
ax.set_xlim(-0.02, max(lam) * 1.05)
ax.set_ylim(bottom=-0.3)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved {OUTPUT}")
print(f"  {n_sig}/{n_total} significant genes")
