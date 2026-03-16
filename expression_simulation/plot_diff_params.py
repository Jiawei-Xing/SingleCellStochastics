"""
Boxplots of inferred OU parameters across simulation conditions,
similar to Figure 2E in the VOUS paper.

Layout: single row with 4 panels — α, σ²/(2α), θ₀, θ₁
Each panel groups by (θ₁, r) condition from the ROC simulations.
Ground truth: α=1, σ=3, θ₀=1; θ₁ and r vary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Positive (H1) datasets: all share α=1, σ=3, θ₀=1
# (label, file, true_theta1, true_r_label)
conditions = [
    # Group by θ₁
    ("$\\theta_1=3$\n$r=5$",   "diff/C_chi-squared.tsv",       3,  5),
    ("$\\theta_1=3$\n$r=50$",  "diff/I_diff_chi-squared.tsv",  3,  50),
    ("$\\theta_1=3$\nPoisson", "diff/J_diff_chi-squared.tsv",  3,  "Poi"),
    ("$\\theta_1=5$\n$r=5$",   "diff/C_diff_chi-squared.tsv",  5,  5),
    ("$\\theta_1=5$\n$r=50$",  "diff/t5_r50_chi-squared.tsv",  5,  50),
    ("$\\theta_1=5$\nPoisson", "diff/t5_r0_chi-squared.tsv",   5,  "Poi"),
    ("$\\theta_1=7$\n$r=5$",   "diff/t7_r5_chi-squared.tsv",   7,  5),
    ("$\\theta_1=7$\n$r=50$",  "diff/t7_r50_chi-squared.tsv",  7,  50),
    ("$\\theta_1=7$\nPoisson", "diff/t7_r0_chi-squared.tsv",   7,  "Poi"),
]

# Load data
dfs = []
for tick, fpath, true_t1, true_r in conditions:
    dfs.append(pd.read_csv(fpath, sep="\t"))

# True values
TRUE_ALPHA = 1
TRUE_SIGMA = 3
TRUE_SIGMA2_2A = TRUE_SIGMA**2 / (2 * TRUE_ALPHA)  # 4.5
TRUE_THETA0 = 1

# ── Figure ──
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
tick_labels = [c[0] for c in conditions]

# Colors by θ₁ group
colors = []
for _, _, t1, _ in conditions:
    if t1 == 3:
        colors.append("tab:blue")
    elif t1 == 5:
        colors.append("tab:orange")
    else:
        colors.append("tab:green")

def styled_boxplot(ax, data_list, true_val=None, ylabel=None):
    bp = ax.boxplot(data_list, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=1.5, alpha=0.3),
                    medianprops=dict(color='black', lw=1),
                    whiskerprops=dict(lw=0.7),
                    capprops=dict(lw=0.7))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.5)
    if true_val is not None:
        ax.axhline(true_val, color='black', ls='--', lw=0.8, alpha=0.5, zorder=0)
    ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Panel 1: α
styled_boxplot(axes[0],
               [df["h1_alpha"].values for df in dfs],
               true_val=TRUE_ALPHA,
               ylabel="$\\alpha$")
axes[0].set_title("$\\alpha$", fontsize=10)

# Panel 2: σ²/(2α)
styled_boxplot(axes[1],
               [df["h1_sigma"].values**2 / (2 * df["h1_alpha"].values) for df in dfs],
               true_val=TRUE_SIGMA2_2A,
               ylabel="$\\sigma^2/(2\\alpha)$")
axes[1].set_title("$\\sigma^2/(2\\alpha)$", fontsize=10)

# Panel 3: θ₀
styled_boxplot(axes[2],
               [df["h1_theta0"].values for df in dfs],
               true_val=TRUE_THETA0,
               ylabel="$\\theta_0$")
axes[2].set_title("$\\theta_0$", fontsize=10)

# Panel 4: θ₁ — different true values per condition
ax = axes[3]
data_t1 = [df["h1_theta1"].values for df in dfs]
bp = ax.boxplot(data_t1, patch_artist=True, showfliers=True,
                flierprops=dict(marker='o', markersize=1.5, alpha=0.3),
                medianprops=dict(color='black', lw=1),
                whiskerprops=dict(lw=0.7),
                capprops=dict(lw=0.7))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.5)
# Draw true θ₁ lines per group
for true_t1, color, start, end in [(3, 'tab:blue', 0.5, 3.5),
                                     (5, 'tab:orange', 3.5, 6.5),
                                     (7, 'tab:green', 6.5, 9.5)]:
    ax.plot([start, end], [true_t1, true_t1], '--', color=color, lw=0.8, alpha=0.6)
ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=0)
ax.set_ylabel("$\\theta_1$", fontsize=9)
ax.set_title("$\\theta_1$", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=0.8)
plt.savefig("diff_params.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved diff_params.png")
