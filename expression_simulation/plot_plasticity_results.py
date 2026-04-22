"""
Plot plasticity simulation results A-J with pairwise shuffled negative comparisons.

Figure 1: LR statistics + fraction significant (real vs shuffled)
Figure 2: Estimated Pagel's lambda comparisons (real vs shuffled)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from pathlib import Path

# ── Nature style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path("plasticity")
FIG_DIR = Path("plasticity")

LABELS = list("ABCDEFGHIJ")

SCENARIO_DESC = {
    "A": "Baseline",
    "B": "High θ",
    "C": "Different θ",
    "D": "High σ",
    "E": "No α (BM)",
    "F": "Weak α",
    "G": "Strong α",
    "H": "Strong r⁻¹",
    "I": "Weak r⁻¹",
    "J": "No r (Poisson)",
}

CMAP = plt.get_cmap("tab10")
PANEL_COLORS = {label: CMAP(i % 10) for i, label in enumerate(LABELS)}


def lighten(color, amount=0.5):
    """Blend *color* toward white by *amount* (0=unchanged, 1=white)."""
    import matplotlib.colors as mc
    r, g, b, a = mc.to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, a)

# ── Load data ─────────────────────────────────────────────────────────────
data_real = {}
data_shuf = {}

for label in LABELS:
    f_real = DATA_DIR / f"plasticity_{label}_nb.tsv"
    f_shuf = DATA_DIR / f"plasticity_{label}_shuffled_nb.tsv"
    if f_real.exists():
        data_real[label] = pd.read_csv(f_real, sep="\t")
    if f_shuf.exists():
        data_shuf[label] = pd.read_csv(f_shuf, sep="\t")

# Keep only labels where both files exist
labels = [l for l in LABELS if l in data_real and l in data_shuf]
display_names = [SCENARIO_DESC[l] for l in labels]
n = len(labels)


# ── Helpers ───────────────────────────────────────────────────────────────
def frac_signif(df):
    signif = df["signif"]
    if signif.dtype == bool:
        return signif.sum() / len(signif)
    return signif.astype(str).str.lower().eq("true").sum() / len(signif)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: LR statistics + fraction significant (real vs shuffled)
# ═══════════════════════════════════════════════════════════════════════════
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))

positions = np.arange(n)
width = 0.35

# Panel (a): LR boxplots — real vs shuffled
real_lr = [data_real[l]["lr"].values for l in labels]
shuf_lr = [data_shuf[l]["lr"].values for l in labels]

bp_real = ax1.boxplot(
    real_lr, positions=positions - width / 2, widths=width * 0.8,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=0.8),
    boxprops=dict(linewidth=0.5),
    whiskerprops=dict(linewidth=0.5),
    capprops=dict(linewidth=0.5),
)
for i, patch in enumerate(bp_real["boxes"]):
    patch.set_facecolor(PANEL_COLORS[labels[i]])
    patch.set_alpha(0.7)

bp_shuf = ax1.boxplot(
    shuf_lr, positions=positions + width / 2, widths=width * 0.8,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=0.8),
    boxprops=dict(linewidth=0.5),
    whiskerprops=dict(linewidth=0.5),
    capprops=dict(linewidth=0.5),
)
for i, patch in enumerate(bp_shuf["boxes"]):
    patch.set_facecolor(lighten(PANEL_COLORS[labels[i]], 0.5))
    patch.set_alpha(0.7)

ax1.set_xticks(positions)
ax1.set_xticklabels(display_names, rotation=45, ha="right")
ax1.set_ylabel("LR statistic")
ax1.axhline(3.841, color="red", ls="--", lw=0.5, label=r"$\chi^2$(1) p=0.05")
handles, _ = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, frameon=False, loc="upper right")
ax1.set_title("a", fontweight="bold", loc="left")

# Panel (b): Fraction significant — real vs shuffled
tp_vals = [frac_signif(data_real[l]) for l in labels]
fp_vals = [frac_signif(data_shuf[l]) for l in labels]

x = np.arange(n)
bar_w = 0.35
for i, l in enumerate(labels):
    ax2.bar(x[i] - bar_w / 2, tp_vals[i], bar_w, color=PANEL_COLORS[l], alpha=0.7)
    ax2.bar(x[i] + bar_w / 2, fp_vals[i], bar_w, color=lighten(PANEL_COLORS[l], 0.5), alpha=0.7)

for i, v in enumerate(tp_vals):
    ax2.text(i - bar_w / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)
for i, v in enumerate(fp_vals):
    ax2.text(i + bar_w / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)

ax2.set_xticks(x)
ax2.set_xticklabels(display_names, rotation=45, ha="right")
ax2.set_ylabel("Fraction significant (q < 0.05)")
ax2.set_ylim(0, 1.15)
ax2.set_title("b", fontweight="bold", loc="left")

fig1.tight_layout()
for fmt in ["png", "pdf"]:
    fig1.savefig(FIG_DIR / f"plasticity_LR_comparison.{fmt}", bbox_inches="tight")
plt.close(fig1)
print("Saved plasticity_LR_comparison.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Pagel's lambda comparisons (real vs shuffled)
# ═══════════════════════════════════════════════════════════════════════════
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(7.5, 3.0))

# Panel (c): Lambda boxplots — real vs shuffled
real_lam = [data_real[l]["h1_lambda"].values for l in labels]
shuf_lam = [data_shuf[l]["h1_lambda"].values for l in labels]

bp_rl = ax3.boxplot(
    real_lam, positions=positions - width / 2, widths=width * 0.8,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=0.8),
    boxprops=dict(linewidth=0.5),
    whiskerprops=dict(linewidth=0.5),
    capprops=dict(linewidth=0.5),
)
for i, patch in enumerate(bp_rl["boxes"]):
    patch.set_facecolor(PANEL_COLORS[labels[i]])
    patch.set_alpha(0.7)

bp_sl = ax3.boxplot(
    shuf_lam, positions=positions + width / 2, widths=width * 0.8,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=0.8),
    boxprops=dict(linewidth=0.5),
    whiskerprops=dict(linewidth=0.5),
    capprops=dict(linewidth=0.5),
)
for i, patch in enumerate(bp_sl["boxes"]):
    patch.set_facecolor(lighten(PANEL_COLORS[labels[i]], 0.5))
    patch.set_alpha(0.7)

ax3.set_xticks(positions)
ax3.set_xticklabels(display_names, rotation=45, ha="right")
ax3.set_ylabel(r"Estimated Pagel's $\lambda$")
ax3.set_title("a", fontweight="bold", loc="left")

# Panel (d): Median lambda comparison (bar chart)
median_real_lam = [np.median(data_real[l]["h1_lambda"].values) for l in labels]
median_shuf_lam = [np.median(data_shuf[l]["h1_lambda"].values) for l in labels]

for i, l in enumerate(labels):
    ax4.bar(x[i] - bar_w / 2, median_real_lam[i], bar_w, color=PANEL_COLORS[l], alpha=0.7)
    ax4.bar(x[i] + bar_w / 2, median_shuf_lam[i], bar_w, color=lighten(PANEL_COLORS[l], 0.5), alpha=0.7)

for i, v in enumerate(median_real_lam):
    ax4.text(i - bar_w / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)
for i, v in enumerate(median_shuf_lam):
    ax4.text(i + bar_w / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)

ax4.set_xticks(x)
ax4.set_xticklabels(display_names, rotation=45, ha="right")
ax4.set_ylabel(r"Median Pagel's $\lambda$")
ax4.set_title("b", fontweight="bold", loc="left")

fig2.tight_layout()
for fmt in ["png", "pdf"]:
    fig2.savefig(FIG_DIR / f"plasticity_lambda_comparison.{fmt}", bbox_inches="tight")
plt.close(fig2)
print("Saved plasticity_lambda_comparison.png/pdf")
