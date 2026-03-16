import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from pathlib import Path
import os

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
PLASTICITY_DIR = Path("selection")
FIG_DIR = Path("selection")
FIG_DIR.mkdir(exist_ok=True)

# Map scenarios: (OU_label, BM_label, Display Name)
# I merged your two dictionaries to match up the corresponding scenarios
SCENARIO_PAIRS = [
    ("A", "E", "Baseline"),
    ("B", "B_BM", "High θ"),
    ("C", "E", "Different θ"),
    ("D", "D_BM", "High σ"),
    ("F", "E", "Weak α"),
    ("I", "I_BM", "Weak $r^{-1}$"),
    ("J", "J_BM", "No r (Poisson)")
]

# ── Load data ─────────────────────────────────────────────────────────────
data_ou = {}
data_bm = {}

for ou_label, bm_label, _ in SCENARIO_PAIRS:
    f_ou = PLASTICITY_DIR / f"selection_{ou_label}_nb.tsv"
    f_bm = PLASTICITY_DIR / f"selection_{bm_label}_nb.tsv"
    if f_ou.exists():
        data_ou[ou_label] = pd.read_csv(f_ou, sep="\t")
    if f_bm.exists():
        data_bm[bm_label] = pd.read_csv(f_bm, sep="\t")

# Filter pairs where BOTH files exist to keep alignment, 
# extracting matching names and datasets
ou_data_list = []
bm_data_list = []
display_names = []

for ou_label, bm_label, name in SCENARIO_PAIRS:
    if ou_label in data_ou and bm_label in data_bm:
        ou_data_list.append(data_ou[ou_label]["lr"].values)
        bm_data_list.append(data_bm[bm_label]["lr"].values)
        display_names.append(name)

# ── Panel (b): compute TP/FP fractions ────────────────────────────────────
def count_signif_true(result_file):
    if not os.path.exists(result_file):
        print(f"Missing: {result_file}")
        return np.nan
    df = pd.read_csv(result_file, sep="\t")
    if "signif" not in df.columns:
        raise ValueError(f"'signif' column not found in {result_file}")
    signif = df["signif"]
    if signif.dtype == bool:
        return int(signif.sum())
    return int(signif.astype(str).str.lower().eq("true").sum())

tp_fp_rows = []
for ou_label, null_label, name in SCENARIO_PAIRS:
    f_ou = PLASTICITY_DIR / f"selection_{ou_label}_nb.tsv"
    f_null = PLASTICITY_DIR / f"selection_{null_label}_nb.tsv"
    tp = count_signif_true(str(f_ou)) / 500
    fp = count_signif_true(str(f_null)) / 500
    tp_fp_rows.append({"name": name, "TP": tp, "FP": fp})
tp_fp_df = pd.DataFrame(tp_fp_rows)

# ── Figure: 1 panel, side-by-side ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

# Compute positions for grouping
positions = np.arange(len(display_names))
width = 0.35  # Offset width 

# Boxplot (a) - OU Model (Shifted left)
bp_ou = ax1.boxplot(
    ou_data_list, positions=positions - width/2, widths=width*0.8,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=0.8),
    boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5)
)
for patch in bp_ou["boxes"]:
    patch.set_facecolor("#1f77b4") # Blue
    patch.set_alpha(0.7)

# Boxplot (b) - BM Model (Shifted right)
bp_bm = ax1.boxplot(
    bm_data_list, positions=positions + width/2, widths=width*0.8,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=0.8),
    boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5)
)
for patch in bp_bm["boxes"]:
    patch.set_facecolor("#ff7f0e") # Orange
    patch.set_alpha(0.7)

# Axes formatting
ax1.set_xticks(positions)
ax1.set_xticklabels(display_names, rotation=45, ha="right")
ax1.set_ylabel("LR statistic")
ax1.axhline(3.841, color="red", ls="--", lw=0.5, label="χ²(1) p=0.05")

# Custom Legend combining both the models and the significance line
ou_patch = mpatches.Patch(color='#1f77b4', alpha=0.7, label='OU')
bm_patch = mpatches.Patch(color='#ff7f0e', alpha=0.7, label='BM')
handles, labels = ax1.get_legend_handles_labels()  # Grab the red dashed line
ax1.legend(handles=[ou_patch, bm_patch] + handles, frameon=False, loc="upper right")
ax1.set_title("a", fontweight="bold", loc="left")

# — Panel (b): TP/FP bar chart —
x = np.arange(len(tp_fp_df))
bar_w = 0.35
ax2.bar(x - bar_w/2, tp_fp_df["TP"], bar_w, color="#1f77b4", alpha=0.7, label="OU")
ax2.bar(x + bar_w/2, tp_fp_df["FP"], bar_w, color="#ff7f0e", alpha=0.7, label="BM")
for i, v in enumerate(tp_fp_df["TP"]):
    if not np.isnan(v):
        ax2.text(i - bar_w/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=5)
for i, v in enumerate(tp_fp_df["FP"]):
    if not np.isnan(v):
        ax2.text(i + bar_w/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=5)
ax2.set_xticks(x)
ax2.set_xticklabels(tp_fp_df["name"], rotation=45, ha="right")
ax2.set_ylabel("Fraction significant")
ax2.set_ylim(0, 1)
ax2.legend(frameon=False, loc="upper right")
ax2.set_title("b", fontweight="bold", loc="left")

plt.tight_layout()
fig.savefig(FIG_DIR / "selection_sim_results_combined.png", bbox_inches="tight")
plt.close(fig)