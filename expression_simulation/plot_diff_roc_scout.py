import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def load_scores(oup_file, egx_file, wlx_file, scout_file):
    """Load and compute signed -log10(p) scores for each method."""
    df_oup = pd.read_csv(oup_file, sep="\t")
    df_egx = pd.read_csv(egx_file, sep=",")
    df_wlx = pd.read_csv(wlx_file, sep="\t")
    df_sct = pd.read_csv(scout_file, sep=",")

    # OUP: signed -log10(p)
    score_oup = (np.sign(df_oup["h1_theta1"].values - df_oup["h1_theta0"].values)
                 * (-np.log10(df_oup["p"].values)))

    # EGX: paired rows (theta0, theta1); p-value from first of each pair
    score_egx = (np.sign(df_egx["ou2_theta"].iloc[1::2].values - df_egx["ou2_theta"].iloc[::2].values)
                 * (-np.log10(np.maximum(1e-100, df_egx["ou2_vs_ou1_pvalue"].iloc[::2].values))))

    # Wilcoxon: signed -log10(p)
    score_wlx = (-np.sign(df_wlx["log2FC"].values) # flipped wilcoxon
                 * (-np.log10(df_wlx["p_value"].values)))

    # SCOUT: OUM AICc weight (higher = more evidence for regime-specific model)
    score_sct = df_sct["OUM_aicc_weight"].values

    return score_oup, score_egx, score_wlx, score_sct


def plot_roc_curve(ax, oup_neg, oup_pos, egx_neg, egx_pos, wlx_neg, wlx_pos,
                   sct_neg, sct_pos, title=None):
    s_oup_neg, s_egx_neg, s_wlx_neg, s_sct_neg = load_scores(oup_neg, egx_neg, wlx_neg, sct_neg)
    s_oup_pos, s_egx_pos, s_wlx_pos, s_sct_pos = load_scores(oup_pos, egx_pos, wlx_pos, sct_pos)

    n_neg = len(s_oup_neg)
    n_pos = len(s_oup_pos)
    truth = np.array([False] * n_neg + [True] * n_pos)

    for scores_neg, scores_pos, label, color in [
        (s_sct_neg, s_sct_pos, "SCOUT", "tab:gray"),
        #(s_wlx_neg, s_wlx_pos, "Wilcoxon", "tab:gray"),
        (s_egx_neg, s_egx_pos, "EvoGeneX", "tab:orange"),
        (s_oup_neg, s_oup_pos, "LaVOUS", "tab:green"),
    ]:
        p = np.concatenate((scores_neg, scores_pos))
        fpr, tpr, _ = roc_curve(truth, p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, color=color, label=f"{label} {roc_auc:.3f}")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    if title:
        ax.set_title(title)
    # Reorder legend: LaVOUS, EvoGeneX, SCOUT
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="lower right")


# Define dataset pairs: (neg_label, pos_label)
datasets = [
    # Row 0
    [("C", "A"), ("I_diff", "I"), ("J_diff", "J")],
    # Row 1
    [("C_diff", "A"), ("t5_r50", "I"), ("t5_r0", "J")],
    # Row 2
    [("t7_r5", "A"), ("t7_r50", "I"), ("t7_r0", "J")],
]

nrows = len(datasets)
ncols = len(datasets[0])
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

for i, row in enumerate(datasets):
    for j, (pos, neg) in enumerate(row):
        oup_neg = f"diff/{neg}_chi-squared.tsv"
        oup_pos = f"diff/{pos}_chi-squared.tsv"
        egx_neg = f"evogenex/egx_{neg}.csv"
        egx_pos = f"evogenex/egx_{pos}.csv"
        wlx_neg = f"Wilcoxon/wilcoxon_{neg}.tsv"
        wlx_pos = f"Wilcoxon/wilcoxon_{pos}.tsv"
        sct_neg = f"scout/scout_{neg}.csv"
        sct_pos = f"scout/scout_{pos}.csv"
        plot_roc_curve(axes[i, j], oup_neg, oup_pos, egx_neg, egx_pos,
                       wlx_neg, wlx_pos, sct_neg, sct_pos)

fig.supxlabel("False Positive Rate")
fig.supylabel("True Positive Rate")
plt.tight_layout(pad=0.5)
plt.savefig("diff_ROC_scout.png", dpi=300, bbox_inches="tight")
plt.show()
