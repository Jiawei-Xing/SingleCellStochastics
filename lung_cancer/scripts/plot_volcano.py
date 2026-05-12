#!/usr/bin/env python
"""Volcano plot for LaVOUS differential expression test (Pri vs Met)."""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

INPUT = Path("outputs/diff/3724_NT_All_diff_Pri-Met_empirical-all_10000.tsv")
OUTPUT = Path("outputs/diff/3724_NT_All_volcano_pri_met_empirical-all_10000.png")
DEFAULT_Q_THRESHOLD = 0.05
DEFAULT_EFFECT_THRESHOLD = 0.5
DEFAULT_OUTLIER_LABELS_PER_SIDE = 6
DEFAULT_LABEL_EXCLUDE_REGEX = r"^(?:Rps|Rpl|Mrps|Mrpl|mt-)"
DEFAULT_FIG_WIDTH = 6
DEFAULT_FIG_HEIGHT = 5
DEFAULT_LABEL_FONTSIZE = 10
P_FLOOR = 1e-17
X_LIM = 60
Y_LIM = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=DEFAULT_Q_THRESHOLD,
        help="q-value cutoff used to classify significant genes.",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=DEFAULT_EFFECT_THRESHOLD,
        help="Absolute delta softplus(theta) cutoff used to color large effects.",
    )
    parser.add_argument(
        "--outlier-labels-per-side",
        type=int,
        default=DEFAULT_OUTLIER_LABELS_PER_SIDE,
        help="Number of most extreme up/down significant genes to label.",
    )
    parser.add_argument(
        "--label-top-dot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also label the gene with the largest -log10(p-value).",
    )
    parser.add_argument(
        "--label-exclude-regex",
        default=DEFAULT_LABEL_EXCLUDE_REGEX,
        help=(
            "Case-insensitive gene-name regex to exclude from automatic labels. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument("--fig-width", type=float, default=DEFAULT_FIG_WIDTH)
    parser.add_argument("--fig-height", type=float, default=DEFAULT_FIG_HEIGHT)
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=DEFAULT_LABEL_FONTSIZE,
        help="Font size for automatically placed gene labels.",
    )
    parser.add_argument("--x-lim", type=float, default=X_LIM)
    parser.add_argument("--y-lim", type=float, default=Y_LIM)
    return parser.parse_args()


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -50, 50)))


def format_threshold(value):
    return f"{value:g}"


def label_candidate_mask(df, label_exclude_regex):
    if not label_exclude_regex:
        return pd.Series(True, index=df.index)
    excluded = df["gene"].astype(str).str.contains(
        re.compile(label_exclude_regex, re.IGNORECASE), na=False
    )
    return ~excluded


def get_label_genes(df, sig, large, labels_per_side, label_top_dot, label_candidates):
    label_sets = []
    candidate_df = df.loc[label_candidates]
    if label_top_dot and not df.empty:
        label_sets.append(candidate_df.nlargest(1, "neg_log_p"))
    if labels_per_side > 0:
        label_sets.extend(
            [
                df.loc[
                    label_candidates & sig & large & (df["delta_diff"] > 0)
                ].nlargest(
                    labels_per_side, "delta_diff"
                ),
                df.loc[
                    label_candidates & sig & large & (df["delta_diff"] < 0)
                ].nsmallest(
                    labels_per_side, "delta_diff"
                ),
            ]
        )

    if not label_sets:
        return df.iloc[0:0]
    return pd.concat(label_sets, axis=0).drop_duplicates(subset=["gene"], keep="first")


def label_offset_candidates(x):
    direction = 1 if x >= 0 else -1
    return [
        (direction * 5, 0),
        (direction * 5, 6),
        (direction * 5, -6),
        (direction * 10, 0),
        (direction * 10, 9),
        (direction * 10, -9),
        (0, 8),
        (0, -8),
        (-direction * 5, 6),
        (-direction * 5, -6),
        (-direction * 10, 0),
    ]


def candidate_score(bbox, accepted_bboxes, point_pixels, point_index, own_index, axes_bbox):
    padded = bbox.padded(1.5)
    label_overlaps = sum(padded.overlaps(existing) for existing in accepted_bboxes)

    in_label = (
        (point_pixels[:, 0] >= padded.x0)
        & (point_pixels[:, 0] <= padded.x1)
        & (point_pixels[:, 1] >= padded.y0)
        & (point_pixels[:, 1] <= padded.y1)
        & (point_index != own_index)
    )
    point_overlaps = int(in_label.sum())

    outside_pixels = (
        max(0, axes_bbox.x0 - padded.x0)
        + max(0, padded.x1 - axes_bbox.x1)
        + max(0, axes_bbox.y0 - padded.y0)
        + max(0, padded.y1 - axes_bbox.y1)
    )
    return (label_overlaps, int(outside_pixels > 0), point_overlaps, outside_pixels)


def place_gene_labels(fig, ax, df, label_genes, label_fontsize):
    if label_genes.empty:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    point_pixels = ax.transData.transform(
        df[["plot_delta_diff", "plot_neg_log_p"]].to_numpy()
    )
    point_index = df.index.to_numpy()
    axes_bbox = ax.get_window_extent(renderer).padded(-2)
    text_effects = [pe.withStroke(linewidth=2.5, foreground="white")]
    accepted_bboxes = []

    for row_index, row in label_genes.iterrows():
        color = "#b2182b" if row["delta_diff"] > 0 else "#2166ac"
        best = None

        for x_offset, y_offset in label_offset_candidates(row["plot_delta_diff"]):
            ha = "left" if x_offset > 0 else "right" if x_offset < 0 else "center"
            va = "bottom" if y_offset > 0 else "top" if y_offset < 0 else "center"
            annotation = ax.annotate(
                row["gene"],
                (row["plot_delta_diff"], row["plot_neg_log_p"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=label_fontsize,
                fontstyle="italic",
                color=color,
                zorder=5,
                path_effects=text_effects,
            )
            bbox = annotation.get_window_extent(renderer).padded(1.5)
            score = candidate_score(
                bbox, accepted_bboxes, point_pixels, point_index, row_index, axes_bbox
            )

            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, annotation, bbox)
            else:
                annotation.remove()

            if score == (0, 0, 0, 0):
                break

        accepted_bboxes.append(best[2])


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    required = {"gene", "h1_theta_T1", "h1_theta_Met", "p", "q"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    # Delta in model-implied expression scale: f(theta_Met) - f(theta_Pri).
    df["sp_T1"] = softplus(df["h1_theta_T1"])
    df["sp_Met"] = softplus(df["h1_theta_Met"])
    df["delta_diff"] = df["sp_Met"] - df["sp_T1"]

    # Drop genes with non-finite statistics, usually from failed convergence.
    finite = np.isfinite(df[["delta_diff", "p", "q"]]).all(axis=1)
    df = df[finite].copy()

    df["neg_log_p"] = -np.log10(df["p"].clip(lower=P_FLOOR))

    # Clip outliers to plot bounds so they appear at the edge instead of off-canvas.
    df["plot_delta_diff"] = df["delta_diff"].clip(-args.x_lim, args.x_lim)
    df["plot_neg_log_p"] = df["neg_log_p"].clip(upper=args.y_lim)

    # Classify genes by q-value significance and delta effect size.
    sig = df["q"] < args.q_threshold
    large = df["delta_diff"].abs() >= args.effect_threshold
    up = sig & large & (df["delta_diff"] > 0)
    down = sig & large & (df["delta_diff"] < 0)
    faded = sig & ~large
    ns = ~sig

    # Key genes for lung cancer metastasis (with per-gene label offsets).
    # Kept commented out for now.
    # highlight_genes = {
    #     # UP in Met - metastasis drivers / EMT / invasion
    #     "S100a6": (10, -10),  # S100 calcium-binding, EMT/migration
    #     "Lgals1": (8, 6),  # galectin-1, immune evasion & angiogenesis
    #     "Spp1": (-40, -12),  # osteopontin, EMT / immune evasion
    #     "Tgfb1": (8, -12),  # TGF-beta, EMT driver
    #     "Mmp14": (8, 6),  # matrix metalloproteinase, invasion
    #     "Snai1": (8, 6),  # EMT transcription factor
    #     # DOWN in Met - adaptation to metastatic site
    #     "Ccl5": (8, 6),  # immune chemokine
    #     "Ifit1": (8, -10),  # interferon response
    #     "H19": (-35, -12),  # imprinted lncRNA
    #     "Igf2": (8, 6),  # imprinted growth factor
    #     "Isg15": (8, 6),  # interferon-stimulated gene
    #     "Hspb1": (-45, 4),  # stress response
    # }

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=200)

    ax.scatter(
        df.loc[ns, "plot_delta_diff"],
        df.loc[ns, "plot_neg_log_p"],
        s=6,
        alpha=0.4,
        color="#cccccc",
        edgecolors="none",
        zorder=1,
        label="Not significant",
    )

    ax.scatter(
        df.loc[faded, "plot_delta_diff"],
        df.loc[faded, "plot_neg_log_p"],
        s=6,
        alpha=0.3,
        color="#cccccc",
        edgecolors="none",
        zorder=1,
    )

    ax.scatter(
        df.loc[down, "plot_delta_diff"],
        df.loc[down, "plot_neg_log_p"],
        s=10,
        alpha=0.6,
        color="#2166ac",
        edgecolors="none",
        zorder=2,
        label=f"Down in S1 ({down.sum()})",
    )

    ax.scatter(
        df.loc[up, "plot_delta_diff"],
        df.loc[up, "plot_neg_log_p"],
        s=10,
        alpha=0.6,
        color="#b2182b",
        edgecolors="none",
        zorder=2,
        label=f"Up in S1 ({up.sum()})",
    )

    label_candidates = label_candidate_mask(df, args.label_exclude_regex)
    label_genes = get_label_genes(
        df,
        sig,
        large,
        args.outlier_labels_per_side,
        args.label_top_dot,
        label_candidates,
    )

    # Highlight key genes with labels. Kept commented out for now.
    # import matplotlib.patheffects as pe
    #
    # text_effects = [pe.withStroke(linewidth=2.5, foreground="white")]
    # for gene, offset in highlight_genes.items():
    #     row = df[df["gene"] == gene]
    #     if row.empty:
    #         continue
    #     x = row["plot_delta_diff"].values[0]
    #     y = row["plot_neg_log_p"].values[0]
    #     color = "#b2182b" if row["delta_diff"].values[0] > 0 else "#2166ac"
    #     if row["q"].values[0] >= args.q_threshold:
    #         color = "#888888"
    #
    #     ax.scatter(
    #         x, y, s=30, color=color, edgecolors="black", linewidths=0.5, zorder=4
    #     )
    #     ax.annotate(
    #         gene,
    #         (x, y),
    #         fontsize=10,
    #         fontweight="bold",
    #         fontstyle="italic",
    #         color=color,
    #         zorder=5,
    #         xytext=offset,
    #         textcoords="offset points",
    #         path_effects=text_effects,
    #     )

    ax.axvline(
        -args.effect_threshold,
        color="black",
        linestyle=":",
        linewidth=0.7,
        alpha=0.5,
        zorder=0,
    )
    ax.axvline(
        args.effect_threshold,
        color="black",
        linestyle=":",
        linewidth=0.7,
        alpha=0.5,
        zorder=0,
    )

    effect_label = (
        r"$|\Delta f(\theta)|"
        rf" \geq {format_threshold(args.effect_threshold)}$"
    )
    ax.text(
        0,
        args.y_lim * 0.04,
        effect_label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    sig_p_cutoff = df.loc[sig, "p"].max()
    if pd.notna(sig_p_cutoff):
        fdr_label = f"FDR {args.q_threshold * 100:g}%"
        fdr_y = -np.log10(max(sig_p_cutoff, P_FLOOR))
        ax.axhline(
            fdr_y,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            zorder=0,
        )
        ax.text(
            args.x_lim * 0.98,
            fdr_y,
            fdr_label,
            ha="right",
            va="bottom",
            fontsize=9,
            color="red",
        )

    ax.set_xlim(-args.x_lim, args.x_lim)
    ax.set_ylim(0, args.y_lim)

    ax.set_xlabel(
        r"$f(\theta_{\mathrm{S1}}) - f(\theta_{\mathrm{T1}})$", fontsize=13
    )
    ax.set_ylabel(r"$-\log_{10}(p\text{-value})$", fontsize=13)
    ax.tick_params(labelsize=11)

    place_gene_labels(fig, ax, df, label_genes, args.label_fontsize)

    ax.legend(
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        loc="lower right",
        markerscale=1.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")

    print(f"Saved {args.output}")
    print(f"  Total genes plotted: {len(df)}")
    print(
        f"  Significant q<{args.q_threshold:g} & "
        f"|delta_diff|>={args.effect_threshold:g}: up={up.sum()}, down={down.sum()}"
    )
    print(f"  Faded (sig but |delta_diff|<{args.effect_threshold:g}): {faded.sum()}")
    if not label_genes.empty:
        print(
            "  Labeled genes: "
            + ", ".join(
                f"{row.gene} ({row.delta_diff:.2f})"
                for row in label_genes.itertuples(index=False)
            )
        )
    if args.label_exclude_regex:
        print(
            f"  Label-excluded genes matching {args.label_exclude_regex!r}: "
            f"{len(df) - int(label_candidates.sum())}"
        )
    if pd.notna(sig_p_cutoff):
        print(f"  Largest p among q-significant genes: {sig_p_cutoff:.2e}")


if __name__ == "__main__":
    main()
