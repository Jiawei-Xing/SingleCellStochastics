#!/usr/bin/env python
"""Plot a circular tree with directed parsimony FP-regime labels."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from Bio import Phylo


COLORS = {
    "Rgl1": "#000000",
    "Nb_FP": "#888888",
    "Nb_DA": "#ff7f00",
    "DA": "#e31a1c",
    "Nb_GLU": "#33a02c",
    "GLU_FP": "#1f78b4",
}

LABEL_MAP = {
    "FP_Rgl1": "Rgl1",
    "FP_NbFP": "Nb_FP",
    "FP_NbDA": "Nb_DA",
    "FP_DA": "DA",
    "FP_NbGLU": "Nb_GLU",
    "FP_GLUFP": "GLU_FP",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a circular tree colored by parsimony-inferred FP regimes."
    )
    parser.add_argument("--tree", required=True, help="Named Newick tree.")
    parser.add_argument("--regime", required=True, help="Parsimony regime CSV.")
    parser.add_argument("--transitions", required=True, help="Parsimony transitions TSV.")
    parser.add_argument("--output", required=True, help="Output PNG.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def polar_xy(radius, angle):
    return radius * np.cos(angle), radius * np.sin(angle)


def circular_layout(tree):
    tree.ladderize(reverse=True)
    leaves = tree.get_terminals()
    theta = {}
    for idx, leaf in enumerate(leaves):
        theta[id(leaf)] = 2 * np.pi * idx / len(leaves)

    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            continue
        child_angles = [theta[id(child)] for child in clade.clades]
        theta[id(clade)] = float(np.mean(child_angles))

    radius = {}
    stack = [(tree.root, 0.0)]
    while stack:
        clade, parent_radius = stack.pop()
        radius[id(clade)] = parent_radius
        for child in reversed(clade.clades):
            stack.append((child, parent_radius + (child.branch_length or 0.0)))
    return theta, radius


def main():
    args = parse_args()

    tree = Phylo.read(args.tree, "newick")
    regime_df = pd.read_csv(args.regime)
    transitions = pd.read_csv(args.transitions, sep="\t")

    regime_df["regime"] = regime_df["regime"].map(lambda r: LABEL_MAP.get(r, r))
    regimes = dict(zip(regime_df["node_name"], regime_df["regime"]))
    transition_by_child = transitions.set_index("child").to_dict("index")

    theta, radius = circular_layout(tree)
    max_radius = max(radius.values()) or 1.0

    branch_segments = {state: [] for state in COLORS}
    arc_segments = {state: [] for state in COLORS}
    change_segments = []
    change_points = []
    backward_points = []

    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            continue
        parent_regime = regimes.get(clade.name, "Rgl1")
        parent_radius = radius[id(clade)]
        child_angles = sorted(theta[id(child)] for child in clade.clades)
        if len(child_angles) >= 2 and parent_radius > 0:
            arc_angles = np.linspace(child_angles[0], child_angles[-1], 32)
            points = np.column_stack(polar_xy(parent_radius, arc_angles))
            arc_segments[parent_regime].extend(zip(points[:-1], points[1:]))

        for child in clade.clades:
            child_regime = regimes.get(child.name, parent_regime)
            angle = theta[id(child)]
            start = polar_xy(parent_radius, angle)
            end = polar_xy(radius[id(child)], angle)
            segment = [start, end]
            branch_segments[child_regime].append(segment)

            info = transition_by_child.get(child.name)
            if info and info["direction"] != "same":
                change_segments.append(segment)
                mid_radius = parent_radius + 0.55 * (radius[id(child)] - parent_radius)
                mid = polar_xy(mid_radius, angle)
                if info["direction"] == "backward":
                    backward_points.append(mid)
                else:
                    change_points.append(mid)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=args.dpi)
    ax.set_aspect("equal")

    for state, segments in arc_segments.items():
        if segments:
            ax.add_collection(
                LineCollection(segments, colors=COLORS[state], linewidths=1.6, alpha=0.95)
            )
    for state, segments in branch_segments.items():
        if segments:
            ax.add_collection(
                LineCollection(segments, colors=COLORS[state], linewidths=2.0, alpha=1.0)
            )

    if change_segments:
        ax.add_collection(
            LineCollection(change_segments, colors="#111111", linewidths=0.6, alpha=0.5)
        )
    if change_points:
        x, y = zip(*change_points)
        ax.scatter(x, y, s=18, c="#111111", marker="o", linewidths=0, zorder=5)
    if backward_points:
        x, y = zip(*backward_points)
        ax.scatter(x, y, s=50, c="#ffffff", edgecolors="#111111", marker="X", linewidths=1.2, zorder=6)

    leaves = tree.get_terminals()
    tip_x, tip_y, tip_colors = [], [], []
    for leaf in leaves:
        state = regimes[leaf.name]
        x, y = polar_xy(radius[id(leaf)], theta[id(leaf)])
        tip_x.append(x)
        tip_y.append(y)
        tip_colors.append(COLORS[state])
    ax.scatter(tip_x, tip_y, s=22, c=tip_colors, linewidths=0, zorder=7)

    margin = max_radius * 1.17
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    ax.set_title("E15_rep1 FP Lineage: Directed Parsimony Regimes", fontsize=13, pad=14)

    regime_counts = regime_df["regime"].value_counts().to_dict()
    handles = [
        mpatches.Patch(color=COLORS[state], label=f"{state}") #({regime_counts.get(state, 0)})")
        for state in COLORS
        if state in regime_counts
    ]
    #handles.extend(
    #    [
    #        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markersize=5, label="Differentiation"),
    #        #plt.Line2D([0], [0], marker="X", color="#111111", markerfacecolor="#ffffff", markersize=6, label="backward transition"),
    #    ]
    #)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=20,
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
