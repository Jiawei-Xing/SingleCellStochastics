#!/usr/bin/env python
"""Circular top-clade tree plot colored by primary vs metastatic labels."""

import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from Bio import Phylo


sys.setrecursionlimit(10000)

COLORS = {"Pri": "#2166ac", "Met": "#b2182b"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree",
        default="top_clade/data/3724_NT_TopClade_vine_tree.nwk",
        help="Top-clade Newick tree.",
    )
    parser.add_argument(
        "--regime",
        default="top_clade/data/3724_NT_TopClade_vine_regime.csv",
        help="Node regime CSV with node_name,regime columns.",
    )
    parser.add_argument(
        "--output",
        default="top_clade/data/3724_NT_TopClade_tree_pri_met_circular.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="KP-Tracer 3724_NT Top Clade - VINE Tree\nPrimary vs Metastatic",
    )
    return parser.parse_args()


def pri_met(label):
    return "Pri" if label == "T1" else "Met"


def load_regimes(path):
    node_regime = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"node_name", "regime"}.issubset(reader.fieldnames or []):
            raise ValueError("Regime CSV must contain node_name and regime columns.")
        for row in reader:
            node_regime[row["node_name"].strip()] = row["regime"].strip()
    return node_regime


def main():
    args = parse_args()
    node_regime = load_regimes(args.regime)

    tree = Phylo.read(args.tree, "newick")
    tree.ladderize(reverse=True)

    all_clades = list(tree.find_clades(order="postorder"))
    leaves = tree.get_terminals()
    n_leaves = len(leaves)

    clade_color = {}
    for clade in all_clades:
        regime = node_regime.get(clade.name or "", "T1")
        clade_color[id(clade)] = COLORS[pri_met(regime)]

    leaf_idx = 0
    theta = {}
    for clade in all_clades:
        if clade.is_terminal():
            theta[id(clade)] = 2 * np.pi * leaf_idx / n_leaves
            leaf_idx += 1
        else:
            child_thetas = [theta[id(child)] for child in clade.clades]
            theta[id(clade)] = np.arctan2(
                np.mean([np.sin(t) for t in child_thetas]),
                np.mean([np.cos(t) for t in child_thetas]),
            )

    radius = {}
    stack = [(tree.root, 0.0)]
    while stack:
        clade, current_radius = stack.pop()
        radius[id(clade)] = current_radius
        for child in reversed(clade.clades):
            branch_length = child.branch_length if child.branch_length else 0.0
            stack.append((child, current_radius + branch_length))

    max_radius = max(radius.values())

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax.set_aspect("equal")

    branch_lw = 0.6
    arc_lw = 0.5

    for clade in all_clades:
        if clade.is_terminal():
            continue

        current_radius = radius[id(clade)]
        child_thetas = sorted(theta[id(child)] for child in clade.clades)

        if len(child_thetas) >= 2:
            theta_min, theta_max = child_thetas[0], child_thetas[-1]
            if theta_max - theta_min > np.pi:
                theta_min, theta_max = theta_max, theta_min + 2 * np.pi
            arc_theta = np.linspace(
                theta_min,
                theta_max,
                max(20, int(abs(theta_max - theta_min) * 50)),
            )
            ax.plot(
                current_radius * np.cos(arc_theta),
                current_radius * np.sin(arc_theta),
                color=clade_color[id(clade)],
                linewidth=arc_lw,
                solid_capstyle="round",
            )

        for child in clade.clades:
            child_radius = radius[id(child)]
            child_theta = theta[id(child)]
            x0 = current_radius * np.cos(child_theta)
            y0 = current_radius * np.sin(child_theta)
            x1 = child_radius * np.cos(child_theta)
            y1 = child_radius * np.sin(child_theta)
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=clade_color[id(child)],
                linewidth=branch_lw,
                solid_capstyle="round",
            )

    for leaf in leaves:
        leaf_radius = radius[id(leaf)]
        leaf_theta = theta[id(leaf)]
        ax.plot(
            leaf_radius * np.cos(leaf_theta),
            leaf_radius * np.sin(leaf_theta),
            "o",
            color=clade_color[id(leaf)],
            markersize=1.2,
            markeredgewidth=0,
        )

    ring_radius = max_radius * 1.04
    ring_width = max_radius * 0.025
    for leaf in leaves:
        leaf_theta = theta[id(leaf)]
        color = clade_color[id(leaf)]
        ax.plot(
            [
                ring_radius * np.cos(leaf_theta),
                (ring_radius + ring_width) * np.cos(leaf_theta),
            ],
            [
                ring_radius * np.sin(leaf_theta),
                (ring_radius + ring_width) * np.sin(leaf_theta),
            ],
            color=color,
            linewidth=1.0,
            solid_capstyle="butt",
        )

    handles = [
        mpatches.Patch(color=COLORS["Pri"], label="Primary"),
        mpatches.Patch(color=COLORS["Met"], label="Metastatic"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fontsize=14,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=1.5,
        handleheight=1.5,
        borderpad=0.8,
    )

    margin = max_radius * 1.12
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    ax.set_title(args.title, fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    n_pri = sum(1 for leaf in leaves if node_regime.get(leaf.name) == "T1")
    n_met = n_leaves - n_pri
    print(f"Saved {args.output}")
    print(f"  {n_pri} Primary, {n_met} Metastatic leaves")


if __name__ == "__main__":
    main()
