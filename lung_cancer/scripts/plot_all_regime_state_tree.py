#!/usr/bin/env python
"""Circular plot of the first VINE state tree with all regimes shown."""

import argparse
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from Bio import Phylo


sys.setrecursionlimit(10000)

DEFAULT_NEXUS = (
    "/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/"
    "test_real_data/Yang_KP-Tracer/VINE_new/3724_NT_All_vine_states.nex"
)

STATE_RE = re.compile(r"state=([^,\]\s]+)")
STATE_COLORS = {
    "T1": "#2166ac",
    "S1": "#b2182b",
    "L1": "#4daf4a",
    "L2": "#ff7f00",
    "L3": "#6a3d9a",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nexus",
        default=DEFAULT_NEXUS,
        help="VINE NEXUS state-tree file; only the first tree is plotted.",
    )
    parser.add_argument(
        "--output",
        default=(
            "top_clade/data/"
            "3724_NT_All_vine_states_first_tree_all_regimes_circular.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="KP-Tracer 3724_NT - VINE State Tree 1\nAll Regimes",
    )
    return parser.parse_args()


def clade_state(clade):
    comment = getattr(clade, "comment", None) or ""
    match = STATE_RE.search(comment)
    if match:
        return match.group(1)
    return "unknown"


def color_for_state(state):
    return STATE_COLORS.get(state, "#777777")


def main():
    args = parse_args()

    tree = next(Phylo.parse(args.nexus, "nexus"))
    tree.ladderize(reverse=True)

    all_clades = list(tree.find_clades(order="postorder"))
    leaves = tree.get_terminals()
    n_leaves = len(leaves)

    states = {id(clade): clade_state(clade) for clade in all_clades}
    state_counts = Counter(states.values())
    leaf_state_counts = Counter(clade_state(leaf) for leaf in leaves)

    clade_color = {
        id(clade): color_for_state(states[id(clade)]) for clade in all_clades
    }

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

    ordered_states = [
        state for state in ["T1", "S1", "L1", "L2", "L3"] if state in state_counts
    ]
    ordered_states.extend(
        sorted(state for state in state_counts if state not in set(ordered_states))
    )
    handles = [
        mpatches.Patch(
            color=color_for_state(state),
            label=f"{state} (n={leaf_state_counts.get(state, 0)})",
        )
        for state in ordered_states
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fontsize=12,
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

    print(f"Saved {args.output}")
    print("Leaf states:")
    for state in ordered_states:
        print(f"  {state}: {leaf_state_counts.get(state, 0)}")


if __name__ == "__main__":
    main()
