#!/usr/bin/env python
"""Plot IGHG1-vs-IGHA1 trees with the same ladderized layout as reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from singlecellstochastics.reconstruct import (
    apply_regimes,
    get_all_nodes,
    get_leaves,
    ladderize,
    layout_circular_tree,
    load_tree_from_newick,
    propagate_regimes,
)


PALETTE = {
    "IGHG1": "#2ca02c",
    "IGHA1": "#d62728",
    "default": "#b8c0c8",
}


def add_ext(prefix: Path, ext: str) -> Path:
    return prefix.parent / f"{prefix.name}{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regime-colored trees in the same ladderized layout used by lavous-reconstruct."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("ighg1_vs_igha1"),
        help="IGHG1-vs-IGHA1 work directory [default: %(default)s]",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory [default: <work-dir>/plots_ladderized]",
    )
    parser.add_argument(
        "--clones",
        nargs="+",
        default=["Clonotype_1261", "Clonotype_133"],
        help="Clone names to plot [default: %(default)s]",
    )
    return parser.parse_args()


def prepare_tree(tree_path: Path, regime_path: Path):
    root = load_tree_from_newick(str(tree_path), normalize=True)
    apply_regimes(root, str(regime_path))
    propagate_regimes(root)
    ladderize(root, reverse=True)

    leaves = get_leaves(root)
    angles = list(np.linspace(0, 2 * np.pi, len(leaves), endpoint=False))
    layout_circular_tree(root, r_current=0.0, leaf_angles=angles)
    return root


def plot_tree(root, output_prefix: Path, title: str) -> None:
    all_nodes = get_all_nodes(root)
    leaves = get_leaves(root)
    max_r = max(node.r for node in all_nodes)

    counts: dict[str, int] = {}
    for leaf in leaves:
        counts[leaf.regime] = counts.get(leaf.regime, 0) + 1

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=220)
    ax.set_aspect("equal")
    ax.axis("off")

    tree_lw = 2.6

    def color_for(node):
        return PALETTE.get(node.regime, PALETTE["default"])

    def draw_node(node):
        if not node.children:
            return

        child_thetas = sorted(child.theta for child in node.children)
        if len(child_thetas) >= 2:
            t_min, t_max = child_thetas[0], child_thetas[-1]
            if t_max - t_min > np.pi:
                t_min, t_max = t_max, t_min + 2 * np.pi
            arc_t = np.linspace(t_min, t_max, max(20, int(abs(t_max - t_min) * 50)))
            ax.plot(
                node.r * np.cos(arc_t),
                node.r * np.sin(arc_t),
                color=color_for(node),
                linewidth=tree_lw,
                solid_capstyle="round",
            )

        for child in node.children:
            r_vals = np.linspace(node.r, child.r, 2)
            ax.plot(
                r_vals * np.cos(child.theta),
                r_vals * np.sin(child.theta),
                color=color_for(child),
                linewidth=tree_lw,
                solid_capstyle="butt",
            )
            draw_node(child)

    draw_node(root)

    for leaf in leaves:
        ax.scatter(
            leaf.r * np.cos(leaf.theta),
            leaf.r * np.sin(leaf.theta),
            s=54,
            color=color_for(leaf),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )

    margin = max_r * 1.14
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    legend_items = [
        Line2D(
            [0],
            [0],
            color=PALETTE[regime],
            lw=4,
            marker="o",
            markersize=5,
            label=f"{regime} tips: {counts.get(regime, 0)}",
        )
        for regime in ("IGHG1", "IGHA1")
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(add_ext(output_prefix, ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(add_ext(output_prefix, ".pdf"), bbox_inches="tight")
    fig.savefig(add_ext(output_prefix, ".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.work_dir / "plots_ladderized"
    out_dir.mkdir(parents=True, exist_ok=True)

    for clone in args.clones:
        tree_path = args.work_dir / "newick" / f"{clone}.nwk"
        regime_path = args.work_dir / "regimes" / f"{clone}.ighg1_vs_igha1.regime.csv"
        if not tree_path.exists():
            raise FileNotFoundError(tree_path)
        if not regime_path.exists():
            raise FileNotFoundError(regime_path)

        root = prepare_tree(tree_path, regime_path)
        output_prefix = out_dir / f"{clone}.ighg1_vs_igha1.ladderized"
        plot_tree(root, output_prefix, f"{clone} IGHG1 vs IGHA1")
        print(f"Wrote {add_ext(output_prefix, '.png')}")
        print(f"Wrote {add_ext(output_prefix, '.pdf')}")
        print(f"Wrote {add_ext(output_prefix, '.svg')}")


if __name__ == "__main__":
    main()
