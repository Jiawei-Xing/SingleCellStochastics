#!/usr/bin/env python
"""Plot predicted-expression reconstruction panels without read-count rings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from singlecellstochastics.reconstruct import (
    get_all_nodes,
    get_leaves,
    ladderize,
    layout_circular_tree,
    load_tree_from_newick,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw reconstruction predicted-expression trees without the "
            "outer read-count ring."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("ighg1_vs_igha1"),
        help="IGHG1-vs-IGHA1 work directory [default: %(default)s]",
    )
    parser.add_argument(
        "--reconstruct-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing tables from reconstruction "
            "[default: <work-dir>/reconstruct/joint_h1_xbp1_jchain]"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory [default: <reconstruct-dir>/plots_predicted_expression_no_outer]",
    )
    parser.add_argument(
        "--clones",
        nargs="+",
        default=["Clonotype_1261", "Clonotype_133"],
        help="Clone names to plot [default: %(default)s]",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=["JCHAIN", "XBP1"],
        help="Gene symbols to plot [default: %(default)s]",
    )
    return parser.parse_args()


def add_ext(prefix: Path, ext: str) -> Path:
    return prefix.parent / f"{prefix.name}{ext}"


def load_reconstruction(tree_path: Path, table_path: Path):
    root = load_tree_from_newick(str(tree_path), normalize=True)
    recon = pd.read_csv(table_path, sep="\t")
    recon = recon.set_index("node_name")

    for node in get_all_nodes(root):
        if node.name not in recon.index:
            raise ValueError(f"Node {node.name!r} missing from {table_path}")
        row = recon.loc[node.name]
        node.true_mu = float(row["infer_mu"])
        node.true_var = float(row["infer_var"])
        if "regime" in row:
            node.regime = str(row["regime"])

    ladderize(root, reverse=True)
    leaves = get_leaves(root)
    angles = list(np.linspace(0, 2 * np.pi, len(leaves), endpoint=False))
    layout_circular_tree(root, r_current=0.0, leaf_angles=angles)
    return root


def plot_predicted_expression(root, output_prefix: Path, title: str) -> None:
    all_nodes = get_all_nodes(root)
    max_r = max(node.r for node in all_nodes)
    mus = [node.true_mu for node in all_nodes]
    mu_min, mu_max = min(mus), max(mus)

    root_mu = root.true_mu
    eps = max(abs(root_mu) * 1e-3, 1e-6)
    vmin_mu = min(mu_min, root_mu - eps)
    vmax_mu = max(mu_max, root_mu + eps)
    cmap_mu = plt.cm.RdBu_r
    norm_mu = mcolors.TwoSlopeNorm(vcenter=root_mu, vmin=vmin_mu, vmax=vmax_mu)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=220)
    ax.set_aspect("equal")
    ax.axis("off")

    tree_linewidth = 3.0

    def draw_tree(node):
        if not node.children:
            return
        cr = node.r
        col_parent = cmap_mu(norm_mu(node.true_mu))

        child_thetas = sorted(child.theta for child in node.children)
        if len(child_thetas) >= 2:
            t_min, t_max = child_thetas[0], child_thetas[-1]
            if t_max - t_min > np.pi:
                t_min, t_max = t_max, t_min + 2 * np.pi
            arc_t = np.linspace(t_min, t_max, max(20, int(abs(t_max - t_min) * 50)))
            ax.plot(
                cr * np.cos(arc_t),
                cr * np.sin(arc_t),
                color=col_parent,
                linewidth=tree_linewidth,
                solid_capstyle="round",
            )

        for child in node.children:
            col_child = cmap_mu(norm_mu(child.true_mu))
            n_seg = 20
            r_vals = np.linspace(cr, child.r, n_seg + 1)
            c1 = np.array(matplotlib.colors.to_rgba(col_parent))
            c2 = np.array(matplotlib.colors.to_rgba(col_child))
            for i in range(n_seg):
                frac = i / n_seg
                col = c1 * (1 - frac) + c2 * frac
                x0 = r_vals[i] * np.cos(child.theta)
                y0 = r_vals[i] * np.sin(child.theta)
                x1 = r_vals[i + 1] * np.cos(child.theta)
                y1 = r_vals[i + 1] * np.sin(child.theta)
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=col,
                    linewidth=tree_linewidth,
                    solid_capstyle="butt",
                )
            draw_tree(child)

    draw_tree(root)

    margin = max_r * 1.12
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    sm = plt.cm.ScalarMappable(norm=norm_mu, cmap=cmap_mu)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.06, shrink=0.75)
    cbar.ax.set_title("Predicted expression", loc="left", fontsize=11)

    fig.tight_layout()
    fig.savefig(add_ext(output_prefix, ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(add_ext(output_prefix, ".pdf"), bbox_inches="tight")
    fig.savefig(add_ext(output_prefix, ".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reconstruct_dir = args.reconstruct_dir or args.work_dir / "reconstruct" / "joint_h1_xbp1_jchain"
    out_dir = args.out_dir or reconstruct_dir / "plots_predicted_expression_no_outer"
    out_dir.mkdir(parents=True, exist_ok=True)

    for clone in args.clones:
        tree_path = args.work_dir / "newick" / f"{clone}.nwk"
        for gene in args.genes:
            table_path = reconstruct_dir / "tables" / f"{clone}.{gene}.h1_reconstruction.tsv"
            if not tree_path.exists():
                raise FileNotFoundError(tree_path)
            if not table_path.exists():
                raise FileNotFoundError(table_path)
            root = load_reconstruction(tree_path, table_path)
            prefix = out_dir / f"{clone}.{gene}.predicted_expression.no_outer"
            plot_predicted_expression(root, prefix, f"{clone} {gene} Predicted Expression")
            print(f"Wrote {add_ext(prefix, '.png')}")
            print(f"Wrote {add_ext(prefix, '.pdf')}")
            print(f"Wrote {add_ext(prefix, '.svg')}")


if __name__ == "__main__":
    main()
