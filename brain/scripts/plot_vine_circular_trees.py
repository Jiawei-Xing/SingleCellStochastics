#!/usr/bin/env python
"""Plot circular ladderized trees from VINE Newick outputs."""

import argparse
import csv
import os
from io import StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from Bio import Phylo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot circular ladderized trees from a VINE manifest or a single tree."
    )
    parser.add_argument(
        "--manifest",
        default="data/vine/crest_clone_manifest.tsv",
        help="VINE manifest written by prepare_crest_vine_inputs.R (manifest mode).",
    )
    parser.add_argument(
        "--outdir",
        default="data/vine",
        help="Directory for output PNGs (manifest mode).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output PNG DPI.")
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional TSV with columns cell_type and group for collapsing labels.",
    )
    parser.add_argument(
        "--suffix",
        default="circular_ladderized",
        help="Output filename suffix, without .png (manifest mode).",
    )
    # Single-tree direct mode: bypass manifest by providing newick + labels + output.
    parser.add_argument("--newick", default=None, help="Single tree mode: Newick file.")
    parser.add_argument(
        "--labels",
        default=None,
        help="Single tree mode: leaf statelabels CSV (leaf_id,state).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Single tree mode: optional clone metadata TSV with leaf_id and "
            "clone_size/cell_type_counts for the clone-size ring. If omitted, "
            "a sibling *_metadata.tsv is used when present."
        ),
    )
    parser.add_argument("--out", default=None, help="Single tree mode: output PNG path.")
    parser.add_argument("--title", default=None, help="Single tree mode: plot title.")
    return parser.parse_args()


def first_newick(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No Newick tree found in {path}")


def read_manifest(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_label_map(path):
    if not path:
        return {}
    with open(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["cell_type"]: row["group"] for row in rows}


def read_state_labels(path, label_map):
    labels = {}
    with open(path) as handle:
        for leaf, state in csv.reader(handle):
            labels[leaf] = label_map.get(state, state)
    return labels


def infer_metadata_path(labels_path):
    if not labels_path:
        return None
    root, _ = os.path.splitext(labels_path)
    if root.endswith("_statelabels"):
        candidate = root[: -len("_statelabels")] + "_metadata.tsv"
        if os.path.exists(candidate):
            return candidate
    return None


def parse_cell_type_count_total(counts):
    total = 0
    for part in counts.split(";"):
        if not part or ":" not in part:
            continue
        _, count = part.rsplit(":", 1)
        try:
            total += int(count)
        except ValueError:
            return None
    return total or None


def parse_clone_size(row):
    clone_size = row.get("clone_size", "")
    if clone_size not in ("", None):
        try:
            return max(int(float(clone_size)), 1)
        except ValueError:
            pass

    counts = row.get("cell_type_counts", "")
    if counts:
        return parse_cell_type_count_total(counts)

    return None


def read_clone_sizes(path):
    if not path:
        return {}

    sizes = {}
    with open(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            leaf = row.get("leaf_id") or row.get("clone_id")
            if not leaf:
                continue
            clone_size = parse_clone_size(row)
            if clone_size is not None:
                sizes[leaf] = clone_size
    return sizes


def clone_size_color(clone_size, max_clone_size):
    if max_clone_size <= 1:
        shade = 1.0
    else:
        scaled = np.log(float(clone_size)) / np.log(float(max_clone_size))
        shade = 1.0 - 0.85 * scaled
    return (shade, shade, shade, 1.0)


def clone_size_legend_values(clone_sizes):
    values = sorted(set(clone_sizes))
    if len(values) <= 3:
        return values
    median = int(np.median(values))
    return sorted(set([values[0], median, values[-1]]))


def palette(states):
    cmaps = [plt.get_cmap(name) for name in ("tab20", "tab20b", "tab20c")]
    colors = []
    for cmap in cmaps:
        colors.extend(cmap(i) for i in range(cmap.N))
    return {state: colors[i % len(colors)] for i, state in enumerate(states)}


def polar_xy(radius, angle):
    return radius * np.cos(angle), radius * np.sin(angle)


def plot_tree(
    newick_path, labels_path, output_path, title, dpi, label_map, metadata_path=None
):
    tree = Phylo.read(StringIO(first_newick(newick_path)), "newick")
    tree.ladderize(reverse=True)

    labels = read_state_labels(labels_path, label_map)
    metadata_path = metadata_path or infer_metadata_path(labels_path)
    clone_sizes = read_clone_sizes(metadata_path)
    leaves = tree.get_terminals()
    n_leaves = len(leaves)
    all_clades = list(tree.find_clades(order="postorder"))

    leaf_states = [labels.get(leaf.name, "unknown") for leaf in leaves]
    state_counts = {state: leaf_states.count(state) for state in set(leaf_states)}
    states = sorted(state_counts, key=lambda s: (-state_counts[s], s))
    colors = palette(states)

    theta = {}
    for idx, leaf in enumerate(leaves):
        theta[id(leaf)] = 2 * np.pi * idx / n_leaves
    for clade in all_clades:
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
            branch_length = child.branch_length or 0.0
            stack.append((child, parent_radius + branch_length))
    max_radius = max(radius.values()) or 1.0

    radial_segments = []
    arc_segments = []
    branch_color = "#333333"

    for clade in all_clades:
        if clade.is_terminal():
            continue
        parent_radius = radius[id(clade)]
        child_angles = sorted(theta[id(child)] for child in clade.clades)
        if len(child_angles) >= 2 and parent_radius > 0:
            arc_angles = np.linspace(child_angles[0], child_angles[-1], 36)
            points = np.column_stack(polar_xy(parent_radius, arc_angles))
            arc_segments.extend(zip(points[:-1], points[1:]))

        for child in clade.clades:
            child_angle = theta[id(child)]
            start = polar_xy(parent_radius, child_angle)
            end = polar_xy(radius[id(child)], child_angle)
            radial_segments.append([start, end])

    fig_size = 12 if n_leaves < 2400 else 14
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")

    ax.add_collection(
        LineCollection(
            radial_segments, colors=branch_color, linewidths=0.16, alpha=0.82
        )
    )
    ax.add_collection(
        LineCollection(arc_segments, colors=branch_color, linewidths=0.14, alpha=0.68)
    )

    tip_r = max_radius
    tip_x = []
    tip_y = []
    tip_colors = []
    for leaf in leaves:
        state = labels.get(leaf.name, "unknown")
        x, y = polar_xy(tip_r, theta[id(leaf)])
        tip_x.append(x)
        tip_y.append(y)
        tip_colors.append(colors[state])
    ax.scatter(tip_x, tip_y, s=1.0, c=tip_colors, linewidths=0, zorder=3)

    ring_inner = max_radius * 1.018
    ring_outer = max_radius * 1.075
    ring_segments = []
    ring_colors = []
    for leaf in leaves:
        angle = theta[id(leaf)]
        ring_segments.append([polar_xy(ring_inner, angle), polar_xy(ring_outer, angle)])
        ring_colors.append(colors[labels.get(leaf.name, "unknown")])
    ax.add_collection(LineCollection(ring_segments, colors=ring_colors, linewidths=0.75))

    observed_clone_sizes = [
        clone_sizes[leaf.name] for leaf in leaves if leaf.name in clone_sizes
    ]
    max_clone_size = max(observed_clone_sizes) if observed_clone_sizes else 1
    if observed_clone_sizes:
        clone_size_inner = max_radius * 1.088
        clone_size_outer = max_radius * 1.132
        clone_size_segments = []
        clone_size_ring_colors = []
        for leaf in leaves:
            clone_size = clone_sizes.get(leaf.name)
            if clone_size is None:
                continue
            angle = theta[id(leaf)]
            clone_size_segments.append(
                [polar_xy(clone_size_inner, angle), polar_xy(clone_size_outer, angle)]
            )
            clone_size_ring_colors.append(clone_size_color(clone_size, max_clone_size))
        ax.add_collection(
            LineCollection(
                clone_size_segments,
                colors=clone_size_ring_colors,
                linewidths=0.75,
                zorder=4,
            )
        )

    margin = max_radius * 1.24
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=14)

    handles = [
        mpatches.Patch(color=colors[state], label=f"{state} ({state_counts[state]})")
        for state in states
    ]
    state_legend = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=7,
        ncol=1,
        borderaxespad=0.0,
    )
    ax.add_artist(state_legend)

    if observed_clone_sizes:
        clone_size_handles = [
            mpatches.Patch(
                color=clone_size_color(value, max_clone_size),
                label=f"{value} cell" if value == 1 else f"{value} cells",
            )
            for value in clone_size_legend_values(observed_clone_sizes)
        ]
        ax.legend(
            handles=clone_size_handles,
            title="Cells / clone",
            loc="center left",
            bbox_to_anchor=(1.01, 0.08),
            frameon=False,
            fontsize=7,
            title_fontsize=7,
            ncol=1,
            borderaxespad=0.0,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    label_map = read_label_map(args.label_map)

    if args.newick is not None:
        if not (args.labels and args.out):
            raise SystemExit("--newick requires --labels and --out as well.")
        title = args.title or os.path.splitext(os.path.basename(args.newick))[0]
        plot_tree(
            args.newick,
            args.labels,
            args.out,
            title,
            args.dpi,
            label_map,
            args.metadata,
        )
        print(f"Saved {args.out}")
        return

    rows = read_manifest(args.manifest)
    outputs = []
    for row in rows:
        prefix = f"{row['dataset']}_{row['unit']}"
        newick = os.path.join(
            os.path.dirname(args.manifest), f"{prefix}_vine_samples.nwk"
        )
        labels = row["statelabels"]
        metadata = row.get("metadata") or None
        output = os.path.join(args.outdir, f"{prefix}_{args.suffix}.png")
        title = f"{row['dataset']} {row['unit']} VINE tree ({row['n_leaves']} leaves)"
        plot_tree(newick, labels, output, title, args.dpi, label_map, metadata)
        outputs.append(output)
        print(f"Saved {output}")

    with open(os.path.join(args.outdir, f"{args.suffix}_plots.txt"), "w") as handle:
        for output in outputs:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()
