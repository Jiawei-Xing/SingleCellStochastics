#!/usr/bin/env python3
"""Prepare pruned IGHG1-vs-IGHA1 LAVOUS inputs for two B-cell clonotypes.

This workflow starts from the original, non-CSR-ranked regime calls in
``regimes/`` and the existing full count matrices in
``lavous_ighg1_vs_other/``.  It then:

* forces the root regime to IGHG1,
* removes IGHG2 lineages,
* removes IGHG1 lineages whose nearest valid ancestry passes through IGHA1,
* writes pruned Newicks, regime CSVs, count matrices, library factors, and SVG
  tree plots under ``ighg1_vs_igha1/``.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CLONES = ["Clonotype_1261", "Clonotype_133"]
TARGET_REGIMES = {"IGHG1", "IGHA1"}
PALETTE = {
    "IGHG1": "#2ca02c",
    "IGHA1": "#d62728",
    "IGHG2": "#17becf",
    "OTHER": "#777777",
}


@dataclass
class Node:
    name: str | None = None
    length: float | None = None
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None

    def is_leaf(self) -> bool:
        return not self.children


def parse_newick(text: str) -> Node:
    text = text.strip()
    pos = 0

    def skip_ws() -> None:
        nonlocal pos
        while pos < len(text) and text[pos].isspace():
            pos += 1

    def parse_label() -> str | None:
        nonlocal pos
        skip_ws()
        start = pos
        while pos < len(text) and text[pos] not in ":,();":
            pos += 1
        label = text[start:pos].strip()
        return label or None

    def parse_length() -> float | None:
        nonlocal pos
        skip_ws()
        if pos >= len(text) or text[pos] != ":":
            return None
        pos += 1
        start = pos
        while pos < len(text) and text[pos] not in ",();":
            pos += 1
        raw = text[start:pos].strip()
        return float(raw) if raw else None

    def parse_subtree() -> Node:
        nonlocal pos
        skip_ws()
        if pos < len(text) and text[pos] == "(":
            pos += 1
            children: list[Node] = []
            while True:
                child = parse_subtree()
                children.append(child)
                skip_ws()
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                if pos < len(text) and text[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"Unexpected Newick character at {pos}: {text[pos:pos+20]!r}")
            node = Node(name=parse_label(), length=parse_length(), children=children)
            for child in children:
                child.parent = node
            return node
        node = Node(name=parse_label(), length=parse_length())
        if node.name is None:
            raise ValueError(f"Leaf without a name near character {pos}")
        return node

    root = parse_subtree()
    skip_ws()
    if pos < len(text) and text[pos] == ";":
        pos += 1
    skip_ws()
    if pos != len(text):
        raise ValueError(f"Trailing Newick text at character {pos}: {text[pos:pos+20]!r}")
    root.length = None
    root.parent = None
    return root


def iter_nodes(node: Node):
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def iter_leaves(node: Node):
    if node.is_leaf():
        yield node
    else:
        for child in node.children:
            yield from iter_leaves(child)


def leaf_names(node: Node) -> list[str]:
    return [leaf.name for leaf in iter_leaves(node) if leaf.name is not None]


def count_leaves(node: Node) -> int:
    return sum(1 for _ in iter_leaves(node))


def read_regimes(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["node_name"]: row["regime"] for row in reader}


def write_regimes(path: Path, root: Node, regimes: dict[str, str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_name", "regime"])
        for node in iter_nodes(root):
            if node.name is None:
                raise ValueError("Cannot write unnamed node to regime CSV")
            writer.writerow([node.name, regimes[node.name]])


def fmt_length(length: float | None) -> str:
    if length is None:
        return ""
    return f":{length:.10g}"


def write_newick(root: Node, path: Path) -> None:
    def rec(node: Node) -> str:
        name = node.name or ""
        if node.children:
            return "(" + ",".join(rec(child) for child in node.children) + ")" + name + fmt_length(node.length)
        return name + fmt_length(node.length)

    path.write_text(rec(root) + ";\n")


def subtree_leaf_names(node: Node) -> list[str]:
    return [leaf.name for leaf in iter_leaves(node) if leaf.name is not None]


def prune_tree(
    node: Node,
    regimes: dict[str, str],
    ancestor_has_igha1: bool = False,
    drop_counts: Counter | None = None,
    dropped_tips: dict[str, str] | None = None,
    is_root: bool = True,
) -> Node | None:
    if drop_counts is None:
        drop_counts = Counter()
    if dropped_tips is None:
        dropped_tips = {}
    if node.name is None:
        raise ValueError("All nodes must be named before pruning")
    regime = regimes.get(node.name)
    if regime is None:
        raise ValueError(f"Node {node.name!r} is missing from regime map")

    reason = None
    if regime == "IGHG2":
        reason = "IGHG2"
    elif regime not in TARGET_REGIMES:
        reason = f"non_target_{regime}"
    elif regime == "IGHG1" and ancestor_has_igha1:
        reason = "IGHG1_after_IGHA1"

    if reason is not None:
        tips = subtree_leaf_names(node)
        drop_counts[reason] += len(tips)
        for tip in tips:
            dropped_tips[tip] = reason
        return None

    next_ancestor_has_igha1 = ancestor_has_igha1 or (regime == "IGHA1")
    kept_children = []
    for child in node.children:
        kept = prune_tree(
            child,
            regimes,
            ancestor_has_igha1=next_ancestor_has_igha1,
            drop_counts=drop_counts,
            dropped_tips=dropped_tips,
            is_root=False,
        )
        if kept is not None:
            kept_children.append(kept)

    if node.children and not kept_children:
        return None

    new_node = Node(name=node.name, length=node.length, children=kept_children)
    for child in kept_children:
        child.parent = new_node
    if is_root:
        new_node.length = None
    return new_node


def collapse_unary(node: Node, is_root: bool = True) -> Node:
    collapsed_children = [collapse_unary(child, is_root=False) for child in node.children]
    node.children = collapsed_children
    for child in collapsed_children:
        child.parent = node
    if not is_root and len(node.children) == 1:
        child = node.children[0]
        child.length = (child.length or 0.0) + (node.length or 0.0)
        child.parent = node.parent
        return child
    return node


def read_count_subset(path: Path, keep: set[str]) -> tuple[list[str], dict[str, list[int]]]:
    rows: dict[str, list[int]] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        genes = header[1:]
        for row in reader:
            if row and row[0] in keep:
                rows[row[0]] = [int(float(value)) for value in row[1:]]
    missing = sorted(keep - set(rows))
    if missing:
        raise RuntimeError(f"{path} is missing {len(missing)} kept tree tips; first: {missing[:3]}")
    return genes, rows


def read_library(path: Path, keep: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if row and row[0] in keep:
                out[row[0]] = float(row[1])
    missing = sorted(keep - set(out))
    if missing:
        raise RuntimeError(f"{path} is missing {len(missing)} kept library factors; first: {missing[:3]}")
    return out


def read_annotation(path: Path) -> dict[str, str]:
    annot = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                annot[row[0]] = row[1]
    return annot


def write_count_matrix(
    path: Path,
    cells: list[str],
    genes: list[str],
    rows: dict[str, list[int]],
    keep_gene: list[bool],
) -> None:
    kept_genes = [gene for gene, keep in zip(genes, keep_gene) if keep]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["cell", *kept_genes])
        for cell in cells:
            values = rows[cell]
            writer.writerow([cell, *[value for value, keep in zip(values, keep_gene) if keep]])


def write_library(path: Path, cells: list[str], factors: dict[str, float], median_factor: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for cell in cells:
            writer.writerow([cell, f"{factors[cell] / median_factor:.10g}"])


def write_annotation(path: Path, genes: list[str], symbols: dict[str, str], keep_gene: list[bool]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for gene, keep in zip(genes, keep_gene):
            if keep:
                writer.writerow([gene, symbols.get(gene, gene)])


def compute_gene_filters(
    clone_rows: dict[str, dict[str, list[int]]],
    genes: list[str],
    symbols: dict[str, str],
    min_cells: int,
) -> tuple[list[bool], list[bool]]:
    expressed = [0] * len(genes)
    for rows in clone_rows.values():
        for values in rows.values():
            for idx, value in enumerate(values):
                if value > 0:
                    expressed[idx] += 1
    full_keep = [count >= min_cells for count in expressed]
    no_ig_keep = [
        keep
        and not (
            symbols.get(gene, gene).startswith("IGH")
            or symbols.get(gene, gene).startswith("IGK")
            or symbols.get(gene, gene).startswith("IGL")
        )
        for gene, keep in zip(genes, full_keep)
    ]
    return full_keep, no_ig_keep


def max_depth(root: Node) -> float:
    best = 0.0

    def rec(node: Node, depth: float) -> None:
        nonlocal best
        best = max(best, depth)
        for child in node.children:
            rec(child, depth + (child.length or 0.0))

    rec(root, 0.0)
    return best or 1.0


def node_depths(root: Node) -> dict[str, float]:
    out = {}

    def rec(node: Node, depth: float) -> None:
        if node.name is not None:
            out[node.name] = depth
        for child in node.children:
            rec(child, depth + (child.length or 0.0))

    rec(root, 0.0)
    return out


def write_svg_plot(path: Path, clone: str, root: Node, regimes: dict[str, str]) -> None:
    leaves = leaf_names(root)
    leaf_step = 16
    width = 920
    height = max(320, 110 + leaf_step * max(1, len(leaves)))
    left = 58
    right = 150
    top = 58
    bottom = 44
    plot_width = width - left - right
    depth_by_name = node_depths(root)
    max_d = max_depth(root)
    y_by_name: dict[str, float] = {}
    x_by_name = {
        name: left + (depth / max_d) * plot_width for name, depth in depth_by_name.items()
    }

    for idx, leaf in enumerate(leaves):
        y_by_name[leaf] = top + idx * leaf_step

    def assign_internal_y(node: Node) -> float:
        if node.name in y_by_name:
            return y_by_name[node.name]
        child_ys = [assign_internal_y(child) for child in node.children]
        y_by_name[node.name] = sum(child_ys) / len(child_ys)
        return y_by_name[node.name]

    assign_internal_y(root)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-family="Arial, sans-serif" font-size="16" font-weight="700">{html.escape(clone)}: IGHG1 vs IGHA1 ({len(leaves)} tips)</text>',
    ]

    def draw_edges(node: Node) -> None:
        if node.name is None:
            return
        x0 = x_by_name[node.name]
        y0 = y_by_name[node.name]
        for child in node.children:
            if child.name is None:
                continue
            x1 = x_by_name[child.name]
            y1 = y_by_name[child.name]
            color = PALETTE.get(regimes.get(child.name, "OTHER"), PALETTE["OTHER"])
            lines.append(
                f'<path d="M {x0:.2f} {y0:.2f} V {y1:.2f} H {x1:.2f}" '
                f'stroke="{color}" stroke-width="1.4" fill="none" stroke-linecap="round"/>'
            )
            draw_edges(child)

    draw_edges(root)

    for node in iter_nodes(root):
        if node.name is None:
            continue
        regime = regimes[node.name]
        color = PALETTE.get(regime, PALETTE["OTHER"])
        radius = 3.2 if node.is_leaf() else 2.0
        lines.append(
            f'<circle cx="{x_by_name[node.name]:.2f}" cy="{y_by_name[node.name]:.2f}" '
            f'r="{radius}" fill="{color}" stroke="white" stroke-width="0.6"/>'
        )

    legend_x = width - right + 24
    legend_y = top
    for idx, regime in enumerate(["IGHG1", "IGHA1"]):
        y = legend_y + idx * 24
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{PALETTE[regime]}"/>')
        lines.append(
            f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Arial, sans-serif" '
            f'font-size="12">{regime}</text>'
        )

    lines.append(
        f'<text x="{left}" y="{height - bottom + 20}" font-family="Arial, sans-serif" '
        f'font-size="11" fill="#555">Root regime forced to IGHG1; IGHG2 and IGHG1-after-IGHA1 lineages pruned.</text>'
    )
    lines.append("</svg>\n")
    path.write_text("\n".join(lines))


def write_combined_svg(path: Path, plot_paths: list[Path]) -> None:
    chunks = []
    y_offset = 0
    width = 940
    for plot_path in plot_paths:
        text = plot_path.read_text()
        view_box = text.split("viewBox=\"", 1)[1].split("\"", 1)[0].split()
        panel_width = int(float(view_box[2]))
        panel_height = int(float(view_box[3]))
        body = text.split(">", 1)[1].rsplit("</svg>", 1)[0]
        chunks.append(f'<g transform="translate(0,{y_offset})">{body}</g>')
        width = max(width, panel_width)
        y_offset += panel_height + 22
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y_offset}" '
        f'viewBox="0 0 {width} {y_offset}">\n' + "\n".join(chunks) + "\n</svg>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None, help="Dowser work directory [default: script directory]")
    parser.add_argument("--out-dir", default="ighg1_vs_igha1", help="Output directory under base-dir")
    parser.add_argument("--regime-dir", default="regimes", help="Original non-CSR regime directory")
    parser.add_argument(
        "--count-source",
        default="lavous_ighg1_vs_other",
        help="Existing directory containing readcounts/library/metadata to subset",
    )
    parser.add_argument("--clones", default=",".join(DEFAULT_CLONES), help="Comma-separated clone IDs")
    parser.add_argument("--root-regime", default="IGHG1", help="Regime to force on each root node")
    parser.add_argument("--min-gene-cells", type=int, default=5, help="Minimum expressing cells for output genes")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    out = base / args.out_dir
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    dirs = {
        "newick": out / "newick",
        "regimes": out / "regimes",
        "readcounts": out / "readcounts",
        "readcounts_no_ig": out / "readcounts_no_ig",
        "library": out / "library",
        "metadata": out / "metadata",
        "plots": out / "plots",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    clones = [clone.strip() for clone in args.clones.split(",") if clone.strip()]
    regime_dir = base / args.regime_dir
    count_source = base / args.count_source
    annot = read_annotation(count_source / "metadata" / "gene_annotation.tsv")

    pruned_roots: dict[str, Node] = {}
    pruned_regimes: dict[str, dict[str, str]] = {}
    kept_cells: dict[str, list[str]] = {}
    drop_by_clone: dict[str, Counter] = {}
    dropped_tip_details: dict[str, dict[str, str]] = {}
    original_roots: dict[str, str] = {}

    for clone in clones:
        tree_path = regime_dir / "newick" / f"{clone}.nwk"
        regime_path = regime_dir / f"{clone}.regime.csv"
        root = parse_newick(tree_path.read_text())
        regimes = read_regimes(regime_path)
        if root.name is None:
            raise RuntimeError(f"{clone} root node is unnamed")
        original_roots[clone] = regimes.get(root.name, "")
        regimes[root.name] = args.root_regime

        drop_counts: Counter = Counter()
        dropped_tips: dict[str, str] = {}
        pruned = prune_tree(root, regimes, drop_counts=drop_counts, dropped_tips=dropped_tips)
        if pruned is None or count_leaves(pruned) < 2:
            raise RuntimeError(f"{clone} has too few tips after pruning")
        pruned = collapse_unary(pruned, is_root=True)

        names = {node.name for node in iter_nodes(pruned)}
        final_regimes = {name: regimes[name] for name in names if name is not None}
        if set(final_regimes.values()) - TARGET_REGIMES:
            raise RuntimeError(f"{clone} retained unexpected regimes: {sorted(set(final_regimes.values()) - TARGET_REGIMES)}")

        pruned_roots[clone] = pruned
        pruned_regimes[clone] = final_regimes
        kept_cells[clone] = leaf_names(pruned)
        drop_by_clone[clone] = drop_counts
        dropped_tip_details[clone] = dropped_tips

        write_newick(pruned, dirs["newick"] / f"{clone}.nwk")
        write_regimes(dirs["regimes"] / f"{clone}.ighg1_vs_igha1.regime.csv", pruned, final_regimes)

    clone_rows: dict[str, dict[str, list[int]]] = {}
    clone_libraries: dict[str, dict[str, float]] = {}
    genes: list[str] | None = None
    all_library_factors = []
    for clone in clones:
        keep = set(kept_cells[clone])
        clone_genes, rows = read_count_subset(
            count_source / "readcounts" / f"{clone}.readcounts.tsv",
            keep,
        )
        if genes is None:
            genes = clone_genes
        elif clone_genes != genes:
            raise RuntimeError(f"Gene columns differ for {clone}")
        libs = read_library(count_source / "library" / f"{clone}.library.tsv", keep)
        clone_rows[clone] = rows
        clone_libraries[clone] = libs
        all_library_factors.extend(libs.values())

    if genes is None:
        raise RuntimeError("No genes loaded")
    median_factor = statistics.median(all_library_factors)
    if median_factor <= 0:
        raise RuntimeError("Median library factor is non-positive")

    full_keep, no_ig_keep = compute_gene_filters(
        clone_rows, genes, annot, min_cells=args.min_gene_cells
    )

    combined_rows: dict[str, list[int]] = {}
    combined_cells: list[str] = []
    for clone in clones:
        cells = kept_cells[clone]
        combined_cells.extend(cells)
        combined_rows.update(clone_rows[clone])
        write_count_matrix(
            dirs["readcounts"] / f"{clone}.readcounts.tsv",
            cells,
            genes,
            clone_rows[clone],
            full_keep,
        )
        write_count_matrix(
            dirs["readcounts_no_ig"] / f"{clone}.readcounts.no_ig.tsv",
            cells,
            genes,
            clone_rows[clone],
            no_ig_keep,
        )
        write_library(
            dirs["library"] / f"{clone}.library.tsv",
            cells,
            clone_libraries[clone],
            median_factor,
        )

    write_count_matrix(
        dirs["readcounts"] / "ighg1_vs_igha1.readcounts.tsv",
        combined_cells,
        genes,
        combined_rows,
        full_keep,
    )
    write_count_matrix(
        dirs["readcounts_no_ig"] / "ighg1_vs_igha1.readcounts.no_ig.tsv",
        combined_cells,
        genes,
        combined_rows,
        no_ig_keep,
    )
    write_annotation(dirs["metadata"] / "gene_annotation.tsv", genes, annot, full_keep)
    write_annotation(dirs["metadata"] / "gene_annotation.no_ig.tsv", genes, annot, no_ig_keep)

    plot_paths = []
    for clone in clones:
        plot_path = dirs["plots"] / f"{clone}.svg"
        write_svg_plot(plot_path, clone, pruned_roots[clone], pruned_regimes[clone])
        plot_paths.append(plot_path)
    write_combined_svg(out / "plots_all.svg", plot_paths)

    with (dirs["metadata"] / "preprocess_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "clone",
                "original_root_regime",
                "forced_root_regime",
                "kept_tips",
                "kept_nodes",
                "kept_ighg1_nodes",
                "kept_igha1_nodes",
                "dropped_ighg2_tips",
                "dropped_ighg1_after_igha1_tips",
                "dropped_other_tips",
            ]
        )
        for clone in clones:
            regimes = pruned_regimes[clone]
            counts = Counter(regimes.values())
            drops = drop_by_clone[clone]
            other_drop = sum(
                count for reason, count in drops.items()
                if reason not in {"IGHG2", "IGHG1_after_IGHA1"}
            )
            writer.writerow(
                [
                    clone,
                    original_roots[clone],
                    args.root_regime,
                    len(kept_cells[clone]),
                    len(regimes),
                    counts["IGHG1"],
                    counts["IGHA1"],
                    drops["IGHG2"],
                    drops["IGHG1_after_IGHA1"],
                    other_drop,
                ]
            )
        writer.writerow([])
        writer.writerow(["gene_filter", "n_genes"])
        writer.writerow([f"expressed_in_at_least_{args.min_gene_cells}_kept_tips", sum(full_keep)])
        writer.writerow(["same_filter_excluding_ig_genes", sum(no_ig_keep)])
        writer.writerow([])
        writer.writerow(["combined_kept_tips", len(combined_cells)])
        writer.writerow(["library_factor_median_before_renorm", f"{median_factor:.10g}"])

    with (dirs["metadata"] / "dropped_tips.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["clone", "tip", "reason"])
        for clone in clones:
            for tip, reason in sorted(dropped_tip_details[clone].items()):
                writer.writerow([clone, tip, reason])

    with (dirs["metadata"] / "lavous_command.txt").open("w") as handle:
        out_name = out.name
        tree_csv = ",".join(f"{out_name}/newick/{clone}.nwk" for clone in clones)
        expr_csv = ",".join(f"{out_name}/readcounts/{clone}.readcounts.tsv" for clone in clones)
        regime_csv = ",".join(
            f"{out_name}/regimes/{clone}.ighg1_vs_igha1.regime.csv"
            for clone in clones
        )
        lib_csv = ",".join(f"{out_name}/library/{clone}.library.tsv" for clone in clones)
        handle.write(
            "lavous-diff \\\n"
            f"  --tree {tree_csv} \\\n"
            f"  --expression {expr_csv} \\\n"
            f"  --regime {regime_csv} \\\n"
            f"  --library {lib_csv} \\\n"
            f"  --annot {out_name}/metadata/gene_annotation.tsv \\\n"
            "  --null IGHG1 \\\n"
            f"  --outdir {out_name}/joint_outputs/diff/full \\\n"
            "  --prefix ighg1_vs_igha1_joint.full\n"
        )

    print(f"Wrote {out}")
    for clone in clones:
        drops = drop_by_clone[clone]
        print(
            f"{clone}: kept {len(kept_cells[clone])} tips; "
            f"dropped IGHG2={drops['IGHG2']}, "
            f"IGHG1_after_IGHA1={drops['IGHG1_after_IGHA1']}; "
            f"root {original_roots[clone]}->{args.root_regime}"
        )
    print(f"Genes kept: full={sum(full_keep)}, no_ig={sum(no_ig_keep)}")


if __name__ == "__main__":
    main()
