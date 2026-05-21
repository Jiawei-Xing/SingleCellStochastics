"""Convert VINE labeled Nexus output into LAVOUS tree and regime inputs."""

import argparse
from io import StringIO
import os
import re

from Bio import Phylo
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a clean Newick tree and node_name,regime CSV from VINE output."
    )
    parser.add_argument("--nexus", required=True, help="VINE Nexus file with [&state=...] labels.")
    parser.add_argument("--tree-out", required=True, help="Output clean Newick tree with named internal nodes.")
    parser.add_argument("--regime-out", required=True, help="Output LAVOUS node_name,regime CSV.")
    parser.add_argument(
        "--merge-states",
        default="",
        help=(
            "Optional comma-separated mappings of raw states to regimes, e.g. "
            "Rgl1=Progenitor,Rgl2AL=Progenitor. Unlisted states keep their raw name."
        ),
    )
    return parser.parse_args()


def parse_merge_states(spec):
    mapping = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --merge-states item: {item}")
        state, regime = item.split("=", 1)
        mapping[state.strip()] = regime.strip()
    return mapping


def first_labeled_tree(nexus_text, path):
    tree_match = re.search(
        r"TREE\s+\S+\s*=\s*(?:\[&R\]\s*)?(.*?);",
        nexus_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not tree_match:
        raise ValueError(f"Could not find a tree in {path}")
    return tree_match.group(1) + ";"


def main():
    args = parse_args()
    state_to_regime = parse_merge_states(args.merge_states)

    with open(args.nexus, "r") as handle:
        tree_str = first_labeled_tree(handle.read(), args.nexus)

    clean_nwk = re.sub(r"\[&state=[^\]]+\]", "", tree_str)
    tree = Phylo.read(StringIO(clean_nwk), "newick")
    node_idx = 0
    for node in tree.find_clades(order="preorder"):
        if node.clades and not node.name:
            node.name = f"internal_node{node_idx}"
            node_idx += 1

    node_states = {}
    for match in re.finditer(r"([A-Za-z0-9_.-]+)\[&state=([^\]]+)\]", tree_str):
        name, state = match.group(1), match.group(2)
        node_states[name] = state

    internal_states = []
    i = 0
    while i < len(tree_str):
        if tree_str[i] == ")":
            i += 1
            match = re.match(r"\[&state=([^\]]+)\]", tree_str[i:])
            if match:
                internal_states.append(match.group(1))
                i += match.end()
            else:
                internal_states.append(None)
        else:
            i += 1

    internal_nodes = [node for node in tree.find_clades(order="postorder") if node.clades]
    if len(internal_nodes) != len(internal_states):
        raise ValueError(
            f"Internal node count mismatch: {len(internal_nodes)} vs {len(internal_states)}"
        )
    for node, state in zip(internal_nodes, internal_states):
        if state:
            node_states[node.name] = state

    regime_rows = []
    all_nodes = list(tree.find_clades(order="preorder"))
    for node in all_nodes:
        state = node_states.get(node.name)
        if state is None:
            continue
        regime_rows.append(
            {
                "node_name": node.name,
                "regime": state_to_regime.get(state, state),
            }
        )

    if len(regime_rows) != len(all_nodes):
        raise ValueError(
            f"States assigned to {len(regime_rows)} nodes, but tree has {len(all_nodes)} nodes"
        )

    os.makedirs(os.path.dirname(args.tree_out) or ".", exist_ok=True)
    Phylo.write(tree, args.tree_out, "newick")
    print(f"Saved clean tree -> {args.tree_out}")

    regime_df = pd.DataFrame(regime_rows)
    os.makedirs(os.path.dirname(args.regime_out) or ".", exist_ok=True)
    regime_df.to_csv(args.regime_out, index=False)
    print(f"Saved regime: {len(regime_df)} nodes -> {args.regime_out}")
    print(regime_df["regime"].value_counts().to_string())


if __name__ == "__main__":
    main()
