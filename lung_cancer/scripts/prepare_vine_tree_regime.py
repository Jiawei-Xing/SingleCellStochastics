"""Convert VINE labeled Nexus output into LAVOUS tree and regime inputs."""

import argparse
import os
import re

import pandas as pd
from ete3 import Tree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a clean Newick tree and merged regime CSV from VINE output."
    )
    parser.add_argument(
        "--nexus",
        default="data/vine/3724_NT_All_vine_states.nex",
        help="VINE Nexus file with [&state=...] labels.",
    )
    parser.add_argument(
        "--tree-out",
        default="data/3724_NT_All_vine_tree.nwk",
        help="Output clean Newick tree with named internal nodes.",
    )
    parser.add_argument(
        "--regime-out",
        default="data/3724_NT_All_vine_regime.csv",
        help="Output LAVOUS node_name,regime CSV.",
    )
    parser.add_argument("--primary", default="T1", help="Primary/null regime label.")
    parser.add_argument(
        "--met-states",
        default="S1,L1,L2,L3",
        help="Comma-separated VINE states to merge as Met.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.nexus, "r") as handle:
        nexus_text = handle.read()

    tree_match = re.search(
        r"TREE\s+\S+\s*=\s*(?:\[&R\]\s*)?(.*?);",
        nexus_text,
        re.IGNORECASE,
    )
    if not tree_match:
        raise ValueError(f"Could not find a tree in {args.nexus}")

    tree_str = tree_match.group(1) + ";"
    clean_nwk = re.sub(r"\[&state=\w+\]", "", tree_str)

    tree = Tree(clean_nwk, format=1)
    node_idx = 0
    for node in tree.traverse("preorder"):
        if not node.is_leaf() and not node.name:
            node.name = f"internal_node{node_idx}"
            node_idx += 1

    node_states = {}
    for match in re.finditer(r"([\w][\w.\-]*)\[&state=(\w+)\]", tree_str):
        name, state = match.group(1), match.group(2)
        node_states[name] = state

    internal_states = []
    i = 0
    while i < len(tree_str):
        if tree_str[i] == "(":
            i += 1
        elif tree_str[i] == ")":
            i += 1
            match = re.match(r"\[&state=(\w+)\]", tree_str[i:])
            if match:
                internal_states.append(match.group(1))
                i += match.end()
            else:
                internal_states.append(None)
        else:
            i += 1

    internal_nodes = [node for node in tree.traverse("postorder") if not node.is_leaf()]
    if len(internal_nodes) != len(internal_states):
        raise ValueError(
            f"Internal node count mismatch: {len(internal_nodes)} vs {len(internal_states)}"
        )
    for node, state in zip(internal_nodes, internal_states):
        if state:
            node_states[node.name] = state

    n_leaf = sum(1 for node in tree.get_leaves() if node.name in node_states)
    n_internal = sum(
        1 for node in tree.traverse() if not node.is_leaf() and node.name in node_states
    )
    print(f"States assigned: {len(node_states)} nodes ({n_leaf} leaves, {n_internal} internal)")

    met_states = {state.strip() for state in args.met_states.split(",") if state.strip()}
    regime_rows = []
    for node in tree.traverse():
        state = node_states.get(node.name, "")
        if state:
            regime = "Met" if state in met_states else args.primary
            regime_rows.append({"node_name": node.name, "regime": regime})

    regime_df = pd.DataFrame(regime_rows)
    os.makedirs(os.path.dirname(args.regime_out) or ".", exist_ok=True)
    regime_df.to_csv(args.regime_out, index=False)
    print(f"Saved regime: {len(regime_df)} nodes -> {args.regime_out}")
    print(regime_df["regime"].value_counts().to_string())

    os.makedirs(os.path.dirname(args.tree_out) or ".", exist_ok=True)
    tree.write(outfile=args.tree_out, format=1, format_root_node=True)
    print(f"Saved clean tree with named internal nodes -> {args.tree_out}")


if __name__ == "__main__":
    main()
