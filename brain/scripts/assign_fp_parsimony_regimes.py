#!/usr/bin/env python
"""Assign FP-lineage regimes to internal tree nodes by directed parsimony."""

import argparse
from collections import deque
import csv
import math
import os

import pandas as pd
from Bio import Phylo


STATES = ["FP_Rgl1", "FP_NbFP", "FP_NbDA", "FP_DA", "FP_NbGLU", "FP_GLUFP"]
FORWARD_EDGES = {
    "FP_Rgl1": ["FP_NbFP"],
    "FP_NbFP": ["FP_NbDA", "FP_NbGLU"],
    "FP_NbDA": ["FP_DA"],
    "FP_NbGLU": ["FP_GLUFP"],
    "FP_DA": [],
    "FP_GLUFP": [],
}
INF = 10**9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer internal FP regimes using directed parsimony."
    )
    parser.add_argument("--tree", required=True, help="Input Newick tree.")
    parser.add_argument(
        "--labels",
        required=True,
        help="Leaf statelabel CSV with rows: leaf_id,FP_regime.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for regime, transition, and labeled tree files.",
    )
    parser.add_argument(
        "--root-state",
        default="FP_Rgl1",
        choices=STATES,
        help="State forced at the root.",
    )
    parser.add_argument(
        "--backward-penalty",
        type=float,
        default=10.0,
        help="Extra cost for transitions opposite the developmental graph.",
    )
    parser.add_argument(
        "--cross-penalty",
        type=float,
        default=12.0,
        help="Extra cost for transitions across DA/GLU branches.",
    )
    return parser.parse_args()


def shortest_paths(graph):
    dist = {state: {state: 0} for state in STATES}
    for start in STATES:
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            node, depth = queue.popleft()
            for child in graph[node]:
                if child in seen:
                    continue
                seen.add(child)
                dist[start][child] = depth + 1
                queue.append((child, depth + 1))
    return dist


def reverse_graph(graph):
    rev = {state: [] for state in STATES}
    for parent, children in graph.items():
        for child in children:
            rev[child].append(parent)
    return rev


FORWARD_DIST = shortest_paths(FORWARD_EDGES)
BACKWARD_DIST = shortest_paths(reverse_graph(FORWARD_EDGES))
STATE_INDEX = {state: i for i, state in enumerate(STATES)}


def transition_direction(parent_state, child_state):
    if parent_state == child_state:
        return "same", 0
    if child_state in FORWARD_DIST[parent_state]:
        return "forward", FORWARD_DIST[parent_state][child_state]
    if child_state in BACKWARD_DIST[parent_state]:
        return "backward", BACKWARD_DIST[parent_state][child_state]
    return "cross_branch", math.nan


def transition_cost(parent_state, child_state, backward_penalty, cross_penalty):
    direction, steps = transition_direction(parent_state, child_state)
    if direction == "same":
        return 0.0
    if direction == "forward":
        return float(steps)
    if direction == "backward":
        return backward_penalty + float(steps)
    return cross_penalty


def read_leaf_labels(path):
    labels = {}
    with open(path) as handle:
        for leaf, state in csv.reader(handle):
            if state not in STATES:
                raise ValueError(f"Unexpected state {state!r} for leaf {leaf!r}")
            labels[leaf] = state
    return labels


def ensure_internal_names(tree):
    idx = 0
    used = {clade.name for clade in tree.find_clades() if clade.name}
    for clade in tree.find_clades(order="preorder"):
        if clade.is_terminal():
            continue
        if clade.name:
            continue
        while f"internal_node{idx}" in used:
            idx += 1
        clade.name = f"internal_node{idx}"
        used.add(clade.name)
        idx += 1


def pick_best(options):
    return min(options, key=lambda item: item[0])


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    tree = Phylo.read(args.tree, "newick")
    ensure_internal_names(tree)
    leaf_labels = read_leaf_labels(args.labels)
    missing = sorted({leaf.name for leaf in tree.get_terminals()} - set(leaf_labels))
    if missing:
        raise ValueError(f"{len(missing)} tree leaves missing labels; first: {missing[:5]}")

    score = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            observed = leaf_labels[clade.name]
            score[id(clade)] = {
                state: (0.0 if state == observed else INF) for state in STATES
            }
            continue

        node_scores = {}
        for parent_state in STATES:
            total = 0.0
            for child in clade.clades:
                child_options = []
                for child_state in STATES:
                    cost = (
                        score[id(child)][child_state]
                        + transition_cost(
                            parent_state,
                            child_state,
                            args.backward_penalty,
                            args.cross_penalty,
                        )
                    )
                    direction, steps = transition_direction(parent_state, child_state)
                    direction_rank = {
                        "same": 0,
                        "forward": 1,
                        "backward": 2,
                        "cross_branch": 3,
                    }[direction]
                    child_options.append(
                        (cost, direction_rank, STATE_INDEX[child_state], child_state)
                    )
                total += pick_best(child_options)[0]
            node_scores[parent_state] = total
        score[id(clade)] = node_scores

    assignment = {id(tree.root): args.root_state}
    for parent in tree.find_clades(order="preorder"):
        parent_state = assignment[id(parent)]
        for child in parent.clades:
            options = []
            for child_state in STATES:
                cost = (
                    score[id(child)][child_state]
                    + transition_cost(
                        parent_state,
                        child_state,
                        args.backward_penalty,
                        args.cross_penalty,
                    )
                )
                direction, _steps = transition_direction(parent_state, child_state)
                direction_rank = {
                    "same": 0,
                    "forward": 1,
                    "backward": 2,
                    "cross_branch": 3,
                }[direction]
                options.append((cost, direction_rank, STATE_INDEX[child_state], child_state))
            assignment[id(child)] = pick_best(options)[3]

    regime_rows = []
    transition_rows = []
    for clade in tree.find_clades(order="preorder"):
        state = assignment[id(clade)]
        regime_rows.append(
            {
                "node_name": clade.name,
                "regime": state,
                "is_leaf": clade.is_terminal(),
                "parsimony_score_at_node": score[id(clade)][state],
            }
        )
        for child in clade.clades:
            child_state = assignment[id(child)]
            direction, steps = transition_direction(state, child_state)
            transition_rows.append(
                {
                    "parent": clade.name,
                    "child": child.name,
                    "parent_regime": state,
                    "child_regime": child_state,
                    "transition": f"{state}->{child_state}",
                    "direction": direction,
                    "steps": steps,
                    "branch_length": child.branch_length or 0.0,
                    "child_is_leaf": child.is_terminal(),
                }
            )

    regime_path = f"{args.out_prefix}_parsimony_regime.csv"
    transition_path = f"{args.out_prefix}_parsimony_transitions.tsv"
    summary_path = f"{args.out_prefix}_parsimony_transition_summary.tsv"
    tree_path = f"{args.out_prefix}_parsimony_named_tree.nwk"

    pd.DataFrame(regime_rows).to_csv(regime_path, index=False)
    transition_df = pd.DataFrame(transition_rows)
    transition_df.to_csv(transition_path, sep="\t", index=False)
    (
        transition_df.groupby(["transition", "direction"], dropna=False)
        .size()
        .reset_index(name="edges")
        .sort_values(["direction", "edges", "transition"], ascending=[True, False, True])
        .to_csv(summary_path, sep="\t", index=False)
    )
    Phylo.write(tree, tree_path, "newick")

    total_score = score[id(tree.root)][args.root_state]
    print(f"Root state: {args.root_state}")
    print(f"Parsimony score: {total_score:g}")
    print(f"Saved {regime_path}")
    print(f"Saved {transition_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {tree_path}")
    print(pd.Series([row['regime'] for row in regime_rows]).value_counts().sort_index().to_string())
    changing = transition_df[transition_df["direction"] != "same"]
    print("Changing edges:")
    print(changing["transition"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
