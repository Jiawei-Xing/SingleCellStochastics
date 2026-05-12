#!/usr/bin/env python
"""Circular tree plot of VINE tree colored by primary vs metastatic."""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from Bio import Phylo

sys.setrecursionlimit(10000)

TREE_FILE = "3724_NT_All_vine_tree.nwk"
REGIME_FILE = "3724_NT_All_vine_regime.csv"  # VINE-reconstructed labels for ALL nodes
STATELABELS_FILE = "3724_NT_All_statelabels.csv"
OUTPUT = "3724_NT_All_tree_pri_met_circular.png"

# --- Read VINE regime labels (internal + leaf nodes) ---
node_regime = {}  # node_name -> "T1" or "Met"
with open(REGIME_FILE) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        node_regime[row[0].strip()] = row[1].strip()

# Also read leaf-only labels for the outer ring
leaf_label = {}
with open(STATELABELS_FILE) as f:
    for row in csv.reader(f):
        leaf_label[row[0].strip()] = row[1].strip()

def pri_met(label):
    return "Pri" if label == "T1" else "Met"

COLORS = {"Pri": "#2166ac", "Met": "#b2182b"}

# --- Read tree ---
tree = Phylo.read(TREE_FILE, "newick")

# --- Ladderize: sort children by subtree size for clean layout ---
tree.ladderize(reverse=True)

all_clades = list(tree.find_clades(order="postorder"))
n_leaves = sum(1 for c in all_clades if c.is_terminal())

# --- Color clades using VINE-reconstructed regime labels ---
clade_color = {}
for clade in all_clades:
    name = clade.name if clade.name else ""
    regime = node_regime.get(name, "T1")
    pm = pri_met(regime)
    clade_color[id(clade)] = COLORS[pm]

# --- Assign angular positions (theta) to leaves ---
# Leaves get evenly spaced angles around the circle
leaf_idx = 0
theta = {}  # id(clade) -> angle in radians
for clade in all_clades:
    if clade.is_terminal():
        theta[id(clade)] = 2 * np.pi * leaf_idx / n_leaves
        leaf_idx += 1
    else:
        child_thetas = [theta[id(c)] for c in clade.clades]
        # Use circular mean for internal nodes
        theta[id(clade)] = np.arctan2(
            np.mean([np.sin(t) for t in child_thetas]),
            np.mean([np.cos(t) for t in child_thetas])
        )

# --- Assign radial positions (r) = distance from root ---
r = {}
stack = [(tree.root, 0.0)]
while stack:
    clade, cur_r = stack.pop()
    r[id(clade)] = cur_r
    for child in reversed(clade.clades):
        bl = child.branch_length if child.branch_length else 0
        stack.append((child, cur_r + bl))

max_r = max(r.values())

# --- Draw circular tree ---
fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
ax.set_aspect("equal")

BRANCH_LW = 0.6
ARC_LW = 0.5

# Collect line segments for batch drawing (faster)
for clade in all_clades:
    if clade.is_terminal():
        continue
    cr = r[id(clade)]
    ct = theta[id(clade)]

    # Arc connecting children at parent's radius
    child_thetas = sorted([theta[id(c)] for c in clade.clades])
    if len(child_thetas) >= 2:
        t_min, t_max = child_thetas[0], child_thetas[-1]
        # Handle wrap-around: if arc spans > pi, go the short way
        if t_max - t_min > np.pi:
            t_min, t_max = t_max, t_min + 2 * np.pi
        arc_t = np.linspace(t_min, t_max, max(20, int(abs(t_max - t_min) * 50)))
        arc_x = cr * np.cos(arc_t)
        arc_y = cr * np.sin(arc_t)
        ax.plot(arc_x, arc_y, color=clade_color[id(clade)], linewidth=ARC_LW,
                solid_capstyle='round')

    # Radial branches to each child
    for child in clade.clades:
        child_r = r[id(child)]
        child_t = theta[id(child)]
        col = clade_color[id(child)]
        # Line from (cr, child_t) to (child_r, child_t) in polar
        x0, y0 = cr * np.cos(child_t), cr * np.sin(child_t)
        x1, y1 = child_r * np.cos(child_t), child_r * np.sin(child_t)
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=BRANCH_LW,
                solid_capstyle='round')

# Leaf tip dots
for leaf in tree.get_terminals():
    lr, lt = r[id(leaf)], theta[id(leaf)]
    col = clade_color[id(leaf)]
    ax.plot(lr * np.cos(lt), lr * np.sin(lt), 'o',
            color=col, markersize=1.2, markeredgewidth=0)

# Outer colored ring for tissue labels
ring_r = max_r * 1.04
ring_w = max_r * 0.025
for leaf in tree.get_terminals():
    lt = theta[id(leaf)]
    pm = pri_met(leaf_label.get(leaf.name, "T1"))
    x0, y0 = ring_r * np.cos(lt), ring_r * np.sin(lt)
    x1, y1 = (ring_r + ring_w) * np.cos(lt), (ring_r + ring_w) * np.sin(lt)
    ax.plot([x0, x1], [y0, y1], color=COLORS[pm], linewidth=1.0,
            solid_capstyle='butt')

# Legend
handles = [
    mpatches.Patch(color=COLORS["Pri"], label="Primary"),
    mpatches.Patch(color=COLORS["Met"], label="Metastatic"),
]
ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=14,
          framealpha=0.95, edgecolor="#cccccc", handlelength=1.5,
          handleheight=1.5, borderpad=0.8)

margin = max_r * 1.12
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.axis("off")
ax.set_title("KP-Tracer 3724_NT — VINE Tree\nPrimary vs Metastatic",
             fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved {OUTPUT}")
n_pri = sum(1 for v in leaf_label.values() if v == "T1")
n_met = sum(1 for v in leaf_label.values() if v != "T1")
print(f"  {n_pri} Pri, {n_met} Met leaves")
