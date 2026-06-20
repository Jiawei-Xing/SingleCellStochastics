"""Gaussian belief propagation for latent expression history reconstruction."""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from Bio import Phylo

# ==========================================
# Core Tree & Math Logic
# ==========================================

class TreeNode:
    def __init__(self, name, dist=0.1):
        self.name = name
        self.dist = dist
        self.children = []
        self.regime = 'default'

        # MFVI Leaf Inputs
        self.is_leaf = False
        self.q_mu = None
        self.q_var = None
        self.read_count = None
        self.library_size = 1.0

        # Internal Storage
        self.up_lam = 0.0
        self.up_eta = 0.0
        self.msg_to_p_lam = 0.0
        self.msg_to_p_eta = 0.0

        # Final Output
        self.true_mu = 0.0
        self.true_var = 0.0

        # Plotting coordinates
        self.r = 0.0
        self.theta = 0.0

def get_ou_params(alpha, sigma_sq, theta, t):
    F = np.exp(-alpha * t)
    c = theta * (1 - F)
    if alpha > 1e-8 and t > 0:
        Q = (sigma_sq / (2 * alpha)) * (1 - np.exp(-2 * alpha * t))
    else:
        Q = sigma_sq * t
    return F, c, Q

def get_bm_params(sigma_sq, t):
    """BM transition: Mean stays same (F=1, c=0), variance grows by sig2*t."""
    return 1.0, 0.0, sigma_sq * t

def upward_pass(node, transition_fn):
    for child in node.children:
        upward_pass(child, transition_fn)
        node.up_lam += child.msg_to_p_lam
        node.up_eta += child.msg_to_p_eta

    if node.is_leaf:
        node.up_lam += 1.0 / node.q_var
        node.up_eta += node.q_mu / node.q_var

    F, c, Q = transition_fn(node)
    denom = Q * node.up_lam + 1.0
    node.msg_to_p_lam = (F**2 * node.up_lam) / denom
    node.msg_to_p_eta = F * (node.up_eta - node.up_lam * c) / denom

def downward_pass(node, transition_fn, root_prior=None):
    if root_prior is not None:
        prior_mu, prior_var = root_prior
        prior_lam = 1.0 / prior_var
        prior_eta = prior_mu / prior_var

        node.true_lam = node.up_lam + prior_lam
        node.true_eta = node.up_eta + prior_eta

    node.true_var = 1.0 / node.true_lam
    node.true_mu = node.true_eta / node.true_lam

    for child in node.children:
        cav_lam = node.true_lam - child.msg_to_p_lam
        cav_eta = node.true_eta - child.msg_to_p_eta

        cav_var = 1.0 / cav_lam if cav_lam > 1e-12 else 1e8
        cav_mu = cav_eta / cav_lam if cav_lam > 1e-12 else 0.0

        F, c, Q = transition_fn(child)

        down_mu = F * cav_mu + c
        down_var = F**2 * cav_var + Q

        down_lam = 1.0 / down_var
        down_eta = down_mu / down_var

        child.true_lam = child.up_lam + down_lam
        child.true_eta = child.up_eta + down_eta

        downward_pass(child, transition_fn)

# ==========================================
# Parsers
# ==========================================

def load_tree_from_newick(newick_path, normalize=True):
    phylo_tree = Phylo.read(newick_path, "newick")
    internal_counter = [0]
    total_length = max(phylo_tree.depths().values()) if normalize else 1.0
    if total_length == 0:
        total_length = 1.0

    def convert_clade(clade):
        name = clade.name if clade.name else f"Internal_{internal_counter[0]}"
        if not clade.name:
            internal_counter[0] += 1

        dist = clade.branch_length if clade.branch_length is not None else 0.0
        dist = dist / total_length
        node = TreeNode(name=name, dist=dist)

        if clade.clades:
            for child in clade.clades:
                node.children.append(convert_clade(child))
        else:
            node.is_leaf = True
        return node

    return convert_clade(phylo_tree.root)


def _path_to_node(root, target_name):
    if root.name == target_name:
        return [root]
    for child in root.children:
        path = _path_to_node(child, target_name)
        if path:
            return [root] + path
    return None


def _mrca_by_leaf_names(root, leaf1, leaf2):
    path1 = _path_to_node(root, leaf1)
    path2 = _path_to_node(root, leaf2)
    if path1 is None:
        raise ValueError(f"Leaf or node {leaf1!r} from regime file is not in the tree.")
    if path2 is None:
        raise ValueError(f"Leaf or node {leaf2!r} from regime file is not in the tree.")

    mrca = root
    for node1, node2 in zip(path1, path2):
        if node1 is node2:
            mrca = node1
        else:
            break
    return mrca


def apply_regimes(root, regime_tsv_path):
    df_regimes = pd.read_csv(regime_tsv_path, sep=None, engine="python")
    df_regimes.columns = [str(col).strip() for col in df_regimes.columns]
    all_nodes = {n.name: n for n in get_all_nodes(root)}

    if {"node_name", "regime"}.issubset(df_regimes.columns):
        for _, row in df_regimes.iterrows():
            node_name = str(row["node_name"])
            if node_name in all_nodes:
                all_nodes[node_name].regime = str(row["regime"])
    elif {"node", "node2", "regime"}.issubset(df_regimes.columns):
        for _, row in df_regimes.iterrows():
            node1 = str(row["node"])
            node2 = "" if pd.isna(row["node2"]) else str(row["node2"])
            node = all_nodes[node1] if node2 == "" else _mrca_by_leaf_names(root, node1, node2)
            node.regime = str(row["regime"])
    else:
        raise ValueError(
            "Regime file must contain either columns 'node_name,regime' "
            "or 'node,node2,regime'."
        )


def propagate_regimes(node):
    """Propagate the regime from each node down to descendants left as 'default'.

    Regime files often label only leaves or named clade roots. The transition
    function used during reconstruction needs a regime for every node, so any
    node still flagged as ``'default'`` inherits the closest ancestor regime.
    """
    for child in node.children:
        if child.regime == "default":
            child.regime = node.regime
        propagate_regimes(child)


def _read_counts_for_gene(counts_tsv_path, gene, cells):
    if counts_tsv_path is None:
        return {cell: None for cell in cells}
    counts = pd.read_csv(counts_tsv_path, sep="\t", index_col=0)
    if gene not in counts.columns:
        raise ValueError(f"Gene {gene!r} not found in count matrix {counts_tsv_path}.")
    return counts.loc[cells, gene].to_dict()


def _library_sizes(library_tsv_path, cells):
    if library_tsv_path is None:
        return {cell: 1.0 for cell in cells}
    lib_df = pd.read_csv(library_tsv_path, sep="\t", index_col=0, header=None)
    lib_df.index = lib_df.index.astype(str)
    return {cell: float(lib_df.loc[cell].iloc[0]) if cell in lib_df.index else 1.0
            for cell in cells}


def apply_expression_data(root, expr_tsv_path, gene, counts_tsv_path=None, library_tsv_path=None):
    if gene is None:
        raise ValueError("--gene is required to select a row from the wide q-parameter table.")

    df_expr = pd.read_csv(expr_tsv_path, sep='\t', index_col=0)
    if gene not in df_expr.index:
        raise ValueError(f"Gene {gene!r} not found in q-parameter table.")

    mean_cols = [col for col in df_expr.columns if str(col).startswith("q_mean_")]
    if not mean_cols:
        raise ValueError("Expression TSV must have q_mean_<cell>/q_std_<cell> columns from the diff test.")
    cells = [str(col)[len("q_mean_"):] for col in mean_cols]
    std_cols = [f"q_std_{cell}" for cell in cells]
    missing = [col for col in std_cols if col not in df_expr.columns]
    if missing:
        raise ValueError(f"Missing q_std columns in q-parameter table: {missing[:5]}")

    row = df_expr.loc[gene]
    q_means = row[mean_cols].to_numpy(dtype=float)
    q_stds = row[std_cols].to_numpy(dtype=float)
    read_counts = _read_counts_for_gene(counts_tsv_path, gene, cells)
    libraries = _library_sizes(library_tsv_path, cells)

    leaves = {n.name: n for n in get_leaves(root)}
    for cell, mean, std in zip(cells, q_means, q_stds):
        if cell in leaves:
            leaf = leaves[cell]
            leaf.q_mu = mean
            leaf.q_var = std**2
            rc = read_counts[cell]
            leaf.read_count = rc if pd.notna(rc) else None
            leaf.library_size = libraries[cell]

    missing_leaves = [name for name, leaf in leaves.items() if leaf.q_mu is None]
    if missing_leaves:
        raise ValueError(
            f"{len(missing_leaves)} tree leaves have no q-params row "
            f"(first 5: {missing_leaves[:5]}). The q-parameter table must cover every leaf."
        )


def load_ou_params(ou_tsv_path, gene=None, hypothesis="h1"):
    df_ou = pd.read_csv(ou_tsv_path, sep=None, engine='python')
    ou_params = {}

    if {"regime", "alpha", "sigma", "theta"}.issubset(df_ou.columns):
        if "gene" in df_ou.columns and gene is not None:
            df_ou = df_ou[df_ou["gene"].astype(str) == str(gene)]
        if "hypothesis" in df_ou.columns:
            df_ou = df_ou[df_ou["hypothesis"].astype(str) == hypothesis]
        if df_ou.empty:
            raise ValueError(
                f"No OU parameters found in {ou_tsv_path} for "
                f"gene={gene!r}, hypothesis={hypothesis!r}."
            )
        for _, row in df_ou.iterrows():
            ou_params[str(row["regime"])] = (row['alpha'], row['sigma']**2, row['theta'])
    else:
        df_ou = pd.read_csv(ou_tsv_path, sep=None, engine='python', index_col=0)
        for regime, row in df_ou.iterrows():
            ou_params[str(regime)] = (row['alpha'], row['sigma']**2, row['theta'])

    return ou_params

def load_bm_params(tsv_path, gene=None):
    """
    Reads BM parameters from the selection-test output (columns h0_mu, h0_sigma
    in a per-gene table) or from a one-row TSV with bare mu/sigma columns.
    Returns (sigma_sq, root_prior) where root_prior = (mu, sigma_sq).
    """
    df = pd.read_csv(tsv_path, sep=None, engine='python')

    if {"h0_mu", "h0_sigma"}.issubset(df.columns):
        if "gene" in df.columns:
            if gene is None:
                raise ValueError("--gene is required when reading BM params from selection-test output.")
            row = df[df["gene"].astype(str) == str(gene)]
            if row.empty:
                raise ValueError(f"Gene {gene!r} not found in {tsv_path}.")
            mu_val = float(row["h0_mu"].iloc[0])
            sigma_val = float(row["h0_sigma"].iloc[0])
        else:
            mu_val = float(df["h0_mu"].iloc[0])
            sigma_val = float(df["h0_sigma"].iloc[0])
    else:
        mu_val = float(df["mu"].iloc[0])
        sigma_val = float(df["sigma"].iloc[0])

    sigma_sq = sigma_val ** 2
    root_prior = (mu_val, sigma_sq)
    return sigma_sq, root_prior

# ==========================================
# Data Export & Visualization
# ==========================================

def get_leaves(node):
    if node.is_leaf: return [node]
    return sum([get_leaves(c) for c in node.children], [])

def get_all_nodes(node):
    nodes = [node]
    for c in node.children:
        nodes.extend(get_all_nodes(c))
    return nodes

def save_inferred_history(root, output_tsv_path, include_regime=True):
    """Saves the reconstructed means and variances to a TSV file."""
    all_nodes = get_all_nodes(root)
    data = []

    for node in all_nodes:
        entry = {
            'node_name': node.name,
            'is_leaf': node.is_leaf,
        }
        if include_regime:
            entry['regime'] = node.regime
        entry['infer_mu'] = node.true_mu
        entry['infer_var'] = node.true_var
        data.append(entry)

    df = pd.DataFrame(data)
    df.to_csv(output_tsv_path, sep='\t', index=False)
    print(f"Successfully saved inferred history table to {output_tsv_path}")

def count_leaves(node):
    """Count number of leaves in subtree."""
    if node.is_leaf:
        return 1
    return sum(count_leaves(c) for c in node.children)

def ladderize(node, reverse=True):
    """Sort children by subtree size for clean tree layout."""
    if node.is_leaf:
        return
    for child in node.children:
        ladderize(child, reverse)
    node.children.sort(key=count_leaves, reverse=reverse)

def layout_circular_tree(node, r_current=0.0, leaf_angles=None):
    node.r = r_current + node.dist
    if node.is_leaf:
        node.theta = leaf_angles.pop(0)
    else:
        for child in node.children:
            layout_circular_tree(child, node.r, leaf_angles)
        node.theta = np.mean([c.theta for c in node.children])

def plot_circular_tree(root, outer, output_path, tree_linewidth=3.0, bar_linewidth=None):
    # Ladderize for clean layout (same as plot_tree_circular.py)
    ladderize(root, reverse=True)

    leaves = get_leaves(root)
    all_nodes = get_all_nodes(root)

    angles = list(np.linspace(0, 2 * np.pi, len(leaves), endpoint=False))
    layout_circular_tree(root, r_current=0.0, leaf_angles=angles)
    max_r = max([n.r for n in all_nodes])

    vars_array = [n.true_var for n in all_nodes]
    mus_array = [n.true_mu for n in all_nodes]

    v_min, v_max = min(vars_array), max(vars_array)
    mu_min, mu_max = min(mus_array), max(mus_array)

    leaf_rc_norm = {
        leaf.name: leaf.read_count / leaf.library_size
        for leaf in leaves
        if leaf.read_count is not None and leaf.library_size > 0
    }
    valid_rcs = list(leaf_rc_norm.values())
    rc_max = max(valid_rcs) if valid_rcs else 1.0

    norm_var = mcolors.Normalize(vmin=v_min - (v_max - v_min)*0.15, vmax=v_max)

    # Diverging colormap centered on root: blue=low, white=root, red=high.
    # Tree color scale uses only mu range; read-count bars have their own Oranges cmap below.
    root_mu = root.true_mu
    eps = max(abs(root_mu) * 1e-3, 1e-6)
    mu_radius = max(abs(mu_min - root_mu), abs(mu_max - root_mu), eps)
    vmin_mu = root_mu - mu_radius
    vmax_mu = root_mu + mu_radius
    cmap_mu = plt.cm.RdBu_r
    norm_mu = mcolors.TwoSlopeNorm(vcenter=root_mu, vmin=vmin_mu, vmax=vmax_mu)

    cmap_var = plt.cm.Blues

    # Cartesian circular layout (same rendering as plot_tree_circular.py)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")

    if bar_linewidth is None:
        bar_linewidth = tree_linewidth
    colorbar_label_fontsize = 14

    def draw_tree(ax, node, val_func, cmap, norm):
        if not node.children:
            return
        cr = node.r
        col_parent = cmap(norm(val_func(node)))

        # Arc connecting children at parent's radius
        child_thetas = sorted([c.theta for c in node.children])
        if len(child_thetas) >= 2:
            t_min, t_max = child_thetas[0], child_thetas[-1]
            if t_max - t_min > np.pi:
                t_min, t_max = t_max, t_min + 2 * np.pi
            arc_t = np.linspace(t_min, t_max, max(20, int(abs(t_max - t_min) * 50)))
            ax.plot(cr * np.cos(arc_t), cr * np.sin(arc_t),
                    color=col_parent, linewidth=tree_linewidth, solid_capstyle='round')

        # Radial branches with gradient
        for child in node.children:
            col_child = cmap(norm(val_func(child)))
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
                ax.plot([x0, x1], [y0, y1], color=col, linewidth=tree_linewidth,
                        solid_capstyle='butt')

            draw_tree(ax, child, val_func, cmap, norm)

    # 1. Plot Variance
    ax1.set_title("Variance of Prediction", fontsize=13, pad=12)
    ax1.axis('off')
    draw_tree(ax1, root, lambda n: n.true_var, cmap_var, norm_var)

    # 2. Plot Expression (diverging: blue=low, white=root, red=high)
    ax2.set_title("Predicted Expression", fontsize=13, pad=12)
    ax2.axis('off')
    draw_tree(ax2, root, lambda n: n.true_mu, cmap_mu, norm_mu)

    # Read count outer bars (expression panel)
    # Use separate sequential colormap (raw counts are always >= 0)
    if outer and valid_rcs:
        cmap_rc = plt.cm.Oranges
        norm_rc = mcolors.Normalize(vmin=0, vmax=rc_max)
        bar_base = max_r * 1.04
        for leaf in leaves:
            rc_norm = leaf_rc_norm.get(leaf.name)
            if rc_norm is not None and rc_norm > 0:
                bar_len = (rc_norm / rc_max) * (max_r * 0.15)
                x0 = bar_base * np.cos(leaf.theta)
                y0 = bar_base * np.sin(leaf.theta)
                x1 = (bar_base + bar_len) * np.cos(leaf.theta)
                y1 = (bar_base + bar_len) * np.sin(leaf.theta)
                ax2.plot([x0, x1], [y0, y1],
                         color=cmap_rc(norm_rc(rc_norm)), linewidth=bar_linewidth,
                         solid_capstyle='butt')

    margin = max_r * 1.25 if outer else max_r * 1.10
    for ax in (ax1, ax2):
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    sm_var = plt.cm.ScalarMappable(norm=norm_var, cmap=cmap_var)
    cbar1 = fig.colorbar(sm_var, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.08, shrink=0.7)
    cbar1.ax.set_title("Variance", loc='left', fontsize=colorbar_label_fontsize)

    sm_mu = plt.cm.ScalarMappable(norm=norm_mu, cmap=cmap_mu)
    cbar2 = fig.colorbar(sm_mu, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.08, shrink=0.7)
    cbar2.ax.set_title("Latent", loc='left', fontsize=colorbar_label_fontsize)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved figure to {output_path}")

# ==========================================
# CLI Execution
# ==========================================

def run_reconst():
    parser = argparse.ArgumentParser(description="1D Ancestral State Reconstruction via Belief Propagation")
    parser.add_argument("--tree", required=True, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--q_params", required=True, help="Path to wide q-parameter table from diff test (TSV with q_mean_<cell>/q_std_<cell> columns)")
    parser.add_argument("--gene", required=True, help="Gene row to reconstruct from --q_params")
    parser.add_argument("--read_counts", required=False, help="Raw count matrix used to add read_count values")
    parser.add_argument("--library", required=False, help="Library size TSV (no header row; column 0 = cell name, column 1 = library size) used to normalize read_count bars in the plot")
    parser.add_argument("--model", required=False, type=str, choices=["ou", "bm"], default="ou", help="Model type: 'ou' (Ornstein-Uhlenbeck, default) or 'bm' (Brownian Motion)")
    parser.add_argument("--regime", required=False, help="Path to node regime mapping (TSV, ou model only)")
    parser.add_argument("--ou", required=False, help="Path to OU parameters table (TSV, ou model only)")
    parser.add_argument("--bm", required=False, help="Path to BM parameters table (TSV, bm model only) — accepts selection-test output (per-gene rows with h0_mu/h0_sigma) or a one-row TSV with mu/sigma")
    parser.add_argument("--hypothesis", required=False, choices=["h0", "h1"], default="h1", help="Hypothesis to use from long-form OU parameter tables")
    parser.add_argument("--no_normalize_tree", action="store_true", help="Use raw Newick branch lengths instead of the fitted model's normalized tree scale")
    parser.add_argument("--no_outer", action='store_false', help="Whether to plot read counts as bars outside the tree")
    parser.add_argument("--tree_linewidth", required=False, type=float, default=3.0, help="Line width for reconstructed tree branches")
    parser.add_argument("--bar_linewidth", required=False, type=float, help="Line width for outer read-count bars; defaults to --tree_linewidth")
    parser.add_argument("--out_fig", required=False, default="history.png", help="Path to save the output figure (e.g., plot.png)")
    parser.add_argument("--out_tsv", required=False, default="history.tsv", help="Path to save the inferred true means and vars (TSV)")

    args = parser.parse_args()

    plt.close('all')

    print("Loading data...")
    root = load_tree_from_newick(args.tree, normalize=not args.no_normalize_tree)
    apply_expression_data(root, args.q_params, gene=args.gene, counts_tsv_path=args.read_counts, library_tsv_path=args.library)
    root.dist = 0.0

    if args.model == "ou":
        if not args.regime or not args.ou:
            parser.error("--regime and --ou are required for the OU model")
        apply_regimes(root, args.regime)
        propagate_regimes(root)
        ou_params = load_ou_params(args.ou, gene=args.gene, hypothesis=args.hypothesis)
        root_params = ou_params.get(root.regime, ou_params.get("shared")) # shared: h0
        if root_params is None:
            parser.error(f"No OU parameters found for root regime {root.regime!r}")
        r_alpha, r_sig2, r_theta = root_params
        alpha_floor = 1e-6
        if r_alpha < alpha_floor:
            print(f"Warning: root regime alpha={r_alpha:.2e} is below {alpha_floor:.0e}; "
                  f"flooring to avoid blow-up of stationary prior variance.")
            r_alpha = alpha_floor
        root_prior = (r_theta, r_sig2 / (2 * r_alpha))

        def transition_fn(node):
            p = ou_params.get(node.regime, ou_params.get("shared"))
            if p is None:
                raise KeyError(f"No OU parameters for regime {node.regime!r}.")
            alpha, sig2, theta = p
            return get_ou_params(alpha, sig2, theta, node.dist)
    else:  # bm
        if not args.bm:
            parser.error("--bm is required for the BM model")
        sigma_sq, root_prior = load_bm_params(args.bm, gene=args.gene)

        def transition_fn(node):
            return get_bm_params(sigma_sq, node.dist)

    print("Running upward pass (leaves -> root)...")
    upward_pass(root, transition_fn)

    print("Running downward pass (root -> leaves)...")
    downward_pass(root, transition_fn, root_prior=root_prior)

    print("Exporting data...")
    save_inferred_history(root, args.out_tsv, include_regime=(args.model == "ou"))

    print("Generating plot...")
    plot_circular_tree(
        root,
        args.no_outer,
        args.out_fig,
        tree_linewidth=args.tree_linewidth,
        bar_linewidth=args.bar_linewidth,
    )


if __name__ == "__main__":
    run_reconst()
