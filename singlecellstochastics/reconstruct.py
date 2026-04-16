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

def _make_transition_fn(model, params):
    """Create a transition function (node -> F, c, Q) based on model type."""
    if model == "ou":
        ou_params = params
        def transition_fn(node):
            alpha, sig2, theta = ou_params[node.regime]
            return get_ou_params(alpha, sig2, theta, node.dist)
        return transition_fn
    else:  # bm
        sigma_sq = params
        def transition_fn(node):
            return get_bm_params(sigma_sq, node.dist)
        return transition_fn

def upward_pass(node, transition_fn):
    for child in node.children:
        upward_pass(child, transition_fn)
        node.up_lam += child.msg_to_p_lam
        node.up_eta += child.msg_to_p_eta

    if node.is_leaf:
        node.up_lam += 1.0 / node.q_var
        node.up_eta += node.q_mu / node.q_var

    if node.dist > 0:
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

def load_tree_from_newick(newick_path):
    phylo_tree = Phylo.read(newick_path, "newick")
    internal_counter = [0]

    def convert_clade(clade):
        name = clade.name if clade.name else f"Internal_{internal_counter[0]}"
        if not clade.name:
            internal_counter[0] += 1

        dist = clade.branch_length if clade.branch_length is not None else 0.0
        node = TreeNode(name=name, dist=dist)

        if clade.clades:
            for child in clade.clades:
                node.children.append(convert_clade(child))
        else:
            node.is_leaf = True
        return node

    return convert_clade(phylo_tree.root)

def apply_regimes(root, regime_tsv_path):
    df_regimes = pd.read_csv(regime_tsv_path, index_col=0)
    all_nodes = {n.name: n for n in get_all_nodes(root)}
    for node_name, row in df_regimes.iterrows():
        if str(node_name) in all_nodes:
            all_nodes[str(node_name)].regime = str(row['regime'])

def apply_expression_data(root, expr_tsv_path):
    df_expr = pd.read_csv(expr_tsv_path, sep='\t', index_col=0)
    leaves = {n.name: n for n in get_leaves(root)}
    for cell_name, row in df_expr.iterrows():
        if str(cell_name) in leaves:
            leaf = leaves[str(cell_name)]
            leaf.q_mu = row['q_mean']
            leaf.q_var = row['q_std']**2
            leaf.read_count = row['read_count']

def load_ou_params(ou_tsv_path):
    df_ou = pd.read_csv(ou_tsv_path, sep='\t', index_col=0)
    ou_params = {}
    for regime, row in df_ou.iterrows():
        ou_params[str(regime)] = (row['alpha'], row['sigma']**2, row['theta'])
    return ou_params

def load_bm_params(tsv_path):
    """
    Reads TSV with header: mu  sigma
    Returns (sigma_sq, root_prior) where root_prior = (mu, sigma_sq).
    """
    df = pd.read_csv(tsv_path, sep=None, engine='python')
    mu_val = df['mu'].iloc[0]
    sigma_val = df['sigma'].iloc[0]
    sigma_sq = sigma_val**2
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

def plot_circular_tree(root, outer, output_path):
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

    valid_rcs = [leaf.read_count for leaf in leaves if leaf.read_count is not None]
    rc_max = max(valid_rcs) if valid_rcs else 1.0
    rc_min = min(valid_rcs) if valid_rcs else 0.0

    if outer:
        global_min = min(mu_min, rc_min)
        global_max = max(mu_max, rc_max)
    else:
        global_min = mu_min
        global_max = mu_max

    norm_var = mcolors.Normalize(vmin=v_min - (v_max - v_min)*0.15, vmax=v_max)

    # Diverging colormap centered on root: blue=low, white=root, red=high
    root_mu = root.true_mu
    cmap_mu = plt.cm.RdBu_r
    norm_mu = mcolors.TwoSlopeNorm(vcenter=root_mu, vmin=global_min, vmax=global_max)

    cmap_var = plt.cm.Blues

    # Cartesian circular layout (same rendering as plot_tree_circular.py)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")

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
                    color=col_parent, linewidth=0.8, solid_capstyle='round')

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
                ax.plot([x0, x1], [y0, y1], color=col, linewidth=0.8,
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
            if leaf.read_count is not None and leaf.read_count > 0:
                bar_len = (leaf.read_count / rc_max) * (max_r * 0.15)
                x0 = bar_base * np.cos(leaf.theta)
                y0 = bar_base * np.sin(leaf.theta)
                x1 = (bar_base + bar_len) * np.cos(leaf.theta)
                y1 = (bar_base + bar_len) * np.sin(leaf.theta)
                ax2.plot([x0, x1], [y0, y1],
                         color=cmap_rc(norm_rc(leaf.read_count)), lw=1.5,
                         solid_capstyle='butt')

    margin = max_r * 1.25 if outer else max_r * 1.10
    for ax in (ax1, ax2):
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    sm_var = plt.cm.ScalarMappable(norm=norm_var, cmap=cmap_var)
    cbar1 = fig.colorbar(sm_var, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.08, shrink=0.7)
    cbar1.ax.set_title("Variance", loc='left', fontsize=10)

    sm_mu = plt.cm.ScalarMappable(norm=norm_mu, cmap=cmap_mu)
    cbar2 = fig.colorbar(sm_mu, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.08, shrink=0.7)
    cbar2.ax.set_title("Value", loc='left', fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    if output_path.endswith('.png'):
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Successfully saved figure to {output_path}")

# ==========================================
# CLI Execution
# ==========================================

def run_reconst():
    parser = argparse.ArgumentParser(description="1D Ancestral State Reconstruction via Belief Propagation")
    parser.add_argument("--tree", required=True, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, help="Path to leaf expression data (TSV)")
    parser.add_argument("--model", required=False, type=str, choices=["ou", "bm"], default="ou", help="Model type: 'ou' (Ornstein-Uhlenbeck, default) or 'bm' (Brownian Motion)")
    parser.add_argument("--regime", required=False, help="Path to node regime mapping (TSV, ou model only)")
    parser.add_argument("--ou", required=False, help="Path to OU parameters table (TSV, ou model only)")
    parser.add_argument("--bm", required=False, help="Path to BM parameters table (TSV, bm model only)")
    parser.add_argument("--no_outer", action='store_false', help="Whether to plot read counts as bars outside the tree")
    parser.add_argument("--out_fig", required=False, default="history.png", help="Path to save the output figure (e.g., plot.png)")
    parser.add_argument("--out_tsv", required=False, default="history.tsv", help="Path to save the inferred true means and vars (TSV)")

    args = parser.parse_args()

    plt.close('all')

    print("Loading data...")
    root = load_tree_from_newick(args.tree)
    apply_expression_data(root, args.expression)
    root.dist = 0.0

    if args.model == "ou":
        if not args.regime or not args.ou:
            parser.error("--regime and --ou are required for the OU model")
        apply_regimes(root, args.regime)
        ou_params = load_ou_params(args.ou)
        transition_fn = _make_transition_fn("ou", ou_params)
        r_alpha, r_sig2, r_theta = ou_params[root.regime]
        root_prior = (r_theta, r_sig2 / (2 * r_alpha))
    else:  # bm
        if not args.bm:
            parser.error("--bm is required for the BM model")
        sigma_sq, root_prior = load_bm_params(args.bm)
        transition_fn = _make_transition_fn("bm", sigma_sq)

    print("Running upward pass (leaves -> root)...")
    upward_pass(root, transition_fn)

    print("Running downward pass (root -> leaves)...")
    downward_pass(root, transition_fn, root_prior=root_prior)

    print("Exporting data...")
    save_inferred_history(root, args.out_tsv, include_regime=(args.model == "ou"))

    print("Generating plot...")
    plot_circular_tree(root, args.no_outer, args.out_fig)


if __name__ == "__main__":
    run_reconst()
