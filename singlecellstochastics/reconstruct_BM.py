import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def get_bm_params(sigma_sq, t):
    """BM transition: Mean stays same (F=1, c=0), variance grows by sig2*t."""
    return 1.0, 0.0, sigma_sq * t

def upward_pass(node, sigma_sq):
    for child in node.children:
        upward_pass(child, sigma_sq)
        node.up_lam += child.msg_to_p_lam
        node.up_eta += child.msg_to_p_eta

    if node.is_leaf:
        node.up_lam += 1.0 / node.q_var
        node.up_eta += node.q_mu / node.q_var

    if node.dist > 0:
        F, c, Q = get_bm_params(sigma_sq, node.dist)
        denom = Q * node.up_lam + 1.0
        node.msg_to_p_lam = (F**2 * node.up_lam) / denom
        node.msg_to_p_eta = F * (node.up_eta - node.up_lam * c) / denom

def downward_pass(node, sigma_sq, root_prior):
    if root_prior is not None:
        prior_mu, prior_var = root_prior
        node.true_lam = node.up_lam + (1.0 / prior_var)
        node.true_eta = node.up_eta + (prior_mu / prior_var)
    
    node.true_var = 1.0 / node.true_lam
    node.true_mu = node.true_eta / node.true_lam

    for child in node.children:
        cav_lam = node.true_lam - child.msg_to_p_lam
        cav_eta = node.true_eta - child.msg_to_p_eta
        
        cav_var = 1.0 / cav_lam if cav_lam > 1e-12 else 1e8
        cav_mu = cav_eta / cav_lam if cav_lam > 1e-12 else 0.0
        
        F, c, Q = get_bm_params(sigma_sq, child.dist)
        
        down_mu = F * cav_mu + c
        down_var = F**2 * cav_var + Q
        
        child.true_lam = child.up_lam + (1.0 / down_var)
        child.true_eta = child.up_eta + (down_mu / down_var)
        downward_pass(child, sigma_sq, root_prior=None)
        
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

def apply_expression_data(root, expr_tsv_path):
    df_expr = pd.read_csv(expr_tsv_path, sep='\t', index_col=0)
    leaves = {n.name: n for n in get_leaves(root)}
    for cell_name, row in df_expr.iterrows():
        if str(cell_name) in leaves:
            leaf = leaves[str(cell_name)]
            leaf.q_mu = row['q_mean']
            leaf.q_var = row['q_std']**2
            leaf.read_count = row['read_count']

def load_simple_bm_params(tsv_path):
    """
    Reads TSV with header: mu  sigma
    Example: 5.744412  2903.189
    """
    # Using sep=None to catch both tabs and multiple spaces
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

def save_inferred_history(root, output_tsv_path):
    """Saves the reconstructed means and variances to a TSV file."""
    all_nodes = get_all_nodes(root)
    data = []
    
    for node in all_nodes:
        data.append({
            'node_name': node.name,
            'is_leaf': node.is_leaf,
            'infer_mu': node.true_mu,
            'infer_var': node.true_var
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_tsv_path, sep='\t', index=False)
    print(f"Successfully saved inferred history table to {output_tsv_path}")

def layout_circular_tree(node, r_current=0.0, leaf_angles=None):
    node.r = r_current + node.dist
    if node.is_leaf:
        node.theta = leaf_angles.pop(0)
    else:
        for child in node.children:
            layout_circular_tree(child, node.r, leaf_angles)
        node.theta = np.mean([c.theta for c in node.children])

def plot_circular_tree(root, outer, output_path):
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
    
    # --- FIX: Unified Global Scale ---
    # Find the absolute min and max across BOTH the tree and the read counts
    if outer:
        global_min = min(mu_min, rc_min)
        global_max = max(mu_max, rc_max)
    else:
        global_min = mu_min
        global_max = mu_max
    
    norm_var = mcolors.Normalize(vmin=v_min - (v_max - v_min)*0.15, vmax=v_max)
    # Use one shared normalizer for the entire right-hand plot
    norm_shared = mcolors.Normalize(vmin=global_min - (global_max - global_min)*0.15, vmax=global_max)
    
    cmap_var = plt.cm.Blues
    cmap_mu = plt.cm.Reds

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(14, 7))
    
    def draw_tree(ax, node, val_func, cmap, norm):
        if node.children:
            thetas = [c.theta for c in node.children]
            th_min, th_max = min(thetas), max(thetas)
            arc_th = np.linspace(th_min, th_max, 100)
            
            ax.plot(arc_th, [node.r]*100, color=cmap(norm(val_func(node))), lw=1.5, zorder=1)
            
            for child in node.children:
                r_vals = np.linspace(node.r, child.r, 20) 
                val_start = val_func(node)
                val_end = val_func(child)
                val_interp = np.linspace(val_start, val_end, 20)
                
                for i in range(len(r_vals)-1):
                    ax.plot([child.theta, child.theta], 
                            [r_vals[i], r_vals[i+1]], 
                            color=cmap(norm(val_interp[i])), lw=1.5, zorder=1, solid_capstyle='butt')
                
                draw_tree(ax, child, val_func, cmap, norm)

    # 1. Plot Variance
    ax1.set_title("Variance of Prediction", pad=20)
    ax1.axis('off')
    draw_tree(ax1, root, lambda n: n.true_var, cmap_var, norm_var)
    #ax1.fill_between(np.linspace(0, 2*np.pi, 100), max_r*1.05, max_r*1.20, color='#F0F8FF', alpha=0.5)

    # 2. Plot Expression (Tree uses norm_shared)
    ax2.set_title("Predicted Expression", pad=20)
    ax2.axis('off')
    draw_tree(ax2, root, lambda n: n.true_mu, cmap_mu, norm_shared)
    
    # 3. Plot Read Counts (Bars ALSO use norm_shared)
    if outer:
        ax2.fill_between(np.linspace(0, 2*np.pi, 100), max_r*1.05, max_r*1.20, color='#FFF5F0', alpha=0.5)
        for leaf in leaves:
            if leaf.read_count is not None and leaf.read_count > 0:
                bar_len = (leaf.read_count / rc_max) * (max_r * 0.15) 
                ax2.plot([leaf.theta, leaf.theta], [max_r*1.05, max_r*1.05 + bar_len], 
                        color=cmap_mu(norm_shared(leaf.read_count)), lw=2.0, solid_capstyle='butt')

    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    sm_var = plt.cm.ScalarMappable(norm=norm_var, cmap=cmap_var)
    cbar1 = fig.colorbar(sm_var, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.1, shrink=0.7)
    cbar1.ax.set_title("Variance", loc='left', fontsize=10)

    # The colorbar now reflects the true global scale (0 to 10+)
    sm_mu = plt.cm.ScalarMappable(norm=norm_shared, cmap=cmap_mu)
    cbar2 = fig.colorbar(sm_mu, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.1, shrink=0.7)
    cbar2.ax.set_title("Value", loc='left', fontsize=10)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved figure to {output_path}")

# ==========================================
# CLI Execution
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Ancestral State Reconstruction via Belief Propagation")
    parser.add_argument("--tree", required=True, help="Path to Newick tree file (.nwk)")
    parser.add_argument("--expression", required=True, help="Path to leaf expression data (TSV)")
    parser.add_argument("--bm", required=True, help="Path to BM parameters table (TSV)")
    parser.add_argument("--no_outer", action='store_false', help="Whether to plot read counts as bars outside the tree")
    parser.add_argument("--out_fig", required=False, default="history.png", help="Path to save the output figure (e.g., plot.png)")
    parser.add_argument("--out_tsv", required=False, default="history.tsv", help="Path to save the inferred true means and vars (TSV)")
    
    args = parser.parse_args()
    
    plt.close('all')

    root = load_tree_from_newick(args.tree)
    apply_expression_data(root, args.expression)
    root.dist = 0.0 

    # 1. Load the single mu and sigma
    sigma_sq, root_prior = load_simple_bm_params(args.bm)
    
    # 2. Upward Pass (uniform sigma_sq)
    upward_pass(root, sigma_sq)

    # 3. Downward Pass (with root prior)
    downward_pass(root, sigma_sq, root_prior=root_prior)

    # 4. Save and Plot
    save_inferred_history(root, args.out_tsv)
    # The plotting function needs a minor tweak to ignore regimes
    plot_circular_tree(root, args.no_outer, args.out_fig)

