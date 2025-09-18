import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import math
from Bio import Phylo


def read_tree(
    newick_file: str,
    normalize_branch_lengths: bool = True,
    name_unnamed_nodes: bool = False    
) -> Phylo.BaseTree.Tree:
    """
    Read a Newick tree, then optionally normalize branch lengths and assign internal node names.
    
    Args:
        newick_file (str): Path to the Newick-formatted tree file.
        normalize_branch_lengths (bool): If True, normalize branch lengths so that the longest root-to-tip path is 1.0.
        name_unnamed_nodes (bool): If True, assign unique names to unnamed internal nodes.

    Returns:
        Tree: A Biopython `Tree` object.
    
    """
    tree = Phylo.read(newick_file, "newick")
    total_length = max(tree.depths().values())

    # Normalize and name nodes
    i = 0
    for c in tree.find_clades():
        if normalize_branch_lengths:
            c.branch_length = (c.branch_length or 0) / total_length
        if c.name is None:
            if name_unnamed_nodes:
                c.name = f"node{i}"
                i += 1
            else:
                raise ValueError("All nodes must be named or set name_unnamed_nodes to True")

    return tree


def assign_nodes_to_regimes(
    tree: Phylo.BaseTree.Tree, 
    regime_file: str
) -> Phylo.BaseTree.Tree:
    """
    Assign regime labels to nodes in the tree based on a CSV file mapping node names to regimes.
    
    Args:
        tree (Tree): A Biopython `Tree` object.
        regime_file (str): Path to a CSV file with two columns: node name and regime label.
    
    Returns:
        Tree: The input tree with an added 'regime' attribute for each node.
    """
    # Directly match node names to regimes
    with open(regime_file) as f:
        reader = csv.reader(f)
        next(reader)
        regime_map = {row[0]: row[1] for row in reader}

    for clade in tree.find_clades():
        if clade.name in regime_map:
            clade.regime = regime_map[clade.name]
        else:
            raise ValueError(f"Clade {clade.name} not found in regime file")

    return tree


def get_ou_expr_one_branch(
    parent_expr: float,
    optim: float,
    alpha: float,
    branch_length: float,
    sigma2: float
) -> float:
    """
    Simulate gene expression for a single branch under the Ornstein-Uhlenbeck (OU) process.

    Args:
        parent_expr (float): Expression value at the parent node.
        optim (float): Optimal expression value (theta) for the OU process.
        alpha (float): Selective strength parameter for the OU process.
        branch_length (float): Length of the branch.
        sigma2 (float): Variance parameter for the OU process.

    Returns:
        float: Simulated expression value at the child node.
    """
    mean = optim + (parent_expr - optim) * np.exp(-alpha * branch_length)
    var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * branch_length))
    std = np.sqrt(np.maximum(var, 1e-10))
    new_expr = np.random.normal(loc=mean, scale=std)
    return new_expr


def get_bm_expr_one_branch(
    parent_expr: float,
    branch_length: float,
    sigma2: float
) -> float:
    """
    Simulate gene expression for a single branch under the Brownian Motion (BM) process.
    
    Args:
        parent_expr (float): Expression value at the parent node.
        branch_length (float): Length of the branch.
        sigma2 (float): Variance parameter for the BM process.
    
    Returns:
        float: Simulated expression value at the child node.
    """
    var = sigma2 * branch_length
    std = np.sqrt(np.maximum(var, 1e-10))
    new_expr = np.random.normal(loc=parent_expr, scale=std)
    return new_expr
    

def reset_all_nodes_expr(
    tree: Phylo.BaseTree.Tree
) -> None:
    """
    Reset expression values for all nodes in the tree.
    
    Args:
        tree (Tree): A Biopython `Tree` object.
        
    Returns:
        None
    """
    for node in tree.find_clades():
        node.expr = None


def reset_all_nodes_read_counts(
        tree: Phylo.BaseTree.Tree
) -> None:
    """
    Reset read counts for all nodes in the tree.
    
    Args:
        tree (Tree): A Biopython `Tree` object.
        
    Returns:
        None
    """
    for node in tree.find_clades():
        node.read_count = None
        

def get_latent_gene_expression_at_tips(
    tree: Phylo.BaseTree.Tree,
    test_regime: str = None,
    root_expr: float = None,
    optim: float = None,
    alpha: float = None,
    sigma2: float = None,
    background_model: str = "OU"
) -> None:
    """
    Recursively simulate latent gene expression values at the tips of a phylogenetic tree under the OU or BM process.

    Args:
        tree: A Biopython `Tree` object.
        test_regime (str, optional): Regime label for which to apply the test OU process.
        root_expr (float, optional): Expression value at the root node.
        optim (float, optional): Optimal expression value (theta) for the OU process.
        alpha (float, optional): Selective strength parameter for the OU process.
        sigma2 (float, optional): Variance parameter for the OU/BM process.
        background_model (str, optional): Model for background lineages ("OU" or "BM").

    Returns:
        None
    """
    def set_expr(node, parent_expr):
        # If at root, set its expression to the root expression
        if node == tree.root:
            new_expr = root_expr
        else:
            # Make sure node has a regime assigned
            if not hasattr(node, "regime"):
                raise ValueError(f"Node {node.name} does not have a regime assigned.")
            
            # Simulate expression based on regime and model
            if node.regime == test_regime:
                new_expr = get_ou_expr_one_branch(parent_expr, optim, alpha, node.branch_length, sigma2)
            elif background_model == "OU":
                new_expr = get_ou_expr_one_branch(parent_expr, root_expr, alpha, node.branch_length, sigma2)
            elif background_model == "BM":
                new_expr = get_bm_expr_one_branch(parent_expr, node.branch_length, sigma2)
            else:
                raise ValueError("background_model input must be OU or BM")
        
        node.expr = new_expr
        
        # Recurse to children
        for child in node.clades:
            set_expr(child, new_expr)
            
    # Start recursion from the root
    set_expr(tree.root, None)


def clamp_latent_gene_expression_at_tips(
    tree: Phylo.BaseTree.Tree, 
    method: str = "clamp"
) -> None:
    """
    Clamp or transform negative latent gene expression values at the tips of the tree.
    
    Args:
        tree (Tree): A Biopython `Tree` object with simulated expression values.
        method (str): Method to handle negative expressions. Options are "clamp" (set
        
    Returns:
        None
    """
    for node in tree.get_terminals():
        if method == "clamp":
            node.expr = max(0, node.expr)
        elif method == "softplus":
            node.expr = math.log(1 + math.exp(node.expr))
        else:
            raise ValueError("method input must be clamp or softplus")


def get_poisson_sampled_read_counts(
    tree: Phylo.BaseTree.Tree
) -> None:
    """
    Simulate observed read counts at the tips of the tree by sampling from a Poisson distribution
    with the latent expression values as the rate parameter.
    
    Args:
        tree (Tree): A Biopython `Tree` object with clamped expression values.
        
    Returns:
        None
    """
    for node in tree.get_terminals():
        node.read_count = np.random.poisson(node.expr)


# simulate gene expression by BM or OU
def simulate(tree, n_genes, root_expr, test_regime, optim=None, alpha=None, sigma2=None):
    read_counts = {}
    plots = []
    for i in range(n_genes):
        plot = []
        
        # Clean the tree from any previous genes
        reset_all_nodes_expr(tree)
        reset_all_nodes_read_counts(tree)
        
        get_latent_gene_expression_at_tips(tree=tree, test_regime=test_regime, root_expr=root_expr, optim=optim, alpha=alpha, sigma2=sigma2, background_model="OU")
        clamp_latent_gene_expression_at_tips(tree)
        get_poisson_sampled_read_counts(tree)
        read_counts[i] = {node.name: node.read_count for node in tree.get_terminals()}

        # # BM-OU
        # for path in paths.values():
        #     for i in range(len(path) - 1):
        #         clade1 = path[i]
        #         clade2 = path[i + 1]
        #         x = (depth[clade1.name], depth[clade2.name])
        #         y = (expr[clade1.name], expr[clade2.name])
        #         regime = node_regime[clade2.name]
                
        #         if regime == test_regime:
        #             plot.append((x, y, "h1"))
        #         else:
        #             plot.append((x, y, "h0"))

        # # Poisson
        # for cell in cells:
        #     read = np.random.poisson(expr[cell], 1)
        #     read_counts[-1][cell] = read[0]
        #     x = (depth[cell], 1.1)
        #     y = (expr[cell], read[0])
        #     regime = node_regime[cell]
                
        #     if regime == test_regime:
        #         plot.append((x, y, "h1"))
        #     else:
        #         plot.append((x, y, "h0"))
        
        # plots.append(plot)
        
    cells = [node.name for node in tree.get_terminals()]
    
    return cells, read_counts


# def plot(plots, n_genes, output_dir, label):
#     cols = math.ceil(math.sqrt(n_genes))
#     rows = math.ceil(n_genes / cols)

#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
#     # Ensure axes is a 2D array and flatten
#     if rows * cols == 1:
#         axes = np.array([[axes]])
#     elif rows == 1 or cols == 1:
#         axes = np.atleast_2d(axes)
#     axes_flat = axes.ravel()

#     for gene, plot in enumerate(plots):
#         ax = axes_flat[gene]
#         for l in set(plot):
#             if l[2] == "h0":
#                 ax.plot(l[0], l[1], color="black", marker="o", markersize=0.1, linewidth=0.1)
#             else:
#                 ax.plot(l[0], l[1], "ro-", markersize=0.1, linewidth=0.5)
#         ax.set_title(f"gene {gene+1}")

#     # Hide unused axes (if any)
#     for ax in axes_flat[n_genes:]:
#         ax.set_visible(False)

#     plt.tight_layout(pad=0.5)
#     plt.savefig(f"{output_dir}/sim_{label}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)


def write_read_counts(read_counts, cells, n_genes, output_dir, label):
    with open(f"{output_dir}/readcounts_{label}.tsv", 'w') as f:
        # Write header for all gene names
        f.write("\t" + "\t".join(str(i) for i in range(1, n_genes + 1)) + "\n")
        # Write read counts for each cell for all genes
        for cell in cells:
            counts = [str(read_counts[gene][cell]) for gene in range(n_genes)]
            f.write(f"{cell}\t" + "\t".join(counts) + "\n")

def run_stochas_sim():
    parser = argparse.ArgumentParser("Stochastic simulation of gene expression evolution along lineage")
    parser.add_argument("--tree", type=str, required=True, help="File path of input tree")
    parser.add_argument("--regime", type=str, required=True, help="File path of input regime")
    parser.add_argument("--test", type=str, required=True, help="Regime for testing (OU)")
    parser.add_argument("--root", type=int, required=True, help="Starting expression at the root")
    parser.add_argument("--n_genes", type=int, default=100, help="Number of genes to simulate")
    parser.add_argument("--sigma2", type=float, required=True, help="Variance for BM or OU")
    parser.add_argument("--optim", type=int, default=None, help="Optimal expression for OU")
    parser.add_argument("--alpha", type=float, required=True, help="Selective strength for OU")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--label", type=str, default="", help="Label for output")
    args = parser.parse_args()

    
    tree = read_tree(args.tree)
    tree = assign_nodes_to_regimes(tree, args.regime)

    cells, read_counts = simulate(tree, args.n_genes, args.root, args.test, args.optim, args.alpha, args.sigma2)
    print(read_counts)
    print(read_counts.keys())
    
    write_read_counts(read_counts, cells, args.n_genes, args.out, args.label)
    
    # plot(plots, args.n_genes, args.out, args.label)


if __name__ == "__main__":
    run_stochas_sim()
