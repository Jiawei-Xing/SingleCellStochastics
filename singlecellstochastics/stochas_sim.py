import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import math
from Bio import Phylo


def read_tree(tree_file, regime_file):
    # read and edit tree
    tree = Phylo.read(tree_file, "newick")
    total_length = max(tree.depths().values())
    clades = [n for n in tree.find_clades()] # sort by depth-first search
    cells = sorted([n.name for n in tree.get_terminals()])
    i = 0
    for clade in clades:
        # Add and normalize branch lengths
        if clade.branch_length is None: 
            clade.branch_length = 0
        else:
            clade.branch_length = clade.branch_length / total_length
        
        # Add internal node names 
        if clade.name is None: 
            clade.name = "node" + str(i)
            i = i + 1

    # most recent common ancestor
    root = [Phylo.BaseTree.Clade(branch_length=0.0, name='node0')]
    paths = {clade.name: root + tree.get_path(clade) for clade in clades}
    mrca = {}
    for i in range(len(cells)):
        for j in range(i, len(cells)):
            m, n = cells[i], cells[j]
            ancestor = [a.name for a in paths[m] if a in paths[n]][-1]
            mrca[(m, n)] = ancestor
            mrca[(n, m)] = ancestor

    # regime for thetas
    with open(regime_file, 'r') as f:
        csv_file = csv.reader(f)
        next(csv_file)
        node_regime = {}
        for row in csv_file:
            if row[1] == '':
                node = mrca[(row[0], row[0])]
            else:
                node = mrca[(row[0], row[1])]
            node_regime[node] = row[2]

    # node depth
    depth = {}
    for clade in clades:
        depth[clade.name] = sum([x.branch_length for x in paths[clade.name]])

    return clades, cells, paths, node_regime, depth

# simulate gene expression with BM process for all lineages
def BM_expression(node, parent_expr=None, expr={}, root_expr=None, BM_sigma2=None):
    
    # simulate expr for this node
    if node.name == 'node0': # init expr for root
        new_expr = root_expr
    else: # BM process
        mean = parent_expr
        var = BM_sigma2 * node.branch_length
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        new_expr = np.random.normal(loc=mean, scale=std)

    # Add expr record
    new_expr = max(0, new_expr)
    expr[node.name] = new_expr
    
    # Recursively simulate for child nodes
    for child in node.clades:
        BM_expression(child, new_expr, expr, root_expr, BM_sigma2)

    return expr

# simulate gene expression with OU process for the testing lineage
def OU_expression(node, parent_expr=None, expr={}, root_expr=None, test_regime=None, optim=None, alpha=None, OU_sigma2=None, BM_sigma2=None, node_regime=None):
    
    # simulate expr for this node
    if node.name == 'node0': # init expr for root
        new_expr = root_expr
    elif node_regime[node.name] == test_regime: # OU process
        mean = optim + (parent_expr - optim) * np.exp(-alpha * node.branch_length)
        var = OU_sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * node.branch_length))
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        new_expr = np.random.normal(loc=mean, scale=std)
    else: # BM process
        mean = parent_expr
        var = BM_sigma2 * node.branch_length
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        new_expr = np.random.normal(loc=mean, scale=std)

    # Add expr record
    new_expr = max(0, new_expr)
    expr[node.name] = new_expr

    # Recursively simulate for child nodes
    for child in node.clades:
        OU_expression(child, new_expr, expr, root_expr, test_regime, optim, alpha, OU_sigma2, BM_sigma2, node_regime)

    return expr

# simulate gene expression by BM or OU
def simulate(mode, clades, cells, paths, node_regime, depth, n_genes, root_expr, test_regime, optim=None, alpha=None, OU_sigma2=None, BM_sigma2=None):
    read_counts = []
    plots = []
    for _ in range(n_genes):
        plot = []
        if mode == "BM":
            expr = BM_expression(node=clades[0], root_expr=root_expr, BM_sigma2=BM_sigma2)
        elif mode == "OU":
            expr = OU_expression(node=clades[0], root_expr=root_expr, test_regime=test_regime, optim=optim, alpha=alpha, OU_sigma2=OU_sigma2, BM_sigma2=BM_sigma2, node_regime=node_regime)
        read_counts.append({})

        # BM-OU
        for path in paths.values():
            for i in range(len(path) - 1):
                clade1 = path[i]
                clade2 = path[i + 1]
                x = (depth[clade1.name], depth[clade2.name])
                y = (expr[clade1.name], expr[clade2.name])
                regime = node_regime[clade2.name]
                
                if regime == test_regime:
                    plot.append((x, y, "OU"))
                else:
                    plot.append((x, y, "BM"))

        # Poisson
        for cell in cells:
            read = np.random.poisson(expr[cell], 1)
            read_counts[-1][cell] = read[0]
            x = (depth[cell], 1.1)
            y = (expr[cell], read[0])
            regime = node_regime[cell]
                
            if regime == test_regime:
                plot.append((x, y, "OU"))
            else:
                plot.append((x, y, "BM"))
        
        plots.append(plot)

    return plots, read_counts

def plot(mode, plots, n_genes, output_dir, label):
    cols = math.ceil(math.sqrt(n_genes))
    rows = math.ceil(n_genes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    # Ensure axes is a 2D array and flatten
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.atleast_2d(axes)
    axes_flat = axes.ravel()

    for gene, plot in enumerate(plots):
        ax = axes_flat[gene]
        for l in set(plot):
            if l[2] == "BM":
                ax.plot(l[0], l[1], color="black", marker="o", markersize=0.1, linewidth=0.1)
            else:
                ax.plot(l[0], l[1], "ro-", markersize=0.1, linewidth=0.5)
        ax.set_title(f"{mode} gene {gene+1}")

    # Hide unused axes (if any)
    for ax in axes_flat[n_genes:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(f"{output_dir}/sim_{mode}_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def write_read_counts(mode, read_counts, cells, n_genes, output_dir, label):
    with open(f"{output_dir}/readcounts_{mode}_{label}.tsv", 'w') as f:
        for i in range(1, n_genes + 1):
            f.write("\t" + str(i))
        f.write("\n")
        for cell in cells:
            f.write(f"{cell}")
            for gene in read_counts:
                f.write(f"\t{gene[cell]}")
            f.write("\n")


def run_stochas_sim():
    parser = argparse.ArgumentParser("Stochastic simulation of gene expression evolution along lineage")
    parser.add_argument("--tree", type=str, required=True, help="File path of input tree")
    parser.add_argument("--regime", type=str, required=True, help="File path of input regime")
    parser.add_argument("--test", type=str, required=True, help="Regime for testing (OU)")
    parser.add_argument("--root", type=int, required=True, help="Starting expression at the root")
    parser.add_argument("--mode", type=str, default="BM,OU", help="Mode for simulation: BM or OU or BM,OU")
    parser.add_argument("--n_genes", type=int, default=100, help="Number of genes to simulate")
    parser.add_argument("--BM_sigma2", type=float, default=5.0, help="Variance for BM")
    parser.add_argument("--optim", type=int, default=None, help="Optimal expression for OU")
    parser.add_argument("--alpha", type=float, default=5.0, help="Selective strength for OU")
    parser.add_argument("--OU_sigma2", type=float, default=10.0, help="Variance for OU")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--label", type=str, default="", help="Label for output")
    args = parser.parse_args()

    # read tree and regime
    clades, cells, paths, node_regime, depth = read_tree(args.tree, args.regime)

    # simulate gene expression
    if "BM" in args.mode:
        plots, read_counts = simulate("BM", clades, cells, paths, node_regime, depth, 
                                    args.n_genes, args.root, args.test, BM_sigma2=args.BM_sigma2)
        plot("BM", plots, args.n_genes, args.out, args.label)
        write_read_counts("BM", read_counts, cells, args.n_genes, args.out, args.label)

    if "OU" in args.mode:
        plots, read_counts = simulate("OU", clades, cells, paths, node_regime, depth, 
                                    args.n_genes, args.root, args.test, args.optim, args.alpha, args.OU_sigma2, args.BM_sigma2)
        plot("OU", plots, args.n_genes, args.out, args.label)
        write_read_counts("OU", read_counts, cells, args.n_genes, args.out, args.label)


if __name__ == "__main__":
    run_stochas_sim()
