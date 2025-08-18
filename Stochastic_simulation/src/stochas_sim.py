import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from Bio import Phylo

parser = argparse.ArgumentParser("Stochastic simulation of gene expression evolution along lineage")
parser.add_argument("--tree", type=str, required=True, help="File path of input tree")
parser.add_argument("--regime", type=str, required=True, help="File path of input regime")
parser.add_argument("--test", type=str, required=True, help="Regime for testing (OU)")
parser.add_argument("--root", type=int, required=True, help="Starting expression at the root")
parser.add_argument("--n_genes", type=int, default=100, help="Number of genes to simulate")
parser.add_argument("--BM_sigma2", type=float, default=1.0, help="Variance for BM")
parser.add_argument("--optim", type=int, default=None, help="Optimal expression for OU")
parser.add_argument("--alpha", type=float, default=None, help="Selective strength for OU")
parser.add_argument("--OU_sigma2", type=float, default=None, help="Variance for OU")
parser.add_argument("--seed", type=int, default=0, help="Starting random seed")
parser.add_argument("--out", type=str, default="readcounts.txt", help="File path of output gene expression")
args = parser.parse_args()

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
    regime = pd.read_csv(regime_file, sep=",", header=0)
    node_regime = {}
    for row in regime.itertuples():
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
def BM_expression(node, parent_expr=None, expr={}, seed=0):
    # set random seed
    np.random.seed(seed)
    
    # simulate expr for this node
    if node.name == 'node0': # init expr for root
        new_expr = args.root
    else: # BM process
        mean = parent_expr
        var = args.BM_sigma2 * node.branch_length
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        new_expr = np.random.normal(loc=mean, scale=std)

    # Add expr record
    new_expr = max(0, new_expr)
    expr[node.name] = new_expr
    
    # Recursively simulate for child nodes
    for child in node.clades:
        BM_expression(child, new_expr, expr, seed)

    return expr

# simulate gene expression with OU process for the testing lineage
def OU_expression(node, parent_expr=None, expr={}, seed=0):
    # set random seed
    np.random.seed(seed)
    
    # simulate expr for this node
    if node.name == 'node0': # init expr for root
        new_expr = args.root
    elif node.name == args.test: # OU process
        mean = args.optim + (parent_expr - args.optim) * np.exp(-args.alpha * node.branch_length)
        var = args.OU_sigma2 / (2 * args.alpha) * (1 - np.exp(-2 * args.alpha * node.branch_length))
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        new_expr = np.random.normal(loc=mean, scale=std)
    else: # BM process
        mean = parent_expr
        var = args.BM_sigma2 * node.branch_length
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        new_expr = np.random.normal(loc=mean, scale=std)

    # Add expr record
    new_expr = max(0, new_expr)
    expr[node.name] = new_expr

    # Recursively simulate for child nodes
    for child in node.clades:
        OU_expression(child, new_expr, expr, seed)

    return expr

# simulate gene expression by BM or OU
def simulate(mode, clades, cells, paths, node_regime, depth):
    results = []
    read_counts = []
    seed = args.seed
    for gene in range(args.n_genes):
        plot = []
        seed += 1
        if mode == "BM":
            expr = BM_expression(clades[0], expr={}, seed=seed)
        else:
            expr = OU_expression(clades[0], expr={}, seed=seed)
        results.append(expr)
        read_counts.append({})

        # BM-OU
        for path in paths.values():
            for i in range(len(path) - 1):
                clade1 = path[i]
                clade2 = path[i + 1]
                x = (depth[clade1.name], depth[clade2.name])
                y = (expr[clade1.name], expr[clade2.name]) # mean of cif
                regime = node_regime[clade2.name]
                
                if regime == args.test:
                    plot.append((x, y, "OU"))
                else:
                    plot.append((x, y, "BM"))

        # Poisson
        for cell in cells:
            read = np.random.poisson(expr[cell], 1)
            read_counts[-1][cell] = read[0]
            x = (depth[cell], 1.0)
            y = (expr[cell], read[0])
            regime = node_regime[cell]
                
            if regime == args.test:
                plot.append((x, y, "OU"))
            else:
                plot.append((x, y, "BM"))

        plt.figure(figsize=(3, 3))
        for l in set(plot):
            if l[2] == "BM":
                plt.plot(l[0], l[1], color="black", marker="o", markersize=0.1, linewidth=0.1)
            else:
                plt.plot(l[0], l[1], "ro-", markersize=0.1, linewidth=0.5)
        plt.title(f"gene {gene+1}")
        plt.savefig(f"gene_{gene+1}.png")
        plt.close()

    with open("readcounts_BM.tsv", 'w') as f:
        for i in range(1, args.n_genes + 1):
            f.write("\t" + str(i))
        f.write("\n")
        for cell in cells:
            f.write(f"{cell}")
            for gene in read_counts:
                f.write(f"\t{gene[cell]}")
            f.write("\n")

if __name__ == "__main__":
    clades, cells, paths, node_regime, depth = read_tree(args.tree, args.regime)
    simulate("BM", clades, cells, paths, node_regime, depth)
    simulate("OU", clades, cells, paths, node_regime, depth)

