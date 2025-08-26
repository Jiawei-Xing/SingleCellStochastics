import numpy as np
import pandas as pd
from Bio import Phylo
import csv
import torch


# precompute and store input data for later usage
def process_data(tree_files, gene_files, regime_files, rnull, device="cpu"):
    """
    Precompute input data and store for optimization.

    Returns:
    tree_list: list of trees
    cells_list: list of cells
    df_list: list of expression data
    diverge_list: list of divergence matrices (tree)
    share_list: list of share matrices (tree)
    epochs_list: list of depths (tree)
    beta_list: list of active regimes (tree)
    """
    tree_list = []
    cells_list = []
    df_list = []
    diverge_list = []
    share_list = []
    epochs_list = []
    beta_list = []
    diverge_list_torch = []
    share_list_torch = []
    epochs_list_torch = []
    beta_list_torch = []
    regime_list = []

    for tree_file, gene_file, regime_file in zip(tree_files, gene_files, regime_files):
        # Read the phylogenetic tree
        tree = Phylo.read(tree_file, "newick")
        cells = sorted([n.name for n in tree.get_terminals()])
        tree_list.append(tree)
        cells_list.append(cells)

        # Read the gene read count matrix, reorder cells as tree leaves
        df = pd.read_csv(gene_file, sep="\t", index_col=0, header=0)
        df.index = df.index.astype(str)  # cell names
        df = df.loc[cells]  # important to match cell orders!
        df_list.append(df)

        # Edit tree
        total_length = max(tree.depths().values())
        clades = [n for n in tree.find_clades()]  # sort by depth-first search
        i = 0
        for clade in clades:
            # Re-scale branch lengths to [0, 1]
            if clade.branch_length is None:
                clade.branch_length = 0
            else:
                clade.branch_length = clade.branch_length / total_length

            # Fix internal node names
            if clade.name is None:
                clade.name = "internal_node" + str(i)
                i = i + 1

        # Store paths from each node to root (add an extra root)
        # root = [Phylo.BaseTree.Clade(branch_length=0.0, name='root')]
        # paths = {'root': root}
        # paths.update({clade.name: root + tree.get_path(clade) for clade in clades})
        paths = {clade.name: tree.get_path(clade) for clade in clades}

        # Precompute node index mapping for all nodes and cells
        node_names = list(paths.keys())
        node_idx = {name: i for i, name in enumerate(node_names)}
        cell_indices = np.array([node_idx[cell] for cell in cells])

        # Precompute paths sets and reverse lineages for each cell
        cell_paths = [set(a.name for a in paths[cell]) for cell in cells]
        cell_lineages = [[a.name for a in paths[cell]][::-1] for cell in cells]

        # Build MRCA index matrix for all cell pairs
        mrca_idx = np.zeros((len(cells), len(cells)), dtype=int)
        for i in range(len(cells)):
            for j in range(len(cells)):
                # Fast set intersection
                common_ancestors = cell_paths[i] & cell_paths[j]
                # Find the deepest (last in path) ancestor
                for ancestor in cell_lineages[i]:
                    if ancestor in common_ancestors:
                        mrca_idx[i, j] = node_idx[ancestor]
                        break

        # depths (time intervals) for all nodes
        depth = np.array(
            [sum(x.branch_length for x in paths[name]) for name in node_names],
            dtype=np.float32,
        )
        share_mat = depth[mrca_idx]  # depth of MRCA
        share_torch = torch.tensor(share_mat, dtype=torch.float32, device=device)

        # diverge_mat: depth[a] + depth[b] - 2 * depth[mrca(a, b)]
        depth_a = depth[cell_indices][:, None]  # shape (n_cells, 1)
        depth_b = depth[cell_indices][None, :]  # shape (1, n_cells)
        diverge_mat = depth_a + depth_b - 2 * share_mat
        diverge_torch = torch.tensor(diverge_mat, dtype=torch.float32, device=device)

        # depths of each cell lineage (leaf to root)
        epochs = [
            np.array([depth[node_idx[x]] for x in lineage]) for lineage in cell_lineages
        ]
        epochs_torch = [
            torch.tensor(ep, dtype=torch.float32, device=device) for ep in epochs
        ]

        # regime for thetas
        with open(regime_file, "r") as f:
            csv_file = csv.reader(f)
            next(csv_file)
            node_regime = {}
            regimes = set()
            for row in csv_file:
                if row[1] == "":
                    # mrca of a cell with itself
                    cell_idx = cells.index(row[0])
                    node = mrca_idx[cell_idx, cell_idx]
                else:
                    # mrca of two cells
                    cell1_idx = cells.index(row[0])
                    cell2_idx = cells.index(row[1])
                    node = mrca_idx[cell1_idx, cell2_idx]

                # regime for each node
                node_regime[node] = row[2]
                regimes.add(row[2])

        # sort regimes and move rnull to 1st
        regimes = sorted(list(regimes))
        regimes.remove(rnull)
        regimes.insert(0, rnull)
        regime_list.append(regimes)
        regime_idx = {regime: i for i, regime in enumerate(regimes)}

        # beta is binary label for activation of regime along each lineage (leaf to root)
        beta = []
        for lineage in cell_lineages:
            regime_indices = [
                regime_idx[node_regime[node_idx[x]]] for x in lineage
            ]  # active regimes
            beta_matrix = np.eye(len(regimes), dtype=int)[
                regime_indices
            ]  # (n_lineage, n_regimes)
            beta.append(beta_matrix)
        beta_torch = [
            torch.tensor(beta_matrix, dtype=torch.float32, device=device)
            for beta_matrix in beta
        ]

        diverge_list.append(diverge_mat)
        share_list.append(share_mat)
        epochs_list.append(epochs)
        beta_list.append(beta)
        diverge_list_torch.append(diverge_torch)
        share_list_torch.append(share_torch)
        epochs_list_torch.append(epochs_torch)
        beta_list_torch.append(beta_torch)

    return (
        tree_list,
        cells_list,
        df_list,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        diverge_list_torch,
        share_list_torch,
        epochs_list_torch,
        beta_list_torch,
        regime_list,
    )
