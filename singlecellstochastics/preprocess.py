"""Input preprocessing for LAVOUS workflows.

The functions in this module align count matrices to tree leaves, normalize
branch lengths, assign regimes to lineages, and precompute covariance helper
matrices consumed by the BM/OU likelihood code.
"""

import numpy as np
import pandas as pd
from Bio import Phylo
import csv
import torch


# precompute and store input data for later usage
def _ordered_regimes(regimes, rnull, source):
    regimes = sorted(list(regimes))
    if rnull not in regimes:
        raise ValueError(f"Null/reference regime {rnull!r} is not present in {source}.")
    regimes.remove(rnull)
    regimes.insert(0, rnull)
    return regimes


def _read_node_regimes(regime_file, cells, node_idx, mrca_idx):
    with open(regime_file, "r") as f:
        csv_file = csv.reader(f)
        header = next(csv_file)
        node_regime = {}
        regimes = set()
        cell_to_idx = {cell: i for i, cell in enumerate(cells)}

        # regime file in mrca format
        if header == ["node", "node2", "regime"]:
            for row in csv_file:
                if row[1] == "":
                    # mrca of a cell with itself
                    cell_idx = cell_to_idx[row[0]]
                    node = mrca_idx[cell_idx, cell_idx]
                else:
                    # mrca of two cells
                    cell1_idx = cell_to_idx[row[0]]
                    cell2_idx = cell_to_idx[row[1]]
                    node = mrca_idx[cell1_idx, cell2_idx]
                node_regime[node] = row[2]
                regimes.add(row[2])

        # regime file in node-regime format
        elif header == ["node_name", "regime"]:
            for row in csv_file:
                node_name = row[0]
                regime = row[1]
                node = node_idx[node_name]
                node_regime[node] = regime
                regimes.add(regime)
        else:
            raise ValueError(
                f"Regime file {regime_file} must have header 'node,node2,regime' "
                "or 'node_name,regime'."
            )

    return node_regime, regimes


def _build_beta(cell_lineages, node_idx, node_regime, regimes, device):
    regime_idx = {regime: i for i, regime in enumerate(regimes)}
    beta = []
    for lineage in cell_lineages:
        regime_indices = []
        for node_name in lineage:
            idx = node_idx[node_name]
            if idx not in node_regime:
                raise ValueError(f"Regime missing for tree node {node_name!r}.")
            regime_indices.append(regime_idx[node_regime[idx]])
        beta_matrix = np.eye(len(regimes), dtype=int)[regime_indices]
        beta.append(beta_matrix)
    beta_torch = [
        torch.tensor(beta_matrix, dtype=torch.float32, device=device)
        for beta_matrix in beta
    ]
    return beta, beta_torch


def _validate_nested_regimes(alt_node_regime, null_node_regime, alt_regimes, source):
    alt_to_null = {}
    for node, alt_regime in alt_node_regime.items():
        if node not in null_node_regime:
            continue
        null_regime = null_node_regime[node]
        previous = alt_to_null.get(alt_regime)
        if previous is not None and previous != null_regime:
            raise ValueError(
                f"Null-regime file {source} is not a collapse of the alternative "
                f"partition: alternative regime {alt_regime!r} maps to both "
                f"{previous!r} and {null_regime!r}."
            )
        alt_to_null[alt_regime] = null_regime

    missing = [regime for regime in alt_regimes if regime not in alt_to_null]
    if missing:
        raise ValueError(
            f"Null-regime file {source} does not cover alternative regimes: {missing}."
        )
    return alt_to_null


def process_data_OU(
    tree_files,
    gene_files,
    regime_files,
    library_files,
    rnull,
    device,
    null_regime_files=None,
):
    """
    Precompute input data and store for optimization.

    ``regime_files`` define the alternative theta partition. If
    ``null_regime_files`` is provided, it defines a nested H0 theta partition
    on the same tree nodes; otherwise H0 is the legacy one-theta shared model.

    Returns the legacy 13-tuple when ``null_regime_files`` is None. When a
    null-regime file list is provided, appends ``null_regime_list``,
    ``null_beta_list``, and ``null_beta_list_torch``.
    """
    if null_regime_files is not None and len(null_regime_files) != len(tree_files):
        raise ValueError("--null_regime must provide one file per --tree file.")

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
    library_list = []
    null_regime_list = []
    null_beta_list = []
    null_beta_list_torch = []

    for idx_file, (tree_file, gene_file, regime_file, library_file) in enumerate(
        zip(tree_files, gene_files, regime_files, library_files)
    ):
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

        # Read library sizes if provided, reorder cells as tree leaves
        if library_file:
            lib_df = pd.read_csv(library_file, sep="\t", index_col=0, header=None)
            lib_df.index = lib_df.index.astype(str)
            lib_df = lib_df.loc[cells]
            library_list.append(lib_df)
        else:
            lib_df = pd.DataFrame(np.ones((len(cells), 1)), index=cells)
            library_list.append(lib_df)

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
                clade.name = "add_internal_node" + str(i)
                i = i + 1

        # Store paths from each node to root
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
                common_ancestors = cell_paths[i] & cell_paths[j]
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
        depth_a = depth[cell_indices][:, None]
        depth_b = depth[cell_indices][None, :]
        diverge_mat = depth_a + depth_b - 2 * share_mat
        diverge_torch = torch.tensor(diverge_mat, dtype=torch.float32, device=device)

        # depths of each cell lineage (leaf to root)
        epochs = [
            np.array([depth[node_idx[x]] for x in lineage]) for lineage in cell_lineages
        ]
        epochs_torch = [
            torch.tensor(ep, dtype=torch.float32, device=device) for ep in epochs
        ]

        node_regime, regimes = _read_node_regimes(
            regime_file, cells, node_idx, mrca_idx
        )
        regimes = _ordered_regimes(regimes, rnull, regime_file)
        beta, beta_torch = _build_beta(
            cell_lineages, node_idx, node_regime, regimes, device
        )
        regime_list.append(regimes)

        if null_regime_files is not None:
            null_file = null_regime_files[idx_file]
            null_node_regime, null_regimes = _read_node_regimes(
                null_file, cells, node_idx, mrca_idx
            )
            null_regimes = _ordered_regimes(null_regimes, rnull, null_file)
            _validate_nested_regimes(node_regime, null_node_regime, regimes, null_file)
            null_beta, null_beta_torch = _build_beta(
                cell_lineages, node_idx, null_node_regime, null_regimes, device
            )
            null_regime_list.append(null_regimes)
            null_beta_list.append(null_beta)
            null_beta_list_torch.append(null_beta_torch)

        diverge_list.append(diverge_mat)
        share_list.append(share_mat)
        epochs_list.append(epochs)
        beta_list.append(beta)
        diverge_list_torch.append(diverge_torch)
        share_list_torch.append(share_torch)
        epochs_list_torch.append(epochs_torch)
        beta_list_torch.append(beta_torch)

    result = (
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
        library_list,
    )
    if null_regime_files is None:
        return result
    return result + (null_regime_list, null_beta_list, null_beta_list_torch)


def process_data_BM(tree_files, gene_files, library_files, device):
    """
    Precompute input data and store for optimization.

    Returns:
    tree_list: list of trees
    cells_list: list of cells
    df_list: list of expression data
    share_list_torch: list of share matrices (torch)
    library_list: list of library size dataframes
    """
    tree_list = []
    cells_list = []
    df_list = []
    share_list_torch = []
    library_list = []

    for tree_file, gene_file, library_file in zip(
        tree_files, gene_files, library_files
    ):
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

        # Read library sizes if provided, reorder cells as tree leaves
        if library_file:
            lib_df = pd.read_csv(library_file, sep="\t", index_col=0, header=None)
            lib_df.index = lib_df.index.astype(str)
            lib_df = lib_df.loc[cells]
            library_list.append(lib_df)

        # if no library sizes provided, use ones
        else:
            lib_df = pd.DataFrame(np.ones((len(cells), 1)), index=cells)
            library_list.append(lib_df)

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
                clade.name = "add_internal_node" + str(i)
                i = i + 1

        # Store paths from each node to root
        paths = {clade.name: tree.get_path(clade) for clade in clades}

        # Precompute node index mapping for all nodes and cells
        node_names = list(paths.keys())
        node_idx = {name: i for i, name in enumerate(node_names)}

        cell_paths = [set(a.name for a in paths[cell]) for cell in cells]
        cell_lineages = [[a.name for a in paths[cell]][::-1] for cell in cells]

        # Build MRCA index matrix for all cell pairs
        mrca_idx = np.zeros((len(cells), len(cells)), dtype=int)
        for i in range(len(cells)):
            for j in range(len(cells)):
                # Fast set intersection for paths
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
        share_mat = depth[mrca_idx]  # depth of each MRCA (i, j)
        share_torch = torch.tensor(share_mat, dtype=torch.float32, device=device)
        share_list_torch.append(share_torch)

    return tree_list, cells_list, df_list, share_list_torch, library_list
