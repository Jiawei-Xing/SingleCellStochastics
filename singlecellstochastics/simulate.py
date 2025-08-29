import numpy as np


# simulate null distribution for each gene
def simulate_null_each(trees, h0_params, N_sim, cells_list):
    """
    Simulate N_sim OU processes along all trees in parallel for a batch of genes.

    trees: list of trees
    h0_params: (batch_size, 1, all_param_dim) numpy array
    N_sim: number of simulations on every tree for each gene
    cells_list: list of cells for each tree
    Returns: (batch_size, N_sim, n_cells) numpy array
    """
    x_sim_list = []

    for i, tree in enumerate(trees):
        cells = cells_list[i]
        alpha = np.tile(
            h0_params[:, 0, 0][:, np.newaxis], (1, N_sim)
        )  # (batch_size, N_sim)
        sigma2 = np.tile(
            h0_params[:, 0, 1][:, np.newaxis], (1, N_sim)
        )  # (batch_size, N_sim)
        theta0 = np.tile(
            h0_params[:, 0, 2][:, np.newaxis], (1, N_sim)
        )  # (batch_size, N_sim)

        # simulate z_values for this tree by OU process
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10))  # (batch_size, N_sim)
        z_root = np.random.normal(loc=theta0, scale=std)  # (batch_size, N_sim)
        z_values = {tree.root.name: z_root}  # (batch_size, N_sim)

        for clade in tree.get_nonterminals(order="preorder"):  # depth first search
            parent_z = z_values[clade.name]
            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = np.array(t, dtype=np.float32)
                mean = theta0 + (parent_z - theta0) * np.exp(
                    -alpha * t
                )  # (batch_size, N_sim)
                var = (
                    sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
                )  # (batch_size, N_sim)
                var = np.maximum(var, 1e-10)  # (batch_size, N_sim)
                std = np.sqrt(var)  # (batch_size, N_sim)
                z_child = np.random.normal(loc=mean, scale=std)  # (batch_size, N_sim)
                z_values[child.name] = z_child  # (batch_size, N_sim)

        # combine z_values for all cells
        z_sim_matrix = np.stack(
            [z_values[cell] for cell in cells], axis=2
        )  # (batch_size, N_sim, n_cells)

        # poisson sampling
        lambda_sim = np.log1p(np.exp(z_sim_matrix))  # softplus equivalent
        x_sim = np.random.poisson(lambda_sim)  # (batch_size, N_sim, n_cells)
        x_sim_list.append(x_sim)  # List of (batch_size, N_sim, n_cells)

    return x_sim_list


# simulate null distribution for all genes
def simulate_null_all(trees, h0_params, N_sim_all, cells_list):
    """
    Sample ou params and simulate N_sim_all OU processes along all trees in parallel for all genes.

    h0_params: (gene_num, all_param_dim) numpy array
    N_sim_all: number of simulations on every tree for all genes
    cells_list: list of cells for each tree
    Returns: (N_sim_all, n_cells) numpy array
    """
    # sample ou params for each tree
    n_genes = h0_params.shape[0]
    idx1 = np.random.choice(n_genes, size=N_sim_all, replace=True)
    idx2 = np.random.choice(n_genes, size=N_sim_all, replace=True)
    idx3 = np.random.choice(n_genes, size=N_sim_all, replace=True)
    alpha = h0_params[idx1, 0]  # (N_sim_all,)
    sigma2 = h0_params[idx2, 1]  # (N_sim_all,)
    theta0 = h0_params[idx3, 2]  # (N_sim_all,)

    x_sim_list = []
    for i, tree in enumerate(trees):
        cells = cells_list[i]

        # simulate z_values for this tree
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10))  # (N_sim_all,)
        z_root = np.random.normal(loc=theta0, scale=std)  # (N_sim_all,)
        z_values = {tree.root.name: z_root}  # (N_sim_all,)

        for clade in tree.get_nonterminals(order="preorder"):  # depth first search
            parent_z = z_values[clade.name]
            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = np.array(t, dtype=np.float32)
                mean = theta0 + (parent_z - theta0) * np.exp(-alpha * t)  # (N_sim_all,)
                var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))  # (N_sim_all,)
                var = np.maximum(var, 1e-10)  # (N_sim_all,)
                std = np.sqrt(var)  # (N_sim_all,)
                z_child = np.random.normal(loc=mean, scale=std)  # (N_sim_all,)
                z_values[child.name] = z_child  # (N_sim_all,)

        # combine z_values for all cells
        z_sim_matrix = np.stack(
            [z_values[cell] for cell in cells], axis=1
        )  # (N_sim_all, n_cells)

        # poisson sampling
        lambda_sim = np.log1p(np.exp(z_sim_matrix))  # softplus equivalent
        x_sim = np.random.poisson(lambda_sim)  # (N_sim_all, n_cells)
        x_sim_list.append(x_sim)  # List of (N_sim_all, n_cells)

    return x_sim_list
