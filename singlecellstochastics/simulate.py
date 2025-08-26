import numpy as np


# simulate null distribution for each gene TODO: sim for each tree separately
def simulate_null_each(trees, h0_params, N_sim, cells):
    """
    Simulate N_sim OU processes along all trees in parallel for a batch of genes.

    h0_params: (batch_size, 1, all_param_dim) numpy array
    Returns: (batch_size, N_sim, n_cells) numpy array
    """
    n_trees = len(trees)
    sims_per_tree = N_sim // n_trees
    remainder = N_sim % n_trees

    x_sim_list = []
    for i, tree in enumerate(trees):
        # get number of simulations for this tree
        n_sims = sims_per_tree + (1 if i < remainder else 0)
        alpha = np.tile(
            h0_params[:, 0, 2 * len(cells)][:, np.newaxis], (1, n_sims)
        )  # (batch_size, n_sims)
        sigma2 = np.tile(
            h0_params[:, 0, 2 * len(cells) + 1][:, np.newaxis], (1, n_sims)
        )  # (batch_size, n_sims)
        theta0 = np.tile(
            h0_params[:, 0, 2 * len(cells) + 2][:, np.newaxis], (1, n_sims)
        )  # (batch_size, n_sims)

        # simulate z_values for this tree by OU process
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10))  # (batch_size, n_sims)
        z_root = np.random.normal(loc=theta0, scale=std)  # (batch_size, n_sims)
        z_values = {tree.root.name: z_root}  # (batch_size, n_sims)

        for clade in tree.get_nonterminals(order="preorder"):  # depth first search
            parent_z = z_values[clade.name]
            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = np.array(t, dtype=np.float32)
                mean = theta0 + (parent_z - theta0) * np.exp(
                    -alpha * t
                )  # (batch_size, n_sims)
                var = (
                    sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
                )  # (batch_size, n_sims)
                var = np.maximum(var, 1e-10)  # (batch_size, n_sims)
                std = np.sqrt(var)  # (batch_size, n_sims)
                z_child = np.random.normal(loc=mean, scale=std)  # (batch_size, n_sims)
                z_values[child.name] = z_child  # (batch_size, n_sims)

        # combine z_values for all cells
        z_sim_matrix = np.stack(
            [z_values[cell] for cell in cells], axis=2
        )  # (batch_size, n_sim, n_cells)

        # poisson sampling
        lambda_sim = np.log1p(np.exp(z_sim_matrix))  # softplus equivalent
        x_sim = np.random.poisson(lambda_sim)  # (batch_size, n_sims, n_cells)
        x_sim_list.append(x_sim)  # (batch_size, n_sims, n_cells)

    # combine x_sim for all trees
    x_sim_matrix = np.concatenate(x_sim_list, axis=1)  # (batch_size, N_sims, n_cells)
    return x_sim_matrix


# simulate null distribution for all genes TODO: sim for each tree separately
def simulate_null_all(trees, h0_params, N_sim_all, cells):
    """
    Sample ou params and simulate N_sim_all OU processes along all trees in parallel for all genes.

    h0_params: (gene_num, all_param_dim) numpy array
    Returns: (N_sim_all, n_cells) numpy array
    """
    n_trees = len(trees)
    sims_per_tree = N_sim_all // n_trees
    remainder = N_sim_all % n_trees

    # sample ou params for each tree
    n_genes = h0_params.shape[0]
    idx1 = np.random.choice(n_genes, size=sims_per_tree + 1, replace=True)
    idx2 = np.random.choice(n_genes, size=sims_per_tree + 1, replace=True)
    idx3 = np.random.choice(n_genes, size=sims_per_tree + 1, replace=True)
    alpha_sample = h0_params[idx1, 2 * len(cells)]  # (sims_per_tree+1,)
    sigma2_sample = h0_params[idx2, 2 * len(cells) + 1]  # (sims_per_tree+1,)
    theta0_sample = h0_params[idx3, 2 * len(cells) + 2]  # (sims_per_tree+1,)

    x_sim_list = []
    start_idx = 0
    for i, tree in enumerate(trees):
        # get number of simulations for this tree
        n_sims = sims_per_tree + (1 if i < remainder else 0)
        alpha = alpha_sample[start_idx : start_idx + n_sims]  # (n_sims,)
        sigma2 = sigma2_sample[start_idx : start_idx + n_sims]  # (n_sims,)
        theta0 = theta0_sample[start_idx : start_idx + n_sims]  # (n_sims,)
        start_idx += n_sims

        # simulate z_values for this tree
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10))  # (n_sims,)
        z_root = np.random.normal(loc=theta0, scale=std)  # (n_sims,)
        z_values = {tree.root.name: z_root}  # (n_sims,)

        for clade in tree.get_nonterminals(order="preorder"):  # depth first search
            parent_z = z_values[clade.name]
            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = np.array(t, dtype=np.float32)
                mean = theta0 + (parent_z - theta0) * np.exp(-alpha * t)  # (n_sims,)
                var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))  # (n_sims,)
                var = np.maximum(var, 1e-10)  # (n_sims,)
                std = np.sqrt(var)  # (n_sims,)
                z_child = np.random.normal(loc=mean, scale=std)  # (n_sims,)
                z_values[child.name] = z_child  # (n_sims,)

        # combine z_values for all cells
        z_sim_matrix = np.stack(
            [z_values[cell] for cell in cells], axis=1
        )  # (n_sim, n_cells)

        # poisson sampling
        lambda_sim = np.log1p(np.exp(z_sim_matrix))  # softplus equivalent
        x_sim = np.random.poisson(lambda_sim)  # (n_sim, n_cells)
        x_sim_list.append(x_sim)  # (n_sim, n_cells)

    # combine x_sim for all trees
    x_sim_matrix = np.concatenate(x_sim_list, axis=0)  # (N_sim_all, n_cells)
    return x_sim_matrix
