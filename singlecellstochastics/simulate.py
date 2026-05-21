"""Null-simulation helpers for empirical calibration."""

import numpy as np

from .likelihood import ou_covariance_fixed_root_numpy, ou_covariance_root_prior_numpy
from .weights import theta_weight_W_numpy


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sample_counts(z_values, log_r, lib, nb):
    mean = np.asarray(lib)[None, None, :] * _softplus(z_values)
    if not nb:
        return np.random.poisson(mean)

    r = np.exp(log_r)[:, :, None]
    p = r / (r + mean)
    return np.random.negative_binomial(r, p)


def _simulate_ou_latent(tree, cells, log_alpha, sigma, theta0, root_mode="stationary"):
    alpha = np.exp(log_alpha)
    sigma2 = np.square(sigma)

    if root_mode == "fixed":
        z_root = np.array(theta0, copy=True)
    elif root_mode == "stationary":
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10))
        z_root = np.random.normal(loc=theta0, scale=std)
    else:
        raise ValueError(f"Invalid root_mode: {root_mode}")
    z_values = {tree.root.name: z_root}

    for clade in tree.get_nonterminals(order="preorder"):
        parent_z = z_values[clade.name]
        for child in clade.clades:
            t = child.branch_length if child.branch_length is not None else 0.0
            t = np.array(t, dtype=np.float64)
            mean = theta0 + (parent_z - theta0) * np.exp(-alpha * t)
            var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
            std = np.sqrt(np.maximum(var, 1e-10))
            z_values[child.name] = np.random.normal(loc=mean, scale=std)

    return np.stack([z_values[cell] for cell in cells], axis=2)


def _ou_covariance(alpha, sigma, diverge, share, root_mode):
    if root_mode == "fixed":
        v = ou_covariance_fixed_root_numpy(alpha, diverge, share)
    elif root_mode == "stationary":
        v = ou_covariance_root_prior_numpy(alpha, diverge)
    else:
        raise ValueError(f"Invalid root_mode: {root_mode}")
    cov = np.square(sigma) * v
    return (cov + cov.T) / 2.0


def _sample_multivariate_normal(mean, cov):
    jitter = 1e-8
    eye = np.eye(cov.shape[0], dtype=np.float64)
    for _ in range(6):
        try:
            return np.random.multivariate_normal(mean, cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.random.multivariate_normal(mean, cov + jitter * eye, check_valid="ignore")


def _simulate_ou_latent_design(
    log_alpha,
    sigma,
    theta,
    diverge,
    share,
    epochs,
    beta,
    root_mode="stationary",
):
    """Simulate latent OU tip values from an explicit regime design.

    This uses the same Gaussian distribution as the OU likelihood: mean
    ``W(alpha, beta) @ theta`` and covariance ``sigma^2 V(alpha, tree)``.
    Shapes: log_alpha/sigma are ``(batch, n_sim)`` and theta is
    ``(batch, n_sim, n_regimes)``.
    """
    batch_size, n_sim = log_alpha.shape
    n_cells = diverge.shape[0]
    z = np.empty((batch_size, n_sim, n_cells), dtype=np.float64)

    for i in range(batch_size):
        for j in range(n_sim):
            alpha = float(np.exp(log_alpha[i, j]))
            W = theta_weight_W_numpy(alpha, epochs, beta)
            mean = W @ theta[i, j, :]
            cov = _ou_covariance(
                alpha, float(abs(sigma[i, j])), diverge, share, root_mode
            )
            z[i, j, :] = _sample_multivariate_normal(mean, cov)

    return z


# simulate null distribution for each gene
def simulate_null_each(
    trees,
    h0_params,
    N_sim,
    cells_list,
    library_list=None,
    nb=True,
    root_mode="stationary",
    diverge_list=None,
    share_list=None,
    epochs_list=None,
    beta_list=None,
):
    """
    Simulate N_sim OU processes along all trees in parallel for a batch of genes.

    trees: list of trees
    h0_params: (batch_size, 1, all_param_dim) numpy array storing
        [log_r, log_alpha, sigma, theta0, ...].
    N_sim: number of simulations on every tree for each gene
    cells_list: list of cells for each tree
    Returns: (batch_size, N_sim, n_cells) numpy array
    """
    x_sim_list = []
    if library_list is None:
        library_list = [None] * len(trees)

    for i, tree in enumerate(trees):
        cells = cells_list[i]
        log_r = np.tile(h0_params[:, 0, 0][:, None], (1, N_sim))
        log_alpha = np.tile(h0_params[:, 0, 1][:, None], (1, N_sim))
        sigma = np.tile(np.abs(h0_params[:, 0, 2])[:, None], (1, N_sim))
        theta = np.tile(h0_params[:, 0, 3:][:, None, :], (1, N_sim, 1))
        lib = (
            np.ones(len(cells), dtype=np.float64)
            if library_list[i] is None
            else np.asarray(library_list[i].values.squeeze(), dtype=np.float64)
        )

        if theta.shape[2] == 1 and beta_list is None:
            z_sim_matrix = _simulate_ou_latent(
                tree, cells, log_alpha, sigma, theta[:, :, 0], root_mode
            )
        else:
            if any(
                x is None for x in (diverge_list, share_list, epochs_list, beta_list)
            ):
                raise ValueError(
                    "Multi-theta null simulation requires tree design matrices."
                )
            z_sim_matrix = _simulate_ou_latent_design(
                log_alpha,
                sigma,
                theta,
                diverge_list[i],
                share_list[i],
                epochs_list[i],
                beta_list[i],
                root_mode,
            )
        x_sim_list.append(_sample_counts(z_sim_matrix, log_r, lib, nb))

    return x_sim_list


# simulate null distribution for all genes
def simulate_null_all(
    trees,
    h0_params,
    N_sim_all,
    cells_list,
    library_list=None,
    nb=True,
    root_mode="stationary",
    diverge_list=None,
    share_list=None,
    epochs_list=None,
    beta_list=None,
):
    """
    Sample ou params and simulate N_sim_all OU processes along all trees in
    parallel for all genes.

    h0_params: (gene_num, all_param_dim) numpy array storing
        [log_r, log_alpha, sigma, theta0, ...].
    N_sim_all: number of simulations on every tree for all genes
    cells_list: list of cells for each tree
    Returns: (N_sim_all, n_cells) numpy array
    """
    if library_list is None:
        library_list = [None] * len(trees)

    n_genes = h0_params.shape[0]
    idx = np.random.choice(n_genes, size=N_sim_all, replace=True)
    log_r = h0_params[idx, 0][:, None]
    log_alpha = h0_params[idx, 1][:, None]
    sigma = np.abs(h0_params[idx, 2])[:, None]
    theta = h0_params[idx, 3:][:, None, :]

    x_sim_list = []
    for i, tree in enumerate(trees):
        cells = cells_list[i]
        lib = (
            np.ones(len(cells), dtype=np.float64)
            if library_list[i] is None
            else np.asarray(library_list[i].values.squeeze(), dtype=np.float64)
        )
        if theta.shape[2] == 1 and beta_list is None:
            z_sim_matrix = _simulate_ou_latent(
                tree, cells, log_alpha, sigma, theta[:, :, 0], root_mode
            )
        else:
            if any(
                x is None for x in (diverge_list, share_list, epochs_list, beta_list)
            ):
                raise ValueError(
                    "Multi-theta null simulation requires tree design matrices."
                )
            z_sim_matrix = _simulate_ou_latent_design(
                log_alpha,
                sigma,
                theta,
                diverge_list[i],
                share_list[i],
                epochs_list[i],
                beta_list[i],
                root_mode,
            )
        x_sim_list.append(_sample_counts(z_sim_matrix, log_r, lib, nb)[:, 0, :])

    return x_sim_list
