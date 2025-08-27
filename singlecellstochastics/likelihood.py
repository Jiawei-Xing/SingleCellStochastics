import numpy as np
import torch
from scipy.special import logsumexp

from .weights import theta_weight_W_numpy, theta_weight_W_torch


# calculate negative log likelihood of OU
def ou_neg_log_lik_numpy(
    params, mode, expr_list, diverge_list, share_list, epochs_list, beta_list
):
    """
    Compute two times negative log likelihood of OU along tree adapted from EvoGeneX.
    Use expression data directly as the OU output.
    Assume the same OU parameters for all trees and average likelihoods with log-sum-exp trick.

    params: OU parameters (alpha, sigma2, theta0, theta1, ..., theta_n)
    mode: 1 for H0, 2 for H1
    expr: expression data (n_cells, 1)
    diverge_list: list of divergence matrices (n_cells, n_cells)
    share_list: list of share matrices (n_cells, n_cells)
    epochs_list: list of time intervals (n_cells, 1)
    beta_list: list of regime activation matrices (n_cells, n_regimes)
    V: covariance matrix from trees (n_cells, n_cells)
    W: theta weight (n_cells, n_regimes)
    """
    alpha, sigma2, theta0 = params[:3]
    n_trees = len(diverge_list)

    # share the same OU parameters for all trees
    loss_list = []
    for i in range(n_trees):
        expr = expr_list[i]
        n_cells = len(expr)
        diverge = diverge_list[i]
        share = share_list[i]
        epochs = epochs_list[i]
        beta = beta_list[i]
        n_regimes = beta[0].shape[1]

        # V: covariance matrix from trees (excluding variance sigma2)
        V = (
            (1 / (2 * alpha))
            * np.exp(-alpha * diverge)
            * (1 - np.exp(-2 * alpha * share))
        )

        # diff = observed expression - expected values (n_cells, 1)
        if mode == 1:
            W = np.ones(n_cells)  # expected values at leaves are same as theta0 at root
            diff = expr - W * theta0
        elif mode == 2:
            W = theta_weight_W_numpy(alpha, epochs, beta)
            diff = expr - W @ params[-n_regimes:]

        # log(det V) + (diff @ V^-1 @ diff) / sigma2 + n_cells * log(sigma2)
        log_det = np.linalg.slogdet(V)[1]  # log det V
        exp = (diff @ np.linalg.solve(V, diff)).item()  # diff @ V^-1 @ diff
        loss = log_det + exp / sigma2 + n_cells * np.log(sigma2)  # -2 * log likelihood
        loss_list.append(loss)

    loss_list = np.array(loss_list)
    average_L = logsumexp(loss_list) - np.log(n_trees)
    return average_L


# calculate expectation of negative OU log likelihood with torch
def ou_neg_log_lik_torch(
    params_batch, sigma2_q, mode, expr_batch, diverge, share, epochs, beta, device="cpu"
):
    """
    Same OU likelihood with torch. Used for ELBO with torch.

    params_batch: (batch_size, N_sim, ou_param_dim)
    sigma2_q: (batch_size, N_sim, n_cells)
    expr_batch: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    Returns: (batch_size, N_sim) tensor of losses
    """
    batch_size, N_sim, n_cells = expr_batch.shape

    # Extract parameters
    alpha = params_batch[:, :, 0]  # (batch_size, N_sim)
    sigma2 = params_batch[:, :, 1]  # (batch_size, N_sim)
    thetas = params_batch[:, :, 2:]  # (batch_size, N_sim, n_regimes)

    # Compute V for all batches (broadcast alpha)
    alpha = alpha[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    sigma2 = sigma2[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    diverge = diverge[None, None, :, :]  # (1, 1, n_cells, n_cells)
    share = share[None, None, :, :]  # (1, 1, n_cells, n_cells)
    V = (
        (1 / (2 * alpha))
        * torch.exp(-alpha * diverge)
        * (1 - torch.exp(-2 * alpha * share))
    )  # (batch_size, N_sim, n_cells, n_cells)

    # Compute W and diff
    if mode == 1:
        W = torch.ones((batch_size, N_sim, n_cells), dtype=torch.float32, device=device)
        diff = expr_batch - W * thetas[:, :, 0:1]  # (batch_size, N_sim, n_cells)
    elif mode == 2:
        W = theta_weight_W_torch(
            alpha.squeeze(-1).squeeze(-1), epochs, beta
        )  # (batch_size, N_sim, n_cells, n_regimes)
        diff = expr_batch - torch.matmul(W, thetas.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, N_sim, n_cells)

    L = torch.linalg.cholesky(V)  # (batch_size, N_sim, n_cells, n_cells)
    # log|V| = 2 * sum(log diag(L))
    log_det = 2 * torch.sum(
        torch.log(torch.diagonal(L, dim1=2, dim2=3)), dim=2
    )  # (batch_size, N_sim)

    # d^T V^{-1} d = ||L^{-1} d||^2
    d = diff.unsqueeze(-1)  # (batch, N_sim, n_cells, 1)
    y = torch.linalg.solve_triangular(
        L, d, upper=False
    )  # y = L^{-1} d (batch, N_sim, n_cells, 1)
    exp = (y.squeeze(-1) ** 2).sum(dim=-1)  # sum(y^2) = y^T y = ||y||^2 (batch, N_sim)

    # trace(V^{-1} Σ) = ||diag(L^{-1} Σ)||_F^2
    S_half = torch.diag_embed(
        torch.sqrt(sigma2_q)
    )  # Σ^{1/2} = diag(σ_i), shape (..., n, n)
    Y = torch.linalg.solve_triangular(L, S_half, upper=False)  # Y = L^{-1} Σ^{1/2}
    tr_term = (Y**2).sum(dim=(-2, -1))  # ||Y||_F^2 = tr(C^{-1} Σ)

    loss = (
        log_det
        + n_cells * torch.log(sigma2).squeeze(-1).squeeze(-1)
        + (exp + tr_term) / sigma2.squeeze(-1).squeeze(-1)
    )
    return loss  # (batch_size, N_sim)
