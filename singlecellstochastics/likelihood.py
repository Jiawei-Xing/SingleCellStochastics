"""Gaussian tree likelihoods for BM and OU latent expression models.

The likelihoods here operate on latent Gaussian expression values or their
variational expectations. Observation-model terms live in `elbo.py`; this module
only handles the tree-structured BM/OU Gaussian prior.
"""

import numpy as np
import torch
from scipy.special import logsumexp
import warnings

from .weights import theta_weight_W_numpy, theta_weight_W_torch
from .trace import TRACE, cov_diag, nan_inf_count


def _apply_pagel_lambda_to_cov(V, raw_lambda, mode, reference):
    """Mix V with diag(V) by a per-batch lambda.

    mode 0: lam=0 (star tree). mode 1: lam=1 (no-op if raw_lambda is None).
    mode 2: lam=sigmoid(raw_lambda).
    """
    if mode == 1 and raw_lambda is None:
        return V
    if mode == 0:
        lam = torch.zeros_like(reference)
    elif mode == 1:
        lam = torch.ones_like(reference)
    elif mode == 2:
        lam = torch.sigmoid(raw_lambda)
    else:
        raise ValueError(f"Invalid Pagel lambda mode: {mode}")
    lam = lam[..., None, None]
    diag_v = torch.diag_embed(torch.diagonal(V, dim1=-2, dim2=-1))
    return lam * V + (1.0 - lam) * diag_v


def ou_covariance_fixed_root_numpy(alpha, diverge, share):
    """OU covariance factor with root fixed at theta0."""
    return (1 / (2 * alpha)) * np.exp(-alpha * diverge) * (1 - np.exp(-2 * alpha * share))


def ou_covariance_root_prior_numpy(alpha, diverge):
    """OU covariance factor with stationary root prior N(theta0, sigma2/(2 alpha))."""
    return (1 / (2 * alpha)) * np.exp(-alpha * diverge)


def ou_covariance_fixed_root_torch(alpha, diverge, share):
    """OU covariance factor with root fixed at theta0."""
    return (1 / (2 * alpha)) * torch.exp(-alpha * diverge) * (1 - torch.exp(-2 * alpha * share))


def ou_covariance_root_prior_torch(alpha, diverge):
    """OU covariance factor with stationary root prior N(theta0, sigma2/(2 alpha))."""
    return (1 / (2 * alpha)) * torch.exp(-alpha * diverge)


# calculate negative log likelihood of OU
def ou_neg_log_lik_numpy(
    params, mode, expr_list, diverge_list, share_list, epochs_list, beta_list,
    root_mode="stationary"
):
    """
    Compute negative log likelihood of OU along tree adapted from EvoGeneX.
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
    alpha, sigma2 = params[:2]
    theta0 = params[2]
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

        # V: covariance matrix from trees (excluding variance sigma2).
        # root_mode="stationary": root ~ N(theta0, sigma2/(2*alpha))
        # root_mode="fixed":      root deterministically equals theta0
        if root_mode == "fixed":
            V = ou_covariance_fixed_root_numpy(alpha, diverge, share)
        elif root_mode == "stationary":
            V = ou_covariance_root_prior_numpy(alpha, diverge)
        else:
            raise ValueError(f"Invalid root_mode: {root_mode}")

        # diff = observed expression - expected values (n_cells, 1)
        if mode == 1:
            W = np.ones(n_cells)  # expected values at leaves are same as theta0 at root
            diff = expr - W * theta0
        elif mode == 2:
            W = theta_weight_W_numpy(alpha, epochs, beta)
            diff = expr - W @ params[-n_regimes:]

        try:
            log_det = np.linalg.slogdet(V)[1]  # log det V
            exp = (diff.T @ np.linalg.solve(V, diff)).item()  # diff @ V^-1 @ diff
        except np.linalg.LinAlgError:
            # Add regularization to prevent singular matrix
            print("warning: singular matrix")
            V_reg = V + 1e-6 * np.eye(n_cells)
            log_det = np.linalg.slogdet(V_reg)[1]  # log det V
            exp = (diff.T @ np.linalg.solve(V_reg, diff)).item()  # diff @ V^-1 @ diff

        loss = log_det + exp / sigma2 + n_cells * np.log(sigma2)  # -2 * log likelihood
        loss_list.append(loss/2)

    loss_list = np.array(loss_list)
    average_L = logsumexp(loss_list) - np.log(n_trees)
    return average_L


# calculate negative log likelihood of OU
def ou_neg_log_lik_numpy_kkt(
    params, mode, expr_list, diverge_list, share_list, epochs_list, beta_list,
    root_mode="stationary"
):
    """
    Compute negative log likelihood of OU along tree with KKT condition, adapted from EvoGeneX.
    Use expression data directly as the OU output.
    Assume the same OU parameters for all trees and average likelihoods with log-sum-exp trick.

    params: alpha
    mode: 1 for H0, 2 for H1
    expr: expression data (n_cells, 1)
    diverge_list: list of divergence matrices (n_cells, n_cells)
    share_list: list of share matrices (n_cells, n_cells)
    epochs_list: list of time intervals (n_cells, 1)
    beta_list: list of regime activation matrices (n_cells, n_regimes)
    V: covariance matrix from trees (n_cells, n_cells)
    W: theta weight (n_cells, n_regimes)
    """
    alpha = params[0]
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

        # V: covariance matrix from trees (excluding variance sigma2).
        # See ou_neg_log_lik_numpy for root_mode semantics.
        if root_mode == "fixed":
            V = ou_covariance_fixed_root_numpy(alpha, diverge, share)
        elif root_mode == "stationary":
            V = ou_covariance_root_prior_numpy(alpha, diverge)
        else:
            raise ValueError(f"Invalid root_mode: {root_mode}")

        # diff = observed expression - expected values (n_cells, 1)
        if mode == 1:
            W = np.ones((n_cells, 1))  # expected values at leaves are same as theta0 at root
        elif mode == 2:
            W = theta_weight_W_numpy(alpha, epochs, beta)
        
        # cholesky decomposition
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            # Add regularization to prevent singular matrix
            print("warning: singular matrix")
            V_reg = V + 1e-6 * np.eye(n_cells)
            L = np.linalg.cholesky(V_reg)

        # log_det = np.linalg.slogdet(V)[1]
        # cholesky: log|V| = 2 * sum(log diag(L))
        log_det = 2 * np.sum(np.log(np.diagonal(L)))

        # theta = (W.T @ V^-1 @ W)^-1 @ W.T @ V^-1 @ expr
        # theta = np.linalg.solve(W.T @ np.linalg.solve(V, W), W.T @ np.linalg.solve(V, expr))
        # cholesky: theta = (S.T @ S)^-1 @ S.T @ y
        S = np.linalg.solve(L, W)      # L^{-1} W
        y = np.linalg.solve(L, expr)   # L^{-1} expr
        theta = np.linalg.solve(S.T @ S, S.T @ y)
        
        # sigma2 = (expr - W @ theta)^T @ V^-1 @ (expr - W @ theta) / n_cells
        # sigma2 = (diff @ np.linalg.solve(V, diff)).item() / n_cells
        # cholesky: sigma2 = (y - S @ theta)^T @ (y - S @ theta) / n_cells
        r = y - S @ theta
        sigma2 = (r.T @ r).item() / n_cells

        loss = log_det + n_cells * np.log(sigma2)  # -2 * log likelihood
        loss_list.append(loss/2)

    loss_list = np.array(loss_list)
    average_L = logsumexp(loss_list) - np.log(n_trees)
    return average_L


# calculate expectation of negative OU log likelihood with torch
def ou_neg_log_lik_torch(
    params_batch, sigma2_q, mode, expr_batch, diverge, share, epochs, beta, device,
    pagel_lambda=None, pagel_lambda_mode=1, root_mode="stationary"
):
    """
    Same OU likelihood with torch. Used for ELBO with torch.

    params_batch: [log_alpha, other OU params]
    sigma2_q: (batch_size, N_sim, n_cells)
    expr_batch: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    Returns: (batch_size, N_sim) tensor of losses
    """
    batch_size, N_sim, n_cells = expr_batch.shape

    # Extract parameters (keep positive)
    alpha = torch.exp(params_batch[0][:, :, 0])  # (batch_size, N_sim)
    sigma2 = params_batch[1][:, :, 0]**2  # (batch_size, N_sim)
    thetas = params_batch[1][:, :, 1:]  # (batch_size, N_sim, n_regimes)

    # Compute V for all batches (broadcast alpha)
    alpha = alpha[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    sigma2 = sigma2[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    diverge = diverge[None, None, :, :]  # (1, 1, n_cells, n_cells)
    if root_mode == "fixed":
        share_b = share[None, None, :, :]
        V = ou_covariance_fixed_root_torch(alpha, diverge, share_b)
    elif root_mode == "stationary":
        V = ou_covariance_root_prior_torch(alpha, diverge)
    else:
        raise ValueError(f"Invalid root_mode: {root_mode}")
    V = _apply_pagel_lambda_to_cov(
        V,
        pagel_lambda,
        pagel_lambda_mode,
        alpha.squeeze(-1).squeeze(-1),
    )
    
    # Compute W and diff
    if mode == 1:
        W = torch.ones(
            (batch_size, N_sim, n_cells), dtype=alpha.dtype, device=device
        )
        diff = expr_batch - W * thetas[:, :, 0:1]  # (batch_size, N_sim, n_cells)
    elif mode == 2:
        W = theta_weight_W_torch(
            alpha.squeeze(-1).squeeze(-1), epochs, beta
        )  # (batch_size, N_sim, n_cells, n_regimes)
        diff = expr_batch - torch.matmul(
            W, thetas.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, N_sim, n_cells)

    # cholesky decomposition for det and inv of matrix
    try:
        L = torch.linalg.cholesky(V)  # (batch_size, N_sim, n_cells, n_cells)
    except RuntimeError:
        # Add regularization to prevent singular matrix
        print("warning: singular matrix")
        L = torch.linalg.cholesky(
            V + 1e-6 * torch.eye(n_cells, device=device, dtype=alpha.dtype)
        )

    # log_det = torch.linalg.slogdet(V_reg)[1]
    # cholesky: log|V| = 2 * sum(log diag(L))
    log_det = 2 * torch.sum(
        torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
    )  # (batch_size, N_sim)
    
    d = diff.unsqueeze(-1)  # (batch, N_sim, n_cells, 1)

    # exp = (d.transpose(-2, -1) @ torch.linalg.solve(V, d)).squeeze(-1).squeeze(-1)  # (batch, N_sim)
    # cholesky: d^T V^{-1} d = ||L^{-1} d||^2
    y = torch.linalg.solve_triangular(
        L, d, upper=False
    )  # y = L^{-1} d (batch, N_sim, n_cells, 1)
    exp = (y.squeeze(-1) ** 2).sum(dim=-1)  # sum(y^2) = y^T y = ||y||^2 (batch, N_sim)

    # tr_term = torch.sum(sigma2_q * torch.diagonal(torch.linalg.inv(V_reg), dim1=-2, dim2=-1), dim=-1)
    # Cholesky: trace(V^{-1} Sigma) = ||diag(L^{-1} Sigma)||_F^2.
    S_half = torch.diag_embed(
        torch.sqrt(sigma2_q)
    )  # Sigma^{1/2} = diag(sigma_i), shape (..., n, n)
    Y = torch.linalg.solve_triangular(L, S_half, upper=False)  # Y = L^{-1} Sigma^{1/2}
    tr_term = (Y**2).sum(dim=(-2, -1))  # ||Y||_F^2 = tr(Y^T Y) (batch, N_sim)

    loss = 0.5 * (
        log_det
        + n_cells * torch.log(sigma2).squeeze(-1).squeeze(-1)
        + (exp + tr_term) / sigma2.squeeze(-1).squeeze(-1)
    )

    return loss  # (batch_size, N_sim)


def psd_safe_cholesky(V, base_jitter=1e-6, max_tries=7):
    # 1) Symmetrize (critical!)
    V = 0.5 * (V + V.transpose(-1, -2))

    # 2) Replace NaN/Inf if any (optional but practical)
    if not torch.isfinite(V).all():
        V = torch.where(torch.isfinite(V), V, torch.zeros_like(V))

    n = V.shape[-1]
    eye = torch.eye(n, dtype=V.dtype, device=V.device)
    jitter = base_jitter

    # 3) Try with exponentially increasing jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(V + jitter * eye)
        except RuntimeError:
            jitter *= 10.0  # 1e-6 -> 1e-5 -> 1e-4 -> ... as needed

    # 4) PSD projection fallback (clip negative eigenvalues), then Cholesky
    evals, evecs = torch.linalg.eigh(V)  # V is symmetric now
    # Use a small floor relative to dtype
    floor = 1e-10 if V.dtype == torch.float64 else 1e-8
    evals = torch.clamp(evals, min=floor)
    V_psd = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return torch.linalg.cholesky(V_psd + jitter * eye)


def cholesky_with_fallback(V, name):
    """Cholesky with two-stage regularization fallback.

    Returns (L, jitter_used) where jitter_used is 0.0 (no jitter), 1e-6
    (single-jitter pass), or -1.0 (psd_safe_cholesky fallback path).
    """
    try:
        return torch.linalg.cholesky(V), 0.0
    except RuntimeError:
        warnings.warn(
            f"Singular matrix encountered in {name} likelihood; applying regularization."
        )
        n_cells = V.shape[-1]
        eye = 1e-6 * torch.eye(n_cells, device=V.device, dtype=V.dtype)
        try:
            return torch.linalg.cholesky(V + eye), 1e-6
        except RuntimeError:
            return psd_safe_cholesky(V), -1.0


# calculate expectation of negative OU log likelihood with torch
def ou_neg_log_lik_torch_kkt(
    param_batch, sigma2_q, mode, expr_batch, diverge, share, epochs, beta, device,
    pagel_lambda=None, pagel_lambda_mode=1, root_mode="stationary"
):
    """
    Same OU likelihood with torch. Used for ELBO with torch.

    params_batch: [log_alpha, other OU params]
    sigma2_q: (batch_size, N_sim, n_cells)
    expr_batch: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    Returns: (batch_size, N_sim) tensor of losses
    """
    batch_size, N_sim, n_cells = expr_batch.shape

    # Extract parameters (keep positive)
    alpha = torch.exp(param_batch[0][:, :, 0])  # (batch_size, N_sim)

    # Compute V for all batches (broadcast alpha)
    alpha = alpha[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    if root_mode == "fixed":
        share_b = share[None, None, :, :]
        V = ou_covariance_fixed_root_torch(alpha, diverge[None, None, :, :], share_b)
    elif root_mode == "stationary":
        V = ou_covariance_root_prior_torch(alpha, diverge[None, None, :, :])
    else:
        raise ValueError(f"Invalid root_mode: {root_mode}")
    V = _apply_pagel_lambda_to_cov(
        V,
        pagel_lambda,
        pagel_lambda_mode,
        alpha.squeeze(-1).squeeze(-1),
    )

    # Compute W and diff
    if mode == 1:
        W = torch.ones(
            (batch_size, N_sim, n_cells, 1), dtype=alpha.dtype, device=device
        )
    elif mode == 2:
        W = theta_weight_W_torch(
            alpha.squeeze(-1).squeeze(-1), epochs, beta
        )  # (batch_size, N_sim, n_cells, n_regimes)

    if TRACE.enabled:
        TRACE.write("alpha", alpha.reshape(-1)[0])
        cond, emin, nneg = cov_diag(V)
        TRACE.write("V_cond", cond)
        TRACE.write("V_min_eig", emin)
        TRACE.write("V_n_neg_diag", nneg)
        nan_v, inf_v = nan_inf_count(V)
        TRACE.write("V_n_nan", nan_v)
        TRACE.write("V_n_inf", inf_v)
        TRACE.write("V_max_abs", float(V.abs().max().detach().cpu()))

    # cholesky decomposition for det and inv of matrix
    L, chol_jitter_used = cholesky_with_fallback(V, "OU")

    if TRACE.enabled:
        TRACE.write("chol_jitter", chol_jitter_used)
        nan_l, inf_l = nan_inf_count(L)
        TRACE.write("L_n_nan", nan_l)
        TRACE.write("L_n_inf", inf_l)
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
        TRACE.write("L_min_diag", float(diag_L.min().detach().cpu()))

    # log_det = torch.linalg.slogdet(V_reg)[1]
    # cholesky: log|V| = 2 * sum(log diag(L))
    log_det = 2 * torch.sum(
        torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
    )  # (batch_size, N_sim)
    if TRACE.enabled:
        TRACE.write("logdet_V", float(log_det.reshape(-1)[0].detach().cpu()))

    # theta = (W.T @ V^-1 @ W)^-1 @ W.T @ V^-1 @ expr
    # cholesky: theta = (S.T @ S)^-1 @ S.T @ y, with S = L^{-1} W, y = L^{-1} expr.
    # Use solve_triangular here (not L_inv @ ...) for tighter numerical stability
    # of the GLS solve that feeds lstsq.
    S = torch.linalg.solve_triangular(L, W, upper=False)
    y = torch.linalg.solve_triangular(L, expr_batch.unsqueeze(-1), upper=False)
    # Use magma backend on CUDA to avoid cusolver SVD failure on large matrices.
    if S.is_cuda:
        prev_lib = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library("magma")
        theta = torch.linalg.lstsq(S, y).solution  # (batch, N_sim, n_regimes, 1)
        torch.backends.cuda.preferred_linalg_library(prev_lib)
    else:
        theta = torch.linalg.lstsq(S, y).solution  # (batch, N_sim, n_regimes, 1)
    if TRACE.enabled:
        nan_t, inf_t = nan_inf_count(theta)
        TRACE.write("theta_n_nan", nan_t)
        TRACE.write("theta_n_inf", inf_t)
        TRACE.write("theta_kkt", theta.reshape(-1).detach().cpu().numpy())

    # sigma2 = (y - S @ theta)^T @ (y - S @ theta) / n_cells
    r = y - S @ theta
    sigma2 = r.square().sum(dim=(-2, -1)) / n_cells # (batch_size, N_sim)

    # trace(V^{-1} Sigma) = sum_j sigma2_q[j] * diag(V^{-1})_j,
    # with diag(V^{-1})_i = sum_k (L^{-1})_{ki}^2. Materialize L^{-1} just for this.
    I_eye = torch.eye(n_cells, dtype=V.dtype, device=device).expand(batch_size, N_sim, -1, -1)
    L_inv = torch.linalg.solve_triangular(L, I_eye, upper=False)
    diag_V_inv = (L_inv ** 2).sum(dim=-2)  # (batch_size, N_sim, n_cells)
    tr_term = (diag_V_inv * sigma2_q).sum(dim=-1)  # (batch_size, N_sim)
    sigma2 += tr_term / n_cells

    if TRACE.enabled:
        TRACE.write("sigma2_kkt", float(sigma2.reshape(-1)[0].detach().cpu()))
        TRACE.write("ou_loss_n_nan", int(torch.isnan(0.5 * (log_det + n_cells * torch.log(sigma2))).sum().detach().cpu()))

    loss = 0.5 * (log_det + n_cells * torch.log(sigma2))  # -log likelihood

    sigma = sigma2.sqrt()  # (batch_size, N_sim)
    theta = theta.squeeze(-1)  # (batch_size, N_sim, n_regimes)

    return loss, sigma, theta  # (batch_size, N_sim)


# calculate expectation of negative BM log likelihood with torch
def bm_neg_log_lik_torch_kkt(
    pagel_lambda, sigma2_q, expr_batch, share, device
):
    """
    Brownian motion likelihood with torch. Used for ELBO with torch.

    Profiles the root mean by GLS and sigma2 by ML, equivalent to a flat prior 
    on the root state. 

    pagel_lambda: (batch_size,) in [0, 1], 0 for star tree, 1 for original tree
    sigma2_q: (batch_size, n_cells) tensor of q variances
    expr_batch: (batch_size, n_cells) tensor of expression data
    share: (n_cells, n_cells) tensor of shared branch lengths

    Returns:
        loss: (batch_size,) tensor of -log likelihood (without 2*pi constant)
        root_mean_gls: (batch_size,) GLS estimate of the root mean
        sigma: (batch_size,) ML estimate of sigma (sqrt of profile sigma2)
    """
    batch_size, n_cells = expr_batch.shape
    pagel_lambda = pagel_lambda[:, None, None]  # (batch_size, 1, 1)

    # V: covariance matrix from trees (excluding variance sigma2)
    diag_share = torch.diag_embed(torch.diagonal(share, dim1=-2, dim2=-1))
    V = (pagel_lambda * share + (1 - pagel_lambda) * diag_share)  # (batch_size, n_cells, n_cells)

    # cholesky decomposition for det and inv of matrix
    L, _ = cholesky_with_fallback(V, "BM")

    # 1. Setup vectors
    Y_vec = expr_batch.unsqueeze(-1)  # (batch_size, n_cells, 1)
    ones = torch.ones_like(Y_vec)

    # 2. Solve for L^{-1} Y and L^{-1} 1.
    # Use solve_triangular here (not L_inv @ ...) for tighter numerical stability
    # of the GLS pieces.
    L_inv_Y = torch.linalg.solve_triangular(L, Y_vec, upper=False)
    L_inv_ones = torch.linalg.solve_triangular(L, ones, upper=False)

    # 3. Calculate 1^T V^{-1} 1 and 1^T V^{-1} Y
    one_V_inv_one = (L_inv_ones.squeeze(-1) ** 2).sum(dim=-1)  # (batch_size,)
    one_V_inv_Y = (L_inv_ones.squeeze(-1) * L_inv_Y.squeeze(-1)).sum(dim=-1)  # (batch_size,)

    # 4. Analytically compute the exact GLS mean
    root_mean_gls = one_V_inv_Y / one_V_inv_one  # (batch_size,)

    # 5. Compute the proper residual d^T V^{-1} d using linearity:
    # L^{-1}(Y - mu*1) = L^{-1}Y - mu * L^{-1}1
    resid = L_inv_Y - root_mean_gls.unsqueeze(-1).unsqueeze(-1) * L_inv_ones
    exp = (resid.squeeze(-1) ** 2).sum(dim=-1)  # (batch_size,)

    # log_det = torch.linalg.slogdet(V_reg)[1]
    # cholesky: log|V| = 2 * sum(log diag(L))
    log_det = 2 * torch.sum(
        torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
    )  # (batch_size,)

    # tr_term = trace(V^{-1} Sigma) = sum_j sigma2_q[j] * diag(V^{-1})_j,
    # with diag(V^{-1})_i = sum_k (L^{-1})_{ki}^2. Materialize L^{-1} just for this.
    I = torch.eye(n_cells, dtype=V.dtype, device=device).expand(batch_size, -1, -1)
    L_inv = torch.linalg.solve_triangular(L, I, upper=False)
    diag_V_inv = (L_inv ** 2).sum(dim=-2)  # (batch_size, n_cells)
    tr_term = (diag_V_inv * sigma2_q).sum(dim=-1)  # (batch_size,)

    sigma2 = (exp + tr_term) / n_cells  # (batch_size,)
    sigma2 = sigma2.clamp_min(torch.finfo(sigma2.dtype).tiny)
    sigma = sigma2.sqrt()  # (batch_size,)

    loss = 0.5 * (log_det + n_cells * torch.log(sigma2))  # -log likelihood (w/o constant)
    return loss, root_mean_gls, sigma  # (batch_size,)
