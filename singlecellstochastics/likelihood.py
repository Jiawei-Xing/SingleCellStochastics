import numpy as np
import torch
from scipy.special import logsumexp

from .weights import theta_weight_W_numpy, theta_weight_W_torch


# calculate negative log likelihood of OU
def ou_neg_log_lik_numpy(
    params, mode, expr_list, diverge_list, share_list, epochs_list, beta_list
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
    params, mode, expr_list, diverge_list, share_list, epochs_list, beta_list
):
    """
    Compute negative log likelihood of OU along tree with KKT conditionadapted from EvoGeneX.
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
    #alpha = np.logaddexp(0, param) # softplus to keep positive
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

        # V: covariance matrix from trees (excluding variance sigma2)
        V = (
            (1 / (2 * alpha))
            * np.exp(-alpha * diverge)
            * (1 - np.exp(-2 * alpha * share))
        )
        
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
    params_batch, sigma2_q, mode, expr_batch, diverge, share, epochs, beta, device
):
    """
    Same OU likelihood with torch. Used for ELBO with torch.

    params_batch: [alpha, other OU params]
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
    share = share[None, None, :, :]  # (1, 1, n_cells, n_cells)
    V = (
        (1 / (2 * alpha))
        * torch.exp(-alpha * diverge)
        * (1 - torch.exp(-2 * alpha * share))
    )  # (batch_size, N_sim, n_cells, n_cells)
    
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
        torch.log(torch.diagonal(L, dim1=2, dim2=3)), dim=2
    )  # (batch_size, N_sim)
    
    d = diff.unsqueeze(-1)  # (batch, N_sim, n_cells, 1)

    # exp = (d.transpose(-2, -1) @ torch.linalg.solve(V, d)).squeeze(-1).squeeze(-1)  # (batch, N_sim)
    # cholesky: d^T V^{-1} d = ||L^{-1} d||^2
    y = torch.linalg.solve_triangular(
        L, d, upper=False
    )  # y = L^{-1} d (batch, N_sim, n_cells, 1)
    exp = (y.squeeze(-1) ** 2).sum(dim=-1)  # sum(y^2) = y^T y = ||y||^2 (batch, N_sim)

    # tr_term = torch.sum(sigma2_q * torch.diagonal(torch.linalg.inv(V_reg), dim1=-2, dim2=-1), dim=-1)
    # cholesky: trace(V^{-1} Σ) = ||diag(L^{-1} Σ)||_F^2
    S_half = torch.diag_embed(
        torch.sqrt(sigma2_q)
    )  # Σ^{1/2} = diag(σ_i), shape (..., n, n)
    Y = torch.linalg.solve_triangular(L, S_half, upper=False)  # Y = L^{-1} Σ^{1/2}
    tr_term = (Y**2).sum(dim=(-2, -1))  # ||Y||_F^2 = tr(Y^T Y) (batch, N_sim)

    #const = n_cells * torch.log(torch.tensor(2 * torch.pi, device=device))
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


# calculate expectation of negative OU log likelihood with torch
def ou_neg_log_lik_torch_kkt(
    param_batch, sigma2_q, mode, expr_batch, diverge, share, epochs, beta, device
):
    """
    Same OU likelihood with torch. Used for ELBO with torch.

    params_batch: [alpha, other OU params]
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
    V = (
        (1 / (2 * alpha))
        * torch.exp(-alpha * diverge)
        * (1 - torch.exp(-2 * alpha * share))
    )  # (batch_size, N_sim, n_cells, n_cells)
    
    # Compute W and diff
    if mode == 1:
        W = torch.ones(
            (batch_size, N_sim, n_cells, 1), dtype=alpha.dtype, device=device
        )
    elif mode == 2:
        W = theta_weight_W_torch(
            alpha.squeeze(-1).squeeze(-1), epochs, beta
        )  # (batch_size, N_sim, n_cells, n_regimes)

    # cholesky decomposition for det and inv of matrix
    try:
        L = torch.linalg.cholesky(V)  # (batch_size, N_sim, n_cells, n_cells)
    except RuntimeError:
        # Add regularization to prevent singular matrix
        print("warning: singular matrix")
        try:
            L = torch.linalg.cholesky(
                V + 1e-6 * torch.eye(n_cells, device=device, dtype=V.dtype)
            )
        except RuntimeError:
            L = psd_safe_cholesky(V, base_jitter=1e-6, max_tries=7)

    # log_det = torch.linalg.slogdet(V_reg)[1]
    # cholesky: log|V| = 2 * sum(log diag(L))
    log_det = 2 * torch.sum(
        torch.log(torch.diagonal(L, dim1=2, dim2=3)), dim=2
    )  # (batch_size, N_sim)
    
    # theta = (W.T @ V^-1 @ W)^-1 @ W.T @ V^-1 @ expr
    # theta = np.linalg.solve(W.T @ np.linalg.solve(V, W), W.T @ np.linalg.solve(V, expr))
    # cholesky: theta = (S.T @ S)^-1 @ S.T @ y
    S = torch.linalg.solve(L, W)      # L^{-1} W
    y = torch.linalg.solve(L, expr_batch.unsqueeze(-1))   # L^{-1} expr
    #theta = torch.linalg.solve(S.T @ S, S.T @ y)
    theta = torch.linalg.lstsq(S, y).solution # (batch, N_sim, n_regimes, 1)    
    
    # sigma2 = (expr - W @ theta)^T @ V^-1 @ (expr - W @ theta) / n_cells
    # sigma2 = (diff @ np.linalg.solve(V, diff)).item() / n_cells
    # cholesky: sigma2 = (y - S @ theta)^T @ (y - S @ theta) / n_cells
    r = y - S @ theta
    sigma2 = (r.square().sum(dim=(-2, -1)) / n_cells) # (batch_size, N_sim)

    # tr_term = torch.sum(sigma2_q * torch.diagonal(torch.linalg.inv(V_reg), dim1=-2, dim2=-1), dim=-1)
    # cholesky: trace(V^{-1} Σ) = ||diag(L^{-1} Σ)||_F^2
    S_half = torch.diag_embed(
        torch.sqrt(sigma2_q)
    )  # Σ^{1/2} = diag(σ_i), shape (..., n, n)
    Y = torch.linalg.solve_triangular(L, S_half, upper=False)  # Y = L^{-1} Σ^{1/2}
    tr_term = (Y**2).sum(dim=(-2, -1))  # ||Y||_F^2 = tr(Y^T Y) (batch, N_sim)
    sigma2 += tr_term / n_cells

    #const = n_cells * torch.log(torch.tensor(2 * torch.pi, device=device))
    loss = 0.5 * (log_det + n_cells * torch.log(sigma2))  # -log likelihood

    sigma = sigma2 ** 0.5  # (batch_size, N_sim)
    theta = theta.squeeze(-1)  # (batch_size, N_sim, n_regimes)

    return loss, sigma, theta  # (batch_size, N_sim)
