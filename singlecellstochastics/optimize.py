from joblib import Parallel, delayed
import numpy as np
import torch
from scipy.optimize import minimize
import wandb

from .likelihood import ou_neg_log_lik_numpy
from .elbo import Lq_neg_log_lik_torch


# optimize OU likelihood with SciPy
def ou_optimize_scipy(
    params_init,
    mode,
    expr_list,
    diverge_list,
    share_list,
    epochs_list,
    beta_list,
    max_iter=500,
    learning_rate=1e-3,
    device="cpu",
):
    """
    Optimize OU likelihood with SciPy L-BFGS-B.
    Returns OU parameters for each batch and sim.
    Used for initializing OU parameters for ELBO with expression data.

    params_init: OU parameters (alpha, sigma2, theta0, theta1, ..., theta_n)
    mode: 1 for H0, 2 for H1
    expr: list of expression data (batch_size, N_sim, n_cells)
    """
    batch_size, N_sim, _ = expr_list[0].shape  # same for all trees

    # optimize OU for batch i, sim j
    def fit_one(i, j):
        expr = [x[i, j, :] for x in expr_list]
        res = minimize(
            ou_neg_log_lik_numpy,
            params_init,
            args=(mode, expr, diverge_list, share_list, epochs_list, beta_list),
            bounds=[(1e-6, None)] * len(params_init),
            method="L-BFGS-B",
        )
        return i, j, res.x

    # parallel execution
    results = Parallel(n_jobs=-1)(
        delayed(fit_one)(i, j) for i in range(batch_size) for j in range(N_sim)
    )

    # Convert results to tensor
    result_tensor = torch.empty(
        (batch_size, N_sim, len(params_init)), dtype=torch.float32, device=device
    )
    for i, j, x in results:
        result_tensor[i, j, :] = torch.tensor(x, dtype=torch.float32, device=device)

    return result_tensor


# optimize ELBO with Adam
def Lq_optimize_torch(
    params,
    mode,
    x_tensor_list,
    gene_names,
    diverge_list_torch,
    share_list_torch,
    epochs_list_torch,
    beta_list_torch,
    max_iter=500,
    learning_rate=1e-3,
    device="cpu",
    wandb_flag=False,
):
    """
    Optimize ELBO with PyTorch Adam.

    params: (batch_size, N_sim, all_param_dim)
    mode: 1 for H0, 2 for H1
    x_tensor: list of (batch_size, N_sim, n_cells)
    gene_names: list of gene names
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    params_tensor = [
        p.clone().detach().requires_grad_(True) for p in params
    ]  # list of (batch_size, N_sim, param_dim)
    batch_size, N_sim, _ = params_tensor[0].shape
    n_trees = len(x_tensor_list)
    best_params = params[-1].clone().detach()  # (batch_size, N_sim, ou_param_dim)
    best_loss = torch.full((batch_size, N_sim), float("inf"), device=device)
    optimizer = torch.optim.Adam(params_tensor, lr=learning_rate)

    for n in range(max_iter):
        optimizer.zero_grad()

        # get loss for each tree
        loss_matrix = torch.zeros(batch_size, N_sim, n_trees, device=device)
        for i in range(n_trees):
            x_tensor = x_tensor_list[i]
            Lq_params = params_tensor[i]
            ou_params = params_tensor[-1]
            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]
            loss = Lq_neg_log_lik_torch(
                Lq_params,
                ou_params,
                mode,
                x_tensor,
                diverge,
                share,
                epochs,
                beta,
                device=device,
            )
            loss_matrix[:, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(
            torch.tensor(n_trees, device=device, dtype=torch.float32)
        )  # (batch_size, N_sim)

        # update params
        loss = average_loss.sum()
        loss.backward()

        with torch.no_grad():
            # s2, alpha, sigma2 should be positive TODO: try log/square transform
            for i in range(n_trees - 1):
                n_cells = x_tensor_list[i].shape[-1]
                if (params_tensor[i][:, :, n_cells:] < 1e-6).any():
                    params_tensor[i][:, :, n_cells:].clamp_(min=1e-6)
            if (params_tensor[-1][:, :, :2] < 1e-6).any():
                params_tensor[-1][:, :, :2].clamp_(min=1e-6)

            # update best params
            mask = average_loss < best_loss
            best_loss[mask] = average_loss[mask].clone().detach()
            best_params[mask] = params_tensor[-1][mask].clone().detach()

            if wandb_flag:
                # plot loss of first gene in batch
                wandb.log(
                    {f"{gene_names[0]}_h{mode-1}_loss": best_loss[0, 0], "iter": n}
                )

        optimizer.step()

    return (
        best_params.clone().detach().cpu().numpy(),
        best_loss.clone().detach().cpu().numpy(),
    )
