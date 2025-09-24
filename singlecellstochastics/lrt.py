import numpy as np
import torch

from .optimize import ou_optimize_scipy, ou_optimize_torch, Lq_optimize_torch
from .em import run_em

# likelihood ratio test
def likelihood_ratio_test(
    expr,
    n_regimes,
    diverge_list,
    share_list,
    epochs_list,
    beta_list,
    diverge_list_torch,
    share_list_torch,
    epochs_list_torch,
    beta_list_torch,
    gene_names,
    device,
    max_iter,
    learning_rate,
    wandb_flag,
    window,
    tol,
    approx, 
    em_iter
):
    """
    Hypothesis testing for lineage-specific gene expression change.

    x_original: (batch_size, N_sim, n_cells) numpy array
    diverge_list, share_list, epochs_list, beta_list: list of numpy arrays
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    gene_names: list of gene names
    cache_dir: directory to save the cached init OU parameters
    window: number of iterations to check convergence
    tol: convergence tolerance
    approx: approximation method
    em_iter: number of EM iterations
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    # use the first gene as the example gene
    x_first = [
        x[0:1, 0:1, :] for x in expr
    ] # (n_cells,)

    # init OU parameters with one example gene
    ou_params_init = np.ones((n_regimes + 2))  # shape: (n_regimes+2)
    ou_params_init_h0 = ou_optimize_scipy(
        ou_params_init,
        1,
        x_first,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        device=device,
    )  # (1, 1, n_regimes+2)

    # optimize OU for null model
    x_tensor = [
        torch.tensor(x, dtype=torch.float32, device=device) for x in expr
    ]  # list of (batch_size, N_sim, n_cells)
    
    # Expand params_init to match batch size
    batch_size, N_sim, _ = x_tensor[0].shape
    ou_params_init_h0_tensor = ou_params_init_h0.expand(batch_size, N_sim, -1)

    ou_params_h0, ou_loss_h0 = ou_optimize_torch(
        ou_params_init_h0_tensor,
        1,
        x_tensor,
        diverge_list_torch,
        share_list_torch,
        epochs_list_torch,
        beta_list_torch,
        device=device,
        max_iter=max_iter,
        learning_rate=learning_rate,
        wandb_flag=wandb_flag,
        gene_names=gene_names,
        window=window,
        tol=tol
    )

    # init Lq with expression data
    pois_params_init = [
        torch.cat((x, torch.ones_like(x, device=device)), dim=2) for x in x_tensor
    ]  # list of (batch_size, N_sim, 2*n_cells)

    init_params = pois_params_init + [ou_params_h0]  # list of (batch_size, N_sim, param_dim)

    # optimize Lq for null model
    if em_iter == 0: # optimize all params
        h0_params, h0_loss = Lq_optimize_torch(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter=max_iter,
            learning_rate=learning_rate,
            device=device,
            wandb_flag=wandb_flag,
            window=window,
            tol=tol,
            approx=approx,
            em=None
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    else: # EM
        h0_params, h0_loss = run_em(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter=max_iter,
            learning_rate=learning_rate,
            device=device,
            wandb_flag=wandb_flag,
            window=window,
            tol=tol,
            approx=approx,
            em_iter=em_iter
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)

    # init OU parameters with one example gene
    ou_params_init_h1 = ou_optimize_scipy(
        ou_params_init,
        2,
        x_first,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        device=device,
    )  # (1, 1, n_regimes+2)

    # Expand params_init to match batch size
    ou_params_init_h1_tensor = ou_params_init_h1.expand(batch_size, N_sim, -1)

    # optimize OU for alternative model
    ou_params_h1, ou_loss_h1 = ou_optimize_torch(
        ou_params_init_h1_tensor,
        2,
        x_tensor,
        diverge_list_torch,
        share_list_torch,
        epochs_list_torch,
        beta_list_torch,
        device=device,
        max_iter=max_iter,
        learning_rate=learning_rate,
        wandb_flag=wandb_flag,
        gene_names=gene_names,
        window=window,
        tol=tol
    )

    # optimize Lq for alternative model
    init_params = pois_params_init + [ou_params_h1]  # list of (batch_size, N_sim, param_dim)

    if em_iter == 0: # optimize all params
        h1_params, h1_loss = Lq_optimize_torch(
            init_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter=max_iter,
            learning_rate=learning_rate,
            device=device,
            wandb_flag=wandb_flag,
            window=window,
            tol=tol,
            approx=approx,
            em=None
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    else: # EM
        h1_params, h1_loss = run_em(
            init_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter=max_iter,
            learning_rate=learning_rate,
            device=device,
            wandb_flag=wandb_flag,
            window=window,
            tol=tol,
            approx=approx,
            em_iter=em_iter
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)

    return h0_params, h0_loss, h1_params, h1_loss, \
        ou_params_h0.clone().detach().cpu().numpy(), \
        ou_loss_h0.clone().detach().cpu().numpy(), \
        ou_params_h1.clone().detach().cpu().numpy(), \
        ou_loss_h1.clone().detach().cpu().numpy()

    