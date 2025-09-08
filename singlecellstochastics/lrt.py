import numpy as np
import torch
import pickle
import os

from .optimize import ou_optimize_scipy, ou_optimize_torch, Lq_optimize_torch
from .em import run_em

# likelihood ratio test
def likelihood_ratio_test(
    x_original,
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
    cache_dir,
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
    x_pseudo = [
        np.maximum(x, 1e-6) for x in x_original
    ]  # add small value to avoid log(0)
    m_init = [
        np.log(np.expm1(x)) if approx != "exp" 
        else np.log(x) 
        for x in x_pseudo
    ] # reverse read counts as Gaussian mean z

    # use the first gene as the example gene
    m_first = [
        m[0:1, 0:1, :] for m in m_init
    ] # (n_cells,)

    # init OU parameters with one example gene
    ou_params_init = np.ones((n_regimes + 2))  # shape: (n_regimes+2)
    ou_params_init_h0 = ou_optimize_scipy(
        ou_params_init,
        1,
        m_first,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        device=device,
    )  # (1, 1, n_regimes+2)
    
    # root square of alpha and sigma2
    ou_params_init_h0_sqrt = torch.cat([
        ou_params_init_h0[:, :, :2].sqrt(), 
        ou_params_init_h0[:, :, 2:]
    ], dim=-1)

    # optimize OU for null model
    m_init_tensor = [
        torch.tensor(m, dtype=torch.float32, device=device) for m in m_init
    ]  # list of (batch_size, N_sim, n_cells)
    
    # Expand params_init to match batch size
    batch_size, N_sim, _ = m_init_tensor[0].shape
    ou_params_init_h0_sqrt = ou_params_init_h0_sqrt.expand(batch_size, N_sim, -1)

    ou_params_h0, ou_loss_h0 = ou_optimize_torch(
        ou_params_init_h0_sqrt,
        1,
        m_init_tensor,
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

    # save init h0 OU parameters to cache
    if cache_dir is not None:
        file_path = cache_dir + "_OUinit_h0.tsv"
        save_ou(file_path, ou_params_h0, ou_loss_h0, gene_names, approx, "1")

    # init Lq with expression data
    pois_params_init = [
        torch.cat((m, torch.ones_like(m, device=device)), dim=2) for m in m_init_tensor
    ]  # list of (batch_size, N_sim, 2*n_cells)

    init_params = pois_params_init + [ou_params_h0]  # list of (batch_size, N_sim, param_dim)

    x_original_tensor = [
        torch.tensor(x, dtype=torch.float32, device=device) for x in x_original
    ]  # list of (batch_size, N_sim, n_cells)

    # optimize Lq for null model
    if em_iter == 0: # optimize all params
        h0_params, h0_loss = Lq_optimize_torch(
            init_params,
            1,
            x_original_tensor,
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
            x_original_tensor,
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
        m_first,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        device=device,
    )  # (1, 1, n_regimes+2)
    
    # root square of alpha and sigma2
    ou_params_init_h1_sqrt = torch.cat([
        ou_params_init_h1[:, :, :2].sqrt(), 
        ou_params_init_h1[:, :, 2:]
    ], dim=-1)
    
    # Expand params_init to match batch size
    ou_params_init_h1_sqrt = ou_params_init_h1_sqrt.expand(batch_size, N_sim, -1)

    # optimize OU for alternative model
    ou_params_h1, ou_loss_h1 = ou_optimize_torch(
        ou_params_init_h1_sqrt,
        2,
        m_init_tensor,
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

    # save init h1 OU parameters to cache
    if cache_dir is not None:
        file_path = cache_dir + "_OUinit_h1.tsv"
        save_ou(file_path, ou_params_h1, ou_loss_h1, gene_names, approx, "2")

    # optimize Lq for alternative model
    init_params = pois_params_init + [ou_params_h1]  # list of (batch_size, N_sim, param_dim)

    if em_iter == 0: # optimize all params
        h1_params, h1_loss = Lq_optimize_torch(
            init_params,
            2,
            x_original_tensor,
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
            x_original_tensor,
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

    return h0_params, h0_loss, h1_params, h1_loss


# save init OU parameters and loss in TSV file with descriptive headers
def save_ou(file_path, ou_params, ou_loss, gene_names, approx, mode):
    """
    Save OU parameters and loss in TSV format with descriptive headers.
    
    file_path : path to save the TSV file
    ou_params : tensor of shape (batch_size, 1, param_dim)
    ou_loss : tensor of shape (batch_size, 1)
    gene_names : list of gene names
    mode : "1" for h0 or "2" for h1
    approx : approximation method
    """
    batch_size, N_sim, param_dim = ou_params.shape # N_sim should be 1
    
    # Create parameter names and header
    if mode == "1": # h0
        thetas = ["theta0"]
        ou_params = ou_params[:, :, :3]
    else: # h1
        thetas = [f"theta{i}" for i in range(param_dim - 2)]
    header = ["gene_name"] + ["alpha", "sigma2"] + thetas + ["loss"]
    
    with open(file_path, 'a') as f:
        f.write('\t'.join(header) + '\n')
        for i in range(batch_size):
            row = [
                gene_names[i], # gene name
                ou_params[i, 0, 0].item() ** 2, # alpha
                ou_params[i, 0, 1].item() ** 2 # sigma2 
            ] + [
                torch.nn.functional.softplus(x).item() if approx != "exp" 
                else torch.exp(x).item() 
                for x in ou_params[i, 0, :] # thetas
            ][2:] + [
                ou_loss[i, 0].item() # loss
            ] 
            f.write('\t'.join(map(str, row)) + '\n')
    