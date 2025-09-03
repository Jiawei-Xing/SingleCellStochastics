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

    ''' comment out: init OU with scipy
    ou_params_init = np.ones((n_regimes + 2))  # shape: (n_regimes+2)

    # Try to load cached OU parameters if cache_dir is provided
    ou_params_h0 = None
    ou_params_h1 = None
    loaded_from_cache = False
    
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        h0_cache_file = os.path.join(cache_dir, "ou_params_h0.pkl")
        h1_cache_file = os.path.join(cache_dir, "ou_params_h1.pkl")
        
        # Try to load cached parameters
        if os.path.exists(h0_cache_file) and os.path.exists(h1_cache_file):
            try:
                with open(h0_cache_file, 'rb') as f:
                    ou_params_h0 = pickle.load(f)
                with open(h1_cache_file, 'rb') as f:
                    ou_params_h1 = pickle.load(f)
                loaded_from_cache = True
                print(f"Loaded cached OU parameters from {cache_dir}")
            except Exception as e:
                print(f"Failed to load cached parameters: {e}")
                ou_params_h0 = None
                ou_params_h1 = None

    # optimize OU for null model (only if not loaded from cache)
    if ou_params_h0 is None:
        ou_params_h0 = ou_optimize_scipy(
        ou_params_init,
        1,
        m_init,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        device=device,
    )  # (batch_size, N_sim, n_regimes+2)
    '''

    # optimize OU for null model
    m_init_tensor = [
        torch.tensor(m, dtype=torch.float32, device=device) for m in m_init
    ]  # list of (batch_size, N_sim, n_cells)

    batch_size, N_sim, _ = m_init_tensor[0].shape
    ou_params_init_tensor = torch.ones(
        (batch_size, N_sim, n_regimes+2), dtype=torch.float32, device=device
    ) * 5 # (batch_size, N_sim, n_regimes+2)

    ou_params_h0, ou_loss_h0 = ou_optimize_torch(
        ou_params_init_tensor,
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
    
    pois_params_init = [
        torch.cat((m, torch.ones_like(m, device=device)), dim=2) for m in m_init_tensor
    ]  # list of (batch_size, N_sim, 2*n_cells)

    ''' # comment out: root squared OU parameters
    init_params = pois_params_init + [
        torch.cat((ou_params_h0[:, :, :2].sqrt(), ou_params_h0[:, :, 2:]), dim=2)
    ]  # list of (batch_size, N_sim, param_dim)
    '''

    init_params = pois_params_init + [ou_params_h0]  # list of (batch_size, N_sim, param_dim)

    x_original_tensor = [
        torch.tensor(x, dtype=torch.float32, device=device) for x in x_original
    ]  # list of (batch_size, N_sim, n_cells)

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

    ''' # comment out: init OU with scipy
    # optimize OU for alternative model (only if not loaded from cache)
    if ou_params_h1 is None:
        ou_params_h1 = ou_optimize_scipy(
            ou_params_init,
            2,
            m_init,
            diverge_list,
            share_list,
            epochs_list,
            beta_list,
            device=device,
        )  # (batch_size, N_sim, n_regimes+2)

    # Save OU parameters to cache if cache_dir is provided and parameters were computed (not loaded)
    if cache_dir is not None and not loaded_from_cache and ou_params_h0 is not None and ou_params_h1 is not None:
        try:
            h0_cache_file = os.path.join(cache_dir, "ou_params_h0.pkl")
            h1_cache_file = os.path.join(cache_dir, "ou_params_h1.pkl")
            
            with open(h0_cache_file, 'wb') as f:
                pickle.dump(ou_params_h0, f)
            with open(h1_cache_file, 'wb') as f:
                pickle.dump(ou_params_h1, f)
            print(f"Saved OU parameters to cache in {cache_dir}")
        except Exception as e:
            print(f"Failed to save parameters to cache: {e}")
    '''

    # read h1 OU parameters from cache
    ou_params_h1, ou_loss_h1 = ou_optimize_torch(
        ou_params_init_tensor,
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

    ''' # comment out: root squared OU parameters
    # optimize Lq for alternative model
    init_params = pois_params_init + [
        torch.cat((ou_params_h1[:, :, :2].sqrt(), ou_params_h1[:, :, 2:]), dim=2)
    ]  # list of (batch_size, N_sim, param_dim)
    '''

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
                ou_params[i, 0, 0] ** 2, # alpha
                ou_params[i, 0, 1] ** 2 # sigma2 
            ] + [
                torch.nn.functional.softplus(x) if approx != "exp" 
                else torch.exp(x) 
                for x in ou_params[i, 0, 2:] # thetas
            ] + [
                ou_loss[i, 0] # loss
            ] 
            f.write('\t'.join(map(str, row)) + '\n')
    