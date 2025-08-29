import numpy as np
import torch
import pickle
import os

from .optimize import ou_optimize_scipy, Lq_optimize_torch
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

    # optimize Lq for null model
    m_init_tensor = [
        torch.tensor(m, dtype=torch.float32, device=device) for m in m_init
    ]  # list of (batch_size, N_sim, n_cells)
    pois_params_init = [
        torch.cat((m, torch.ones_like(m, device=device)), dim=2) for m in m_init_tensor
    ]  # list of (batch_size, N_sim, 2*n_cells)
    init_params = pois_params_init + [
        torch.cat((ou_params_h0[:, :, :2].sqrt(), ou_params_h0[:, :, 2:]), dim=2)
    ]  # list of (batch_size, N_sim, param_dim)

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

    # optimize Lq for alternative model
    init_params = pois_params_init + [
        torch.cat((ou_params_h1[:, :, :2].sqrt(), ou_params_h1[:, :, 2:]), dim=2)
    ]  # list of (batch_size, N_sim, param_dim)

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
