import numpy as np
import torch

from .optimize import ou_optimize_scipy, Lq_optimize_torch


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
    device="cpu",
    max_iter=500,
    learning_rate=1e-3,
):
    """
    Hypothesis testing for lineage-specific gene expression change.

    x_original: (batch_size, N_sim, n_cells) numpy array
    diverge_list, share_list, epochs_list, beta_list: list of numpy arrays
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    gene_names: list of gene names
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    x_pseudo = [
        np.maximum(x, 1e-6) for x in x_original
    ]  # add small value to avoid log(0)
    m_init = [
        np.log(np.expm1(x)) for x in x_pseudo
    ]  # reverse read counts as Gaussian mean z
    ou_params_init = np.ones((n_regimes + 2))  # shape: (n_regimes+2)

    # optimize OU for null model
    ou_params_h0 = ou_optimize_scipy(
        ou_params_init,
        1,
        m_init,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        max_iter=500,
        learning_rate=1e-3,
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
        ou_params_h0
    ]  # list of (batch_size, N_sim, param_dim)

    x_original_tensor = [
        torch.tensor(x, dtype=torch.float32, device=device) for x in x_original
    ]  # list of (batch_size, N_sim, n_cells)
    h0_params, h0_loss = Lq_optimize_torch(
        init_params,
        1,
        x_original_tensor,
        gene_names,
        diverge_list_torch,
        share_list_torch,
        epochs_list_torch,
        beta_list_torch,
        max_iter=500,
        learning_rate=1e-3,
        device=device,
    )  # (batch_size, N_sim, all_param_dim)

    # optimize OU for alternative model
    ou_params_h1 = ou_optimize_scipy(
        ou_params_init,
        2,
        m_init,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        max_iter=500,
        learning_rate=1e-3,
        device=device,
    )  # (batch_size, N_sim, n_regimes+2)

    # optimize Lq for alternative model
    init_params = pois_params_init + [
        ou_params_h1
    ]  # list of (batch_size, N_sim, param_dim)
    h1_params, h1_loss = Lq_optimize_torch(
        init_params,
        2,
        x_original_tensor,
        gene_names,
        diverge_list_torch,
        share_list_torch,
        epochs_list_torch,
        beta_list_torch,
        max_iter=500,
        learning_rate=1e-3,
        device=device,
    )  # (batch_size, N_sim, all_param_dim)

    return h0_params, h0_loss, h1_params, h1_loss
