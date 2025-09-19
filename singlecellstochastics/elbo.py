import numpy as np
import torch

from .likelihood import ou_neg_log_lik_torch
from .approx import E_log_softplus_taylor, E_softplus_taylor, E_log_softplus_MC, E_softplus_MC, E_log_exp, E_exp
    
# calculate negative log likelihood of ELBO with torch
def Lq_neg_log_lik_torch(
    Lq_params, ou_params, mode, x_tensor, diverge, share, epochs, beta, device, approx
):
    """
    ELBO for approximating model evidence.

    Lq_params: (batch_size, N_sim, 2*n_cells)
    ou_params: (batch_size, N_sim, ou_param_dim)
    mode: 1 for H0, 2 for H1
    x_tensor: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    Returns: (batch_size, N_sim) tensor of losses
    """
    n_cells = x_tensor.shape[-1]
    m = Lq_params[:, :, :n_cells]  # (batch_size, N_sim, n_cells)
    s2 = Lq_params[:, :, n_cells:]**2 + 1e-6  # (batch_size, N_sim, n_cells)

    # OU -log lik
    term1 = ou_neg_log_lik_torch(
        ou_params, s2, mode, m, diverge, share, epochs, beta, device=device
    )  # (batch_size, N_sim)

    # Poisson -log lik
    if approx == "softplus_taylor":
        E_log_approx = E_log_softplus_taylor(m, s2)
        E_approx = E_softplus_taylor(m, s2)
    elif approx == "softplus_MC":
        E_log_approx = E_log_softplus_MC(m, s2)
        E_approx = E_softplus_MC(m, s2)
    elif approx == "exp":
        E_log_approx = E_log_exp(m, s2)
        E_approx = E_exp(m, s2)
    else:
        raise ValueError(f"Invalid approximation method: {approx}")

    term2 = -torch.sum(x_tensor * E_log_approx - E_approx, dim=2)  # (batch_size, N_sim)

    # -entropy
    term3 = -torch.sum(0.5 * torch.log(s2), dim=2)  # (batch_size, N_sim)

    loss = term1 + term2 + term3  # (batch_size, N_sim)
    return loss  # (batch_size, N_sim)
