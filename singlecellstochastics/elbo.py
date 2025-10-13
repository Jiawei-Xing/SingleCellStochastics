import numpy as np
import torch

from .likelihood import ou_neg_log_lik_torch, ou_neg_log_lik_torch_kkt
from .approx import E_log_softplus_taylor, E_softplus_taylor, E_log_softplus_MC, E_softplus_MC, E_log_exp, E_exp
    
# calculate negative log likelihood of ELBO with torch
def Lq_neg_log_lik_torch(
    Lq_params, 
    ou_params, 
    mode, 
    x_tensor, 
    diverge, 
    share, 
    epochs, 
    beta, 
    device, 
    approx, 
    prior, 
    kkt
):
    """
    ELBO for approximating model evidence.

    Lq_params: (batch_size, N_sim, 2*n_cells)
    ou_params: [alpha, other OU params]
    mode: 1 for H0, 2 for H1
    x_tensor: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    device: torch device
    approx: approximation method for Poisson likelihood expectation
    prior: L2 regularization strength for OU alpha
    kkt: whether to use KKT condition for OU likelihood
    Returns: (batch_size, N_sim) tensor of losses
    """
    n_cells = x_tensor.shape[-1]
    m = Lq_params[:, :, :n_cells]  # (batch_size, N_sim, n_cells)
    s2 = Lq_params[:, :, n_cells:]**2  # (batch_size, N_sim, n_cells)
    
    # OU -log lik
    if kkt:
        term1, sigma, theta = ou_neg_log_lik_torch_kkt(
            ou_params, s2, mode, m, diverge, share, epochs, beta, device
        )  # (batch_size, N_sim)
    else:
        term1 = ou_neg_log_lik_torch(
            ou_params, s2, mode, m, diverge, share, epochs, beta, device
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

    #const= - torch.lgamma(x_tensor + 1)
    term2 = -torch.sum(x_tensor * E_log_approx - E_approx, dim=-1) # (batch_size, N_sim)

    # -entropy
    #const = torch.log(torch.tensor(2 * torch.pi * torch.e, device=device))
    term3 = -0.5 * torch.sum(torch.log(s2), dim=-1)  # (batch_size, N_sim)

    reg = 0.5 * prior *  (ou_params[0][:, :, 0] ** 2)  # (batch_size, N_sim)
    loss = term1 + term2 + term3 + reg  # (batch_size, N_sim)

    if kkt:
        return loss, sigma, theta  # (batch_size, N_sim)
    else:
        return loss  # (batch_size, N_sim)
