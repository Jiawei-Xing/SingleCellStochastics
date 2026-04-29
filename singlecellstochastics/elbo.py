"""Evidence lower bound for LAVOUS latent-expression models.

This module combines the Gaussian tree prior from `likelihood.py` with
softplus/exponential count-observation expectations from `approx.py`.
"""

import numpy as np
import torch

from .defaults import DEFAULT_LOG_S2_CLAMP
from .likelihood import ou_neg_log_lik_torch, ou_neg_log_lik_torch_kkt, bm_neg_log_lik_torch_kkt
from .approx import E_log_softplus_taylor, E_softplus_taylor, E_log_softplus_MC, E_softplus_MC, E_log_exp, E_exp, E_log_r_softplus_MC, E_log_r_exp
from .trace import TRACE, nan_inf_count
    
# calculate negative log likelihood of ELBO with torch
def Lq_neg_log_lik_torch(
    Lq_params,
    log_r,
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
    kkt,
    nb,
    lib,
    const,
    fix_alpha=False,
    ou_lambda=None,
    ou_lambda_mode=1,
    log_s2_clamp=DEFAULT_LOG_S2_CLAMP,
    root_mode="stationary",
):
    """
    ELBO for approximating model evidence.

    Lq_params: (batch_size, N_sim, 2*n_cells), storing [q_mean, log_q_s2]
    log_r: (batch_size, N_sim) log dispersion parameter for negative binomial
    ou_params: [alpha, other OU params] (OU) or pagel_lambda (BM)
    mode: 0 for BM, 1 for OU-H0, 2 for OU-H1
    x_tensor: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    device: torch device
    approx: approximation method for Poisson likelihood expectation
    prior: L2 regularization strength for OU alpha
    kkt: whether to use KKT condition for OU likelihood
    nb: whether to use negative binomial likelihood
    lib: library size normalization
    const: whether to include constant terms in likelihood
    
    Returns: (batch_size, N_sim) tensor of losses
    """
    n_cells = x_tensor.shape[-1]
    m = Lq_params[..., :n_cells]  # (batch_size, ..., n_cells)
    # Second half stores log(s2). Clamp to keep s2 strictly positive and bounded
    # so the MC sqrt(s2) is well-defined under GPU/batched autograd.
    s2 = torch.exp(torch.clamp(
        Lq_params[..., n_cells:2*n_cells], log_s2_clamp[0], log_s2_clamp[1]
    ))  # (batch_size, ..., n_cells)

    if TRACE.enabled:
        m0 = m.reshape(-1, n_cells)[0].detach().cpu().numpy()
        s2_0 = s2.reshape(-1, n_cells)[0].detach().cpu().numpy()
        TRACE.write("q_mean_min", float(m0.min()))
        TRACE.write("q_mean_max", float(m0.max()))
        TRACE.write("q_mean_median", float(np.median(m0)))
        TRACE.write("q_s2_min", float(s2_0.min()))
        TRACE.write("q_s2_max", float(s2_0.max()))
        TRACE.write("q_s2_median", float(np.median(s2_0)))
        nan_m, inf_m = nan_inf_count(m)
        nan_s, inf_s = nan_inf_count(s2)
        TRACE.write("q_mean_n_nan", nan_m)
        TRACE.write("q_mean_n_inf", inf_m)
        TRACE.write("q_s2_n_nan", nan_s)
        TRACE.write("q_s2_n_inf", inf_s)
    
    # term1: -log likelihood of OU or BM
    if mode == 0:  # BM -log lik
        term1, mu, sigma = bm_neg_log_lik_torch_kkt(
            ou_params, s2, m, share, device
        )  # (batch_size, N_sim)
        
        if const:
            term1 += n_cells/2 * (1 + torch.log(2 * torch.tensor(torch.pi, device=device)))
    
    elif kkt: # OU -log lik
        term1, sigma, theta = ou_neg_log_lik_torch_kkt(
            ou_params, s2, mode, m, diverge, share, epochs, beta, device,
            pagel_lambda=ou_lambda,
            pagel_lambda_mode=ou_lambda_mode,
            root_mode=root_mode,
        )  # (batch_size, N_sim)

        if const:
            term1 += n_cells/2 * (1 + torch.log(2 * torch.tensor(torch.pi, device=device)))

    else: # OU -log lik without KKT
        term1 = ou_neg_log_lik_torch(
            ou_params, s2, mode, m, diverge, share, epochs, beta, device,
            pagel_lambda=ou_lambda,
            pagel_lambda_mode=ou_lambda_mode,
            root_mode=root_mode,
        )  # (batch_size, N_sim)

        if const:
            term1 += n_cells/2 * torch.log(2 * torch.tensor(torch.pi, device=device))

    # term2: Poisson -log lik
    if not nb:
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

        term2 = -torch.sum(x_tensor * E_log_approx - lib * E_approx, dim=-1) # (batch_size, N_sim)
    
    # term2: Negative binomial -log lik
    else:
        log_r_raw = log_r  # save pre-cleanup value for trace
        # log_r is clamped to its bounds in the optimizer
        log_r = log_r.unsqueeze(-1)
        r = torch.exp(log_r)
        if approx == "softplus_MC":
            E_log_approx = E_log_softplus_MC(m, s2)
            E_log_r_approx = E_log_r_softplus_MC(m, s2, r, lib)
        elif approx == "exp":
            E_log_approx = E_log_exp(m, s2)
            E_log_r_approx = E_log_r_exp(m, s2, log_r, lib)
        else:
            raise ValueError(f"Invalid approximation method: {approx}")

        nb_per_cell = (
            torch.lgamma(x_tensor + r) - torch.lgamma(r) + r * log_r +
            x_tensor * E_log_approx - (x_tensor + r) * E_log_r_approx
        )
        term2 = -torch.sum(nb_per_cell, dim=-1)  # (batch_size, N_sim)

        if TRACE.enabled:
            # log_r used inside (post clamp), and pre-clamp original.
            try:
                TRACE.write("log_r_used", float(log_r.reshape(-1)[0].detach().cpu()))
                TRACE.write("log_r_raw", float(log_r_raw.reshape(-1)[0].detach().cpu()))
                TRACE.write("r_used", float(r.reshape(-1)[0].detach().cpu()))
            except Exception:
                pass
            E_log_0 = E_log_approx.reshape(-1, n_cells)[0]
            E_logr_0 = E_log_r_approx.reshape(-1, n_cells)[0]
            TRACE.write("E_log_softplus_min", float(E_log_0.min().detach().cpu()))
            TRACE.write("E_log_softplus_max", float(E_log_0.max().detach().cpu()))
            TRACE.write("E_log_r_softplus_min", float(E_logr_0.min().detach().cpu()))
            TRACE.write("E_log_r_softplus_max", float(E_logr_0.max().detach().cpu()))
            nan_e1, inf_e1 = nan_inf_count(E_log_approx)
            nan_e2, inf_e2 = nan_inf_count(E_log_r_approx)
            TRACE.write("E_log_softplus_n_nan_inf", nan_e1 + inf_e1)
            TRACE.write("E_log_r_softplus_n_nan_inf", nan_e2 + inf_e2)
            nb_per_cell_0 = nb_per_cell.reshape(-1, n_cells)[0]
            n_neg_inf_cells = int(torch.isinf(nb_per_cell_0).sum().detach().cpu())
            n_nan_cells = int(torch.isnan(nb_per_cell_0).sum().detach().cpu())
            TRACE.write("nb_per_cell_n_neg_inf_or_inf", n_neg_inf_cells)
            TRACE.write("nb_per_cell_n_nan", n_nan_cells)
            TRACE.write("nb_per_cell_min", float(nb_per_cell_0.min().detach().cpu()))
            TRACE.write("nb_per_cell_max", float(nb_per_cell_0.max().detach().cpu()))
            # mu_obs = lib * softplus(m) for softplus_MC
            with torch.no_grad():
                mu_obs = lib * torch.nn.functional.softplus(m.reshape(-1, n_cells)[0])
                TRACE.write("mu_obs_max", float(mu_obs.max().detach().cpu()))
                TRACE.write("mu_obs_median", float(mu_obs.median().detach().cpu()))

    if const:
        term2 += -torch.sum(x_tensor * torch.log(lib) - torch.lgamma(x_tensor + 1), dim=-1) # x!

    # term3: -entropy
    term3 = -0.5 * torch.sum(torch.log(s2), dim=-1)  # (batch_size, N_sim)

    if const:
        term3 += -n_cells/2 * (1 + torch.log(2 * torch.tensor(torch.pi, device=device)))

    # optional: regulation for OU alpha
    if mode == 0 or fix_alpha:  # no regularization for BM or fixed alpha
        reg = 0
    else:  # L2 regularization on log alpha (shrinks alpha toward 1)
        reg = 0.5 * prior * (ou_params[0][..., 0] ** 2)  # (batch_size, N_sim)
    loss = term1 + term2 + term3 + reg  # (batch_size, N_sim)

    if TRACE.enabled:
        TRACE.write("term1_ou", float(term1.reshape(-1)[0].detach().cpu()))
        TRACE.write("term2_nb_or_pois", float(term2.reshape(-1)[0].detach().cpu()))
        TRACE.write("term3_neg_entropy", float(term3.reshape(-1)[0].detach().cpu()))
        if isinstance(reg, torch.Tensor):
            TRACE.write("reg_alpha_l2", float(reg.reshape(-1)[0].detach().cpu()))
        else:
            TRACE.write("reg_alpha_l2", float(reg))
        TRACE.write("loss_total", float(loss.reshape(-1)[0].detach().cpu()))

    if mode == 0: # BM
        return loss, mu, sigma  # (batch_size, N_sim)
    elif kkt: # OU with KKT
        return loss, sigma, theta  # (batch_size, N_sim)
    else: # OU without KKT
        return loss  # (batch_size, N_sim)
