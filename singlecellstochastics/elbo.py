import numpy as np
import torch

from .likelihood import ou_neg_log_lik_torch


# calculate negative log likelihood of ELBO with torch
def Lq_neg_log_lik_torch(
    Lq_params, ou_params, mode, x_tensor, diverge, share, epochs, beta, device="cpu"
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
    s2 = Lq_params[:, :, n_cells:]  # (batch_size, N_sim, n_cells)

    # OU -log lik
    term1 = 0.5 * ou_neg_log_lik_torch(
        ou_params, s2, mode, m, diverge, share, epochs, beta, device=device
    )  # (batch_size, N_sim)

    # Approximation for E[log(softplus(z))] and E[softplus(z)] TODO: try exponential instead of softplus
    def E_log_softplus(u, sigma2):  # Taylor series for E[log]
        result = torch.empty_like(u)

        # u < 2, taylor0 + log(exp(z))
        def weighted(u, sigma2):
            log2 = torch.log(torch.tensor(2.0, device=device))

            # Taylor expansion at zero
            taylor0 = (
                torch.log(log2)
                + (0.5 / log2) * u
                + ((log2 - 1) / (8 * log2**2)) * (u**2 + sigma2)
            )

            # Weighting term
            w = 1 / (1 + torch.exp(2 * (u + 2)))

            return (1 - w) * taylor0 + w * u

        # 2 <= u < 10, log(z + exp(-z))
        def approx(u, sigma2):
            return (
                torch.log(u + torch.exp(-u))
                - ((1 - torch.exp(-u)) ** 2) / (2 * (u + torch.exp(-u)) ** 2) * sigma2
            )

        # u >= 10, logz
        def log_approx(u, sigma2):
            return torch.log(u)  # - sigma2 / (2 * u**2)

        result[u < 2] = weighted(u[u < 2], sigma2[u < 2])
        result[(u >= 2) & (u < 10)] = approx(
            u[(u >= 2) & (u < 10)], sigma2[(u >= 2) & (u < 10)]
        )
        result[u >= 10] = log_approx(u[u >= 10], sigma2[u >= 10])

        return result

    def E_softplus(u, sigma2):  # Taylor series for E[softplus]
        result = torch.empty_like(u)

        # u < 5, taylor(u)
        def taylor(u, sigma2):
            softplus = torch.nn.functional.softplus(u)
            sig = torch.sigmoid(u)
            return softplus + 0.5 * sigma2 * sig * (1 - sig)

        # u >= 5, z #+ exp(-z)
        def approx(u, sigma2):
            return u  # + torch.exp(-u + sigma2/2)

        result[u < 5] = taylor(u[u < 5], sigma2[u < 5])
        result[u >= 5] = approx(u[u >= 5], sigma2[u >= 5])

        return result

    # Poisson -log lik
    term2 = -torch.sum(
        x_tensor * E_log_softplus(m, s2) - E_softplus(m, s2), dim=2
    )  # (batch_size, N_sim)

    # -entropy
    term3 = -torch.sum(0.5 * torch.log(s2), dim=2)  # (batch_size, N_sim)

    loss = term1 + term2 + term3  # (batch_size, N_sim)
    return loss  # (batch_size, N_sim)
