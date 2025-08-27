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
        ou_params, s2, mode, m, diverge, share, epochs, beta
    )  # (batch_size, N_sim)

    # Monte Carlo approximation for E[log(softplus(z))] and E[softplus(z)] with reparameterization
    def E_log_softplus(u, sigma2, n_samples=1000):
        """
        Monte Carlo approximation for E[log(softplus(z))] where z ~ N(u, sigma2)
        Uses reparameterization trick: z = u + sqrt(sigma2) * epsilon, where epsilon ~ N(0,1)
        
        Parameters:
        -----------
        u : torch.Tensor
            Mean of the normal distribution
        sigma2 : torch.Tensor
            Variance of the normal distribution
        n_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        result : torch.Tensor
            Monte Carlo approximation of E[log(softplus(z))]
        """
        # Reparameterization: z = u + sqrt(sigma2) * epsilon, epsilon ~ N(0,1)
        epsilon = torch.randn(n_samples, *u.shape, device=u.device, dtype=u.dtype)
        z = u.unsqueeze(0) + torch.sqrt(sigma2).unsqueeze(0) * epsilon
        
        # Apply log(softplus) transformation
        log_softplus_z = torch.log(torch.nn.functional.softplus(z))
        
        # Take the mean over samples dimension
        result = torch.mean(log_softplus_z, dim=0)
        
        return result
    
    def E_softplus(u, sigma2, n_samples=1000):
        """
        Monte Carlo approximation for E[softplus(z)] where z ~ N(u, sigma2)
        Uses reparameterization trick: z = u + sqrt(sigma2) * epsilon, where epsilon ~ N(0,1)
        
        Parameters:
        -----------
        u : torch.Tensor
            Mean of the normal distribution
        sigma2 : torch.Tensor
            Variance of the normal distribution
        n_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        result : torch.Tensor
            Monte Carlo approximation of E[softplus(z)]
        """
        # Reparameterization: z = u + sqrt(sigma2) * epsilon, epsilon ~ N(0,1)
        epsilon = torch.randn(n_samples, *u.shape, device=u.device, dtype=u.dtype)
        z = u.unsqueeze(0) + torch.sqrt(sigma2).unsqueeze(0) * epsilon
        
        # Apply softplus transformation
        softplus_z = torch.nn.functional.softplus(z)
        
        # Take the mean over samples dimension
        result = torch.mean(softplus_z, dim=0)
        
        return result

    # Poisson -log lik
    term2 = -torch.sum(
        x_tensor * E_log_softplus(m, s2) - E_softplus(m, s2), dim=2
    )  # (batch_size, N_sim)

    # -entropy
    term3 = -torch.sum(0.5 * torch.log(s2), dim=2)  # (batch_size, N_sim)

    loss = term1 + term2 + term3  # (batch_size, N_sim)
    return loss  # (batch_size, N_sim)