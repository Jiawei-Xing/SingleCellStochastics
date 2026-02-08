from math import dist
import torch

from .weights import theta_weight_W_torch

def importance_sampling(
    params,
    mode,
    x_tensor,
    diverge_list_torch, 
    share_list_torch, 
    epochs_list_torch, 
    beta_list_torch, 
    device,
    nb,
    lib,
    n_samples
):
    """
    Estimate the real log-likelihood using importance sampling.
    Uses fixed q parameters from optimization as proposal distribution,
    and computes importance weights from samples drawn from q.
    log p(x) ≈ log_sum_exp(log p(x,z_i) - log q(z_i|x)) - log(n_samples)
    
    Args:
        params: list of (Lq_params_tree1, ..., Lq_params_treeN, log_r, ou_params)
        mode: 1 for H0, 2 for H1
        x_tensor: list of (batch_size, N_sim, n_cells) * trees observed counts
        diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: tree parameters
        device: torch device
        nb: whether to use negative binomial
        lib: library size normalization
    
    Returns:
        negative log_likelihood: (batch_size, N_sim) tensor of log-likelihood estimates
    """
    log_r = params[-2]
    ou_params = params[-1]
    ntree = len(x_tensor)

    # Compute log p(x,z_i) and log q(z_i|x), looping over trees
    log_p_q = []
    for i in range(ntree):
        # sample z_i ~ q(z|x) for each tree
        ncell = x_tensor[i].shape[-1]
        q_mean, q_var = params[i], params[i + ncell] # (batch_size, N_sim, n_cells)
        dist = torch.distributions.Normal(q_mean, torch.sqrt(q_var))
        samples = dist.sample((n_samples,)) # (n_samples, batch_size, N_sim, n_cells)

        # log q(z_i|x)
        log_q = torch.log(samples).sum(dim=-1)  # (n_samples, batch_size, N_sim)

        # OU params
        batch_size, N_sim, n_cells = x_tensor[i].shape
        alpha = torch.exp(ou_params[:, :, 0])  # (batch_size, N_sim)
        sigma2 = ou_params[:, :, 1]**2  # (batch_size, N_sim)
        thetas = ou_params[:, :, 2:]  # (batch_size, N_sim, n_regimes)

        alpha = alpha[:, :, None, None]  # (batch_size, N_sim, 1, 1)
        sigma2 = sigma2[:, :, None, None]  # (batch_size, N_sim, 1, 1)
        diverge = diverge_list_torch[i][None, None, :, :]  # (1, 1, n_cells, n_cells)
        share = share_list_torch[i][None, None, :, :]  # (1, 1, n_cells, n_cells)

        # OU covariance matrix
        V = (
            (sigma2 / (2 * alpha))
            * torch.exp(-alpha * diverge)
            * (1 - torch.exp(-2 * alpha * share))
        )  # (batch_size, N_sim, n_cells, n_cells)
        
        # OU mean
        if mode == 1:
            W = torch.ones(
                (batch_size, N_sim, n_cells), dtype=alpha.dtype, device=device
            )
            mean = W * thetas[:, :, 0:1]  # (batch_size, N_sim, n_cells)
        elif mode == 2:
            W = theta_weight_W_torch(
                alpha.squeeze(-1).squeeze(-1), epochs_list_torch[i], beta_list_torch[i]
            )  # (batch_size, N_sim, n_cells, n_regimes)
            mean = torch.matmul(
                W, thetas.unsqueeze(-1)
            ).squeeze(-1) # (batch_size, N_sim, n_cells)

        # OU log-likelihood: log p(z_i)
        dist = torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=V
        )
        log_p_z = dist.log_prob(samples)   # (n_samples, batch_size, N_sim)

        # log p(x_i|z_i)
        if not nb:
            dist = torch.distributions.Poisson(rate=torch.nn.functional.softplus(samples) * lib[i])
        else:            
            dist = torch.distributions.NegativeBinomial(
                total_count=torch.nn.functional.softplus(log_r) * lib[i], 
                logits=samples
            )
        log_p_x_given_z = dist.log_prob(x_tensor[i].unsqueeze(0))  # (n_samples, batch_size, N_sim, n_cells)
        log_p_x_given_z = log_p_x_given_z.sum(dim=-1)  # (n_samples, batch_size, N_sim)

        # log p(x,z_i) = log p(z_i) + log p(x_i|z_i)
        log_p_xz = log_p_z + log_p_x_given_z  # (n_samples, batch_size, N_sim)

        # log p(x,z_i) - log q(z_i|x)
        log_p_q.append(log_p_xz - log_q)  # (n_samples, batch_size, N_sim)

    # log_sum_exp over samples, then average over trees
    log_p_q = torch.stack(log_p_q, dim=0)  # (ntree, n_samples, batch_size, N_sim)
    log_likelihood = torch.logsumexp(log_p_q, dim=1) - torch.log(torch.tensor(n_samples, dtype=log_p_q.dtype, device=device))  # (ntree, batch_size, N_sim)
    log_likelihood = log_likelihood.mean(dim=0)  # (batch_size, N_sim)

    return -log_likelihood
