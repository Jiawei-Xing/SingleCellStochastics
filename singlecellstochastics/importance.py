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
    Using samples z_i ~ q(z|x), we estimate:
    log p(x) ≈ log_sum_exp(log p(x,z_i) - log q(z_i|x)) - log(n_samples)
    
    Args:
        params: list of (Lq_params_tree1, ..., Lq_params_treeN, log_r, ou_params)
        mode: 1 for H0, 2 for H1
        x_tensor: list of (batch_size, N_sim, n_cells) * trees observed counts
        diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: tree parameters
        device: torch device
        approx, prior, kkt: ELBO parameters
        nb: whether to use negative binomial
        lib: library size normalization
    
    Returns:
        negative log_likelihood: (batch_size, N_sim)
    """
    log_r = params[-2].unsqueeze(0) # add dim for samples
    ou_params = [params[-1][..., :1], params[-1][..., 1:]] # split alpha and others for ou likelihood
    ntree = len(x_tensor)

    # Compute log p(x,z_i) and elbo, looping over trees
    log_p_q = []
    for i in range(ntree):
        # sample z_i ~ q(z|x) for each tree
        n_cells = x_tensor[i].shape[-1]
        q_mean, q_std = params[i][:, :, :n_cells], params[i][:, :, n_cells:2*n_cells] # (batch_size, N_sim, n_cells)
        dist = torch.distributions.Normal(q_mean, q_std)
        samples = dist.sample((n_samples,)) # (n_samples, batch_size, N_sim, n_cells)

        # log q(z_i|x_i)
        log_q_z = dist.log_prob(samples).sum(dim=-1) # (n_samples, batch_size, N_sim)

        # OU params
        batch_size, N_sim, n_cells = x_tensor[i].shape
        alpha = torch.exp(ou_params[0][:, :, 0])  # (batch_size, N_sim)
        sigma2 = ou_params[1][:, :, 0]**2  # (batch_size, N_sim)
        thetas = ou_params[1][:, :, 1:]  # (batch_size, N_sim, n_regimes)

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
            mu = torch.nn.functional.softplus(samples) * lib[i]
            dist = torch.distributions.NegativeBinomial(
                total_count = torch.exp(log_r), 
                logits = torch.log(mu) - log_r
            )
        log_p_x_given_z = dist.log_prob(x_tensor[i].unsqueeze(0))  # (n_samples, batch_size, N_sim, n_cells)
        log_p_x_given_z = log_p_x_given_z.sum(dim=-1)  # (n_samples, batch_size, N_sim)

        # log p(x,z_i) = log p(z_i) + log p(x_i|z_i)
        log_p_xz = log_p_z + log_p_x_given_z  # (n_samples, batch_size, N_sim)

        # log p(x,z_i) - log q(z_i|x_i)
        log_p_q.append(log_p_xz - log_q_z)  # (n_samples, batch_size, N_sim)

    # sum log across trees, then log_sum_exp over samples
    log_p_q = torch.stack(log_p_q, dim=0).sum(dim=0)  # (n_samples, batch_size, N_sim)
    log_likelihood = torch.logsumexp(log_p_q, dim=0) - torch.log(torch.tensor(n_samples, dtype=log_p_q.dtype, device=device))  # (batch_size, N_sim)

    return -log_likelihood
