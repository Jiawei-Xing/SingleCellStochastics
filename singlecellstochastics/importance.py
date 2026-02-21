import torch
import pandas as pd

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
    n_samples,
    mix
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
        n_samples: importance sampling sample number
        mix: weight for q(z) in the mixture proposal q(z) and p(z)
    
    Returns:
        negative log_likelihood: (batch_size, N_sim)
    """
    log_r = params[-2].unsqueeze(0) # add dim for samples
    ou_params = [params[-1][..., :1], params[-1][..., 1:]] # split alpha and others for ou likelihood
    ntree = len(x_tensor)

    # Compute log p(x,z_i) - log q(z_i|x_i), looping over trees
    log_p_q = []
    for i in range(ntree):
        # sample z_i ~ q(z|x) for each tree
        n_cells = x_tensor[i].shape[-1]
        q_mean, q_std = params[i][:, :, :n_cells], abs(params[i][:, :, n_cells:2*n_cells])  # (batch_size, N_sim, n_cells)
        dist_q = torch.distributions.Normal(q_mean, q_std)
        samples_q = dist_q.sample((n_samples,)) # (n_samples, batch_size, N_sim, n_cells)

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
        dist_ou = torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=V
        )

        if mix == 1:
            # log p(z_i)
            log_p_z = dist_ou.log_prob(samples_q)   # (n_samples, batch_size, N_sim)

            # log p(x_i|z_i)
            if not nb:
                dist_nb = torch.distributions.Poisson(rate=torch.nn.functional.softplus(samples_q) * lib[i])
            else:            
                mu = torch.nn.functional.softplus(samples_q) * lib[i]
                dist_nb = torch.distributions.NegativeBinomial(
                    total_count = torch.exp(log_r), 
                    logits = torch.log(mu) - log_r
                )
            log_p_x_given_z = dist_nb.log_prob(x_tensor[i].unsqueeze(0))  # (n_samples, batch_size, N_sim, n_cells)
            log_p_x_given_z = log_p_x_given_z.sum(dim=-1)  # (n_samples, batch_size, N_sim)

            # log p(x,z_i) = log p(z_i) + log p(x_i|z_i)
            log_p_xz = log_p_z + log_p_x_given_z  # (n_samples, batch_size, N_sim)

            # log p(x,z_i) - log q(z_i|x_i)
            log_q_z = dist_q.log_prob(samples_q).sum(dim=-1) # (n_samples, batch_size, N_sim)
            log_p_q.append(log_p_xz - log_q_z)  # (n_samples, batch_size, N_sim)
        else:
            # mixture weights
            mix = torch.as_tensor(mix, device=device, dtype=q_mean.dtype)
            log_mix = torch.log(mix)
            log_1m  = torch.log1p(-mix)  # log(1-mix)

            # sample from p(z) or q(z)
            samples_p = dist_ou.sample((n_samples,))  # (S,B,N,C)
            mask = (torch.rand(n_samples, batch_size, N_sim, device=device) < mix).to(mix.dtype)
            mask = mask[..., None]  # (S,B,N,1)
            samples_mixed = mask * samples_q + (1 - mask) * samples_p  # (S,B,N,C)

            # log p(z_i)
            log_p_z = dist_ou.log_prob(samples_mixed)   # (n_samples, batch_size, N_sim)

            # log p(x_i|z_i)
            if not nb:
                dist_nb = torch.distributions.Poisson(rate=torch.nn.functional.softplus(samples_mixed) * lib[i])
            else:            
                mu = torch.nn.functional.softplus(samples_mixed) * lib[i]
                dist_nb = torch.distributions.NegativeBinomial(
                    total_count = torch.exp(log_r), 
                    logits = torch.log(mu) - log_r
                )
            log_p_x_given_z = dist_nb.log_prob(x_tensor[i].unsqueeze(0))  # (n_samples, batch_size, N_sim, n_cells)
            log_p_x_given_z = log_p_x_given_z.sum(dim=-1)  # (n_samples, batch_size, N_sim)

            # log p(x,z_i) = log p(z_i) + log p(x_i|z_i)
            log_p_xz = log_p_z + log_p_x_given_z  # (n_samples, batch_size, N_sim)

            # mixture proposal
            log_q_z = dist_q.log_prob(samples_mixed).sum(dim=-1) # (n_samples, batch_size, N_sim)
            denominator = torch.logsumexp(
                torch.stack([log_mix + log_q_z, log_1m + log_p_z], dim=0),  # (2,S,B,N)
                dim=0
            )  # (S,B,N)

            log_p_q.append(log_p_xz - denominator)  # (S,B,N)
            mix = mix.item()  # convert to scalar for printing

    # sum log across trees, then log_sum_exp over samples
    log_p_q = torch.stack(log_p_q, dim=0).sum(dim=0)  # (n_samples, batch_size, N_sim)
    log_likelihood = torch.logsumexp(log_p_q, dim=0) - torch.log(torch.tensor(n_samples, dtype=log_p_q.dtype, device=device))  # (batch_size, N_sim)

    # output weights distribution to check ESS
    w = torch.softmax(log_p_q[..., 0], dim=0) # (n_samples, batch_size)
    ESS = 1 / (w**2).sum(dim=0) # (batch_size,)
    print(f"ESS: {ESS}")
    df = pd.DataFrame(
        log_p_q[..., 0].detach().cpu().numpy(),
        index=[f"sample_{i}" for i in range(log_p_q.shape[0])],
        columns=[f"batch_{j}" for j in range(log_p_q.shape[1])]
    )
    df.to_csv(f"IS{n_samples}_mix{mix}_weights.tsv", sep="\t", mode="a")

    return -log_likelihood
