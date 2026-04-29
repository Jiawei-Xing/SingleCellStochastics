"""Differential-expression likelihood-ratio test orchestration."""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from .em import run_em
from .importance import importance_sampling
from .optimize import Lq_optimize_torch_OU, ou_optimize_torch


def _lq_to_mean_std(lq):
    """Convert internal [mean, log_s2] q-params to [mean, std] for export."""
    n_cells = lq.shape[-1] // 2
    mean = lq[..., :n_cells]
    std = torch.exp(lq[..., n_cells:2 * n_cells] / 2)
    return torch.cat((mean, std), dim=-1)


def _gene_seed(gene_names, base_seed=42):
    """Return a deterministic seed from the sorted batch gene_names.

    All genes in a batch share this seed (one torch.manual_seed call) but
    receive distinct noise via positional indexing into the batched MC draw.
    H0 and H1 share the seed (common random numbers) so the LRT difference
    has lower variance. Reproducibility holds for a fixed input file and
    fixed --batch size; changing batch composition changes the seed."""
    key = "|".join(sorted(str(g) for g in gene_names))
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ int(base_seed)) & 0x7fffffff

# likelihood ratio test
def likelihood_ratio_test(
    expr,
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
    max_iter,
    learning_rate,
    dtype,
    device,
    wandb_flag,
    window,
    tol,
    approx,
    em_iter,
    batch_start,
    prior,
    init,
    kkt,
    grid,
    nb,
    library_list,
    importance,
    const,
    mix,
    h0_step_callback=None,
    h1_step_callback=None,
    grad_clip_norm=None,
    seed_per_gene=True,
    root_mode="stationary",
):
    """
    Hypothesis testing for lineage-specific gene expression change.

    expr: (batch_size, N_sim, n_cells) numpy array
    diverge_list, share_list, epochs_list, beta_list: list of numpy arrays
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    gene_names: list of gene names
    window: number of iterations to check convergence
    tol: convergence tolerance
    approx: approximation method
    em_iter: number of EM iterations
    batch_start: start index of the batch
    prior: prior for alpha
    init: whether to initialize with OU optimization
    kkt: whether to use KKT condition in optimization
    grid: grid search for alpha
    nb: whether to use negative binomial likelihood
    library_list: list of library size normalization factors
    importance: number of importance samples for likelihood
    const: whether to use constant terms in likelihood
    mix: weight for q(z) in the mixture proposal p(z) & q(z) for importance sampling

    Returns: (batch_size, N_sim) numpy array of params and losses for null and alternative models
    """
    # Initialize OU means for torch: q-mean = x
    m_init_tensor = [
        torch.tensor(x, dtype=dtype, device=device) for x in expr
    ]  # list of (batch_size, N_sim, n_cells)

    # Initialize OU parameters
    batch_size, N_sim, _ = m_init_tensor[0].shape
    ou_params_init = torch.ones(
        (batch_size, N_sim, n_regimes + 2), dtype=dtype, device=device
    )

    # Initial run with OU model
    if init:
        ou_params_h0, ou_loss_h0 = ou_optimize_torch(
            ou_params_init,
            2,
            m_init_tensor,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            device,
            max_iter,
            learning_rate,
            wandb_flag,
            gene_names,
            window,
            tol,
            kkt,
            root_mode=root_mode,
        )
        ou_params_init = ou_params_h0

    # Initialize all Lq parameters for torch.
    # The optimizer stores [mean, log_s2] internally; init q-std = 1 (log_s2 ~= 0).
    pois_params_init = [
        torch.cat((m, torch.zeros_like(m)), dim=-1) for m in m_init_tensor
    ]  # list of (batch_size, N_sim, 2*n_cells)

    # Initialize dispersion parameter for negative binomial
    log_r = torch.zeros(
        (batch_size, N_sim), dtype=dtype, device=device
    )  # (batch_size, N_sim), init r=exp(0)=1 (moderate overdispersion)

    # Combine all initial parameters
    init_params = pois_params_init + [log_r] + [ou_params_init]  # list of (batch_size, N_sim, param_dim)
 
    # Convert expression data to torch tensor
    x_tensor = [
        torch.tensor(x, dtype=dtype, device=device) for x in expr
    ]  # list of (batch_size, N_sim, n_cells)

    # Convert librayy size to torch tensor
    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=dtype, device=device) for lib in library_list
    ]  # list of (n_cells,)

    # optimize Lq for null model
    if em_iter == 0: # optimize all params
        h0_params, h0_loss = Lq_optimize_torch_OU(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            prior,
            kkt,
            nb,
            library_list_tensor,
            const,
            step_callback=h0_step_callback,
            grad_clip_norm=grad_clip_norm,
            seed_per_call=(_gene_seed(gene_names) if seed_per_gene else None),
            root_mode=root_mode,
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    else: # EM
        h0_params, h0_loss = run_em(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            em_iter,
            prior,
            kkt,
            nb,
            library_list_tensor,
            const,
            root_mode=root_mode,
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    
    # optimize Lq for alternative model
    if em_iter == 0: # optimize all params
        h1_params, h1_loss = Lq_optimize_torch_OU(
            h0_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            prior,
            kkt,
            nb,
            library_list_tensor,
            const,
            step_callback=h1_step_callback,
            grad_clip_norm=grad_clip_norm,
            seed_per_call=(_gene_seed(gene_names) if seed_per_gene else None),
            root_mode=root_mode,
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    else: # EM
        h1_params, h1_loss = run_em(
            h0_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            em_iter,
            prior,
            kkt,
            nb,
            library_list_tensor,
            const,
            root_mode=root_mode,
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    
    # Optional: grid search for alpha when fixing other parameters
    if grid > 0:
        alphas = [i+1 for i in range(grid)]
        h0_elbos = []
        h1_elbos = []
        for alpha in alphas: # try different initial alpha
            init_params[-1][0, 0, 0] = np.log(alpha)
            h0_params_grid, h0_loss_grid = Lq_optimize_torch_OU(
                init_params,
                1,
                x_tensor,
                gene_names,
                diverge_list_torch,
                share_list_torch,
                epochs_list_torch,
                beta_list_torch,
                1,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                approx,
                None,
                prior,
                kkt,
                nb,
                library_list_tensor,
                const
            )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
            h1_params_grid, h1_loss_grid = Lq_optimize_torch_OU(
                init_params,
                2,
                x_tensor,
                gene_names,
                diverge_list_torch,
                share_list_torch,
                epochs_list_torch,
                beta_list_torch,
                1,
                learning_rate,
                device,
                wandb_flag,
                window,
                tol,
                approx,
                None,
                prior,
                kkt,
                nb,
                library_list_tensor,
                const
            )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
            h0_elbos.append(h0_loss_grid.clone().detach().cpu().numpy()[0, 0])
            h1_elbos.append(h1_loss_grid.clone().detach().cpu().numpy()[0, 0])

        plt.plot(alphas, h0_elbos, marker='o', label='h0 -ELBO', ms=2)
        plt.plot(alphas, h1_elbos, marker='o', label='h1 -ELBO', ms=2)
        plt.xlabel("alpha")
        plt.ylabel("-ELBO")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        best_idx = int(np.argmin(h0_elbos))
        plt.scatter(alphas[best_idx], h0_elbos[best_idx], color="red", zorder=5, label="h0 best alpha")
        best_idx = int(np.argmin(h1_elbos))
        plt.scatter(alphas[best_idx], h1_elbos[best_idx], color="green", zorder=5, label="h1 best alpha")
        plt.legend()
        plt.savefig(f"{batch_start}_elbo.png")
        plt.close()
    
    # importance sampling using elbo as a proposal for more accurate likelihood estimation
    if importance > 0:
        print(f"\nPerforming importance sampling x{importance}...\n")
        h0_loss = importance_sampling(
            h0_params, 
            1, 
            x_tensor, 
            diverge_list_torch, 
            share_list_torch, 
            epochs_list_torch, 
            beta_list_torch, 
            device, 
            nb,
            library_list_tensor,
            importance,
            mix
        )

        h1_loss = importance_sampling(
            h1_params, 
            2, 
            x_tensor, 
            diverge_list_torch, 
            share_list_torch, 
            epochs_list_torch, 
            beta_list_torch, 
            device, 
            nb,
            library_list_tensor,
            importance,
            mix
        )

    # Strip the L2 prior on log_alpha from the returned losses
    if prior > 0:
        h0_log_alpha = h0_params[-1][..., 0]
        h1_log_alpha = h1_params[-1][..., 0]
        h0_loss = h0_loss - 0.5 * prior * h0_log_alpha ** 2
        h1_loss = h1_loss - 0.5 * prior * h1_log_alpha ** 2

    # combine logr and ou params
    h0_model_params = torch.cat((h0_params[-2].unsqueeze(-1), h0_params[-1]), dim=-1)
    h1_model_params = torch.cat((h1_params[-2].unsqueeze(-1), h1_params[-1]), dim=-1)

    # Output variational parameters as mean + std for compatibility with
    # existing q-mean-std files. Internally h*_params store mean + log_s2.
    h0_q_params = [
        _lq_to_mean_std(n).clone().detach().cpu().numpy()
        for n in h0_params[:-2]
    ]
    h1_q_params = [
        _lq_to_mean_std(n).clone().detach().cpu().numpy()
        for n in h1_params[:-2]
    ]

    return h0_model_params.clone().detach().cpu().numpy(), \
        h0_loss.clone().detach().cpu().numpy(), \
        h1_model_params.clone().detach().cpu().numpy(), \
        h1_loss.clone().detach().cpu().numpy(), \
        h0_q_params, h1_q_params

    
