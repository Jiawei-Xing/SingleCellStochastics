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
    std = torch.exp(lq[..., n_cells : 2 * n_cells] / 2)
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
    return (int(h[:8], 16) ^ int(base_seed)) & 0x7FFFFFFF


# likelihood ratio test
def _expand_params_for_alt(params, n_regimes):
    """Expand H0 parameter layout into an H1 layout with ``n_regimes`` thetas."""
    expanded = [p.clone().detach() for p in params[:-1]]
    ou_params = params[-1]
    if ou_params.shape[-1] == n_regimes + 2:
        expanded.append(ou_params.clone().detach())
        return expanded

    new_ou = torch.ones(
        (*ou_params.shape[:-1], n_regimes + 2),
        dtype=ou_params.dtype,
        device=ou_params.device,
    )
    new_ou[..., 0:2] = ou_params[..., 0:2]
    theta = ou_params[..., 2:]
    if theta.shape[-1] == 0:
        fill = torch.ones_like(new_ou[..., 2:3])
    else:
        fill = theta.mean(dim=-1, keepdim=True)
    new_ou[..., 2:] = fill.expand(*fill.shape[:-1], n_regimes)
    expanded.append(new_ou)
    return expanded


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
    h0_n_regimes=None,
    h0_beta_list_torch=None,
):
    """
    Hypothesis testing for lineage-specific gene expression change.

    By default this preserves the legacy LRT: H0 has one shared theta and H1
    has one theta per regime in ``beta_list_torch``. If ``h0_n_regimes`` and
    ``h0_beta_list_torch`` are supplied, H0 is fit with that collapsed
    multi-theta regime partition using the same OU likelihood as H1.
    """
    legacy_one_theta_h0 = h0_n_regimes is None
    h0_n_regimes = 1 if legacy_one_theta_h0 else h0_n_regimes
    h0_mode = 1 if legacy_one_theta_h0 else 2
    h0_beta_torch = beta_list_torch if legacy_one_theta_h0 else h0_beta_list_torch
    if h0_beta_torch is None:
        raise ValueError("h0_beta_list_torch is required when h0_n_regimes is set.")

    # Initialize OU means for torch: q-mean = x
    m_init_tensor = [
        torch.tensor(x, dtype=dtype, device=device) for x in expr
    ]  # list of (batch_size, N_sim, n_cells)

    # Initialize OU parameters. Legacy H0 keeps the old wider parameter vector
    # so old runs are unchanged; multi-theta H0 uses its own theta count.
    batch_size, N_sim, _ = m_init_tensor[0].shape
    h0_param_dim = n_regimes + 2 if legacy_one_theta_h0 else h0_n_regimes + 2
    ou_params_h0_init = torch.ones(
        (batch_size, N_sim, h0_param_dim), dtype=dtype, device=device
    )

    # Initial run with OU model
    if init:
        ou_params_h0, ou_loss_h0 = ou_optimize_torch(
            ou_params_h0_init,
            h0_mode,
            m_init_tensor,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            h0_beta_torch,
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
        ou_params_h0_init = ou_params_h0

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
    init_params_h0 = pois_params_init + [log_r] + [ou_params_h0_init]

    # Convert expression data to torch tensor
    x_tensor = [
        torch.tensor(x, dtype=dtype, device=device) for x in expr
    ]  # list of (batch_size, N_sim, n_cells)

    # Convert library size to torch tensor
    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=dtype, device=device)
        for lib in library_list
    ]  # list of (n_cells,)

    # optimize Lq for null model
    if em_iter == 0:  # optimize all params
        h0_params, h0_loss = Lq_optimize_torch_OU(
            init_params_h0,
            h0_mode,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            h0_beta_torch,
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
        )
    else:  # EM
        h0_params, h0_loss = run_em(
            init_params_h0,
            h0_mode,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            h0_beta_torch,
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
        )

    h1_init_params = (
        h0_params
        if legacy_one_theta_h0
        else _expand_params_for_alt(h0_params, n_regimes)
    )

    # optimize Lq for alternative model
    if em_iter == 0:  # optimize all params
        h1_params, h1_loss = Lq_optimize_torch_OU(
            h1_init_params,
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
        )
    else:  # EM
        h1_params, h1_loss = run_em(
            h1_init_params,
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
        )

    # Optional: grid search for alpha when fixing other parameters
    if grid > 0:
        alphas = [i + 1 for i in range(grid)]
        h0_elbos = []
        h1_elbos = []
        for alpha in alphas:  # try different initial alpha
            h0_grid_init = [p.clone().detach() for p in init_params_h0]
            h0_grid_init[-1][0, 0, 0] = np.log(alpha)
            h1_grid_init = _expand_params_for_alt(h0_grid_init, n_regimes)
            h0_params_grid, h0_loss_grid = Lq_optimize_torch_OU(
                h0_grid_init,
                h0_mode,
                x_tensor,
                gene_names,
                diverge_list_torch,
                share_list_torch,
                epochs_list_torch,
                h0_beta_torch,
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
                const,
            )
            h1_params_grid, h1_loss_grid = Lq_optimize_torch_OU(
                h1_grid_init,
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
                const,
            )
            h0_elbos.append(h0_loss_grid.clone().detach().cpu().numpy()[0, 0])
            h1_elbos.append(h1_loss_grid.clone().detach().cpu().numpy()[0, 0])

        plt.plot(alphas, h0_elbos, marker="o", label="h0 -ELBO", ms=2)
        plt.plot(alphas, h1_elbos, marker="o", label="h1 -ELBO", ms=2)
        plt.xlabel("alpha")
        plt.ylabel("-ELBO")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        best_idx = int(np.argmin(h0_elbos))
        plt.scatter(
            alphas[best_idx],
            h0_elbos[best_idx],
            color="red",
            zorder=5,
            label="h0 best alpha",
        )
        best_idx = int(np.argmin(h1_elbos))
        plt.scatter(
            alphas[best_idx],
            h1_elbos[best_idx],
            color="green",
            zorder=5,
            label="h1 best alpha",
        )
        plt.legend()
        plt.savefig(f"{batch_start}_elbo.png")
        plt.close()

    # importance sampling using elbo as a proposal for more accurate likelihood estimation
    if importance > 0:
        print(f"\nPerforming importance sampling x{importance}...\n")
        h0_loss = importance_sampling(
            h0_params,
            h0_mode,
            x_tensor,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            h0_beta_torch,
            device,
            nb,
            library_list_tensor,
            importance,
            mix,
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
            mix,
        )

    # Strip the L2 prior on log_alpha from the returned losses
    if prior > 0:
        h0_log_alpha = h0_params[-1][..., 0]
        h1_log_alpha = h1_params[-1][..., 0]
        h0_loss = h0_loss - 0.5 * prior * h0_log_alpha**2
        h1_loss = h1_loss - 0.5 * prior * h1_log_alpha**2

    # combine logr and ou params
    h0_model_params = torch.cat((h0_params[-2].unsqueeze(-1), h0_params[-1]), dim=-1)
    h1_model_params = torch.cat((h1_params[-2].unsqueeze(-1), h1_params[-1]), dim=-1)

    # Output variational parameters as mean + std for compatibility with
    # existing q-mean-std files. Internally h*_params store mean + log_s2.
    h0_q_params = [
        _lq_to_mean_std(n).clone().detach().cpu().numpy() for n in h0_params[:-2]
    ]
    h1_q_params = [
        _lq_to_mean_std(n).clone().detach().cpu().numpy() for n in h1_params[:-2]
    ]

    return (
        h0_model_params.clone().detach().cpu().numpy(),
        h0_loss.clone().detach().cpu().numpy(),
        h1_model_params.clone().detach().cpu().numpy(),
        h1_loss.clone().detach().cpu().numpy(),
        h0_q_params,
        h1_q_params,
    )
