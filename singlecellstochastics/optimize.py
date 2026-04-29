"""Optimization routines for BM/OU likelihoods and variational objectives."""

from joblib import Parallel, delayed
import numpy as np
import torch
from scipy.optimize import minimize
import wandb

from .defaults import (
    DEFAULT_LOG_ALPHA_CLAMP,
    DEFAULT_LOG_R_CLAMP,
    DEFAULT_LOG_S2_CLAMP,
)
from .likelihood import ou_neg_log_lik_numpy, ou_neg_log_lik_numpy_kkt
from .likelihood import ou_neg_log_lik_torch, ou_neg_log_lik_torch_kkt, bm_neg_log_lik_torch_kkt
from .elbo import Lq_neg_log_lik_torch
from .trace import TRACE


# optimize OU likelihood with SciPy
def ou_optimize_scipy(
    params_init,
    mode,
    expr_list,
    diverge_list,
    share_list,
    epochs_list,
    beta_list,
    device,
):
    """
    Optimize OU likelihood with SciPy L-BFGS-B.
    Returns OU parameters for each batch and sim.
    Used for initializing OU parameters for ELBO with expression data.

    params_init: OU parameters (alpha, sigma2, theta0, theta1, ..., theta_n)
    mode: 1 for H0, 2 for H1
    expr: list of expression data (batch_size, N_sim, n_cells)
    """
    batch_size, N_sim, _ = expr_list[0].shape  # same for all trees

    # optimize OU for batch i, sim j
    def fit_one(i, j):
        expr = [x[i, j, :] for x in expr_list]
        res = minimize(
            ou_neg_log_lik_numpy_kkt,
            params_init,
            args=(mode, expr, diverge_list, share_list, epochs_list, beta_list),
            bounds=[(1e-6, None)]*2 + [(None, None)]*(len(params_init)-2),
            method="L-BFGS-B",
        )
        print(f"\nh{mode-1} OU scipy init params: {res.x}")
        return i, j, res.x

    # parallel execution
    results = Parallel(n_jobs=-1)(
        delayed(fit_one)(i, j) for i in range(batch_size) for j in range(N_sim)
    )

    # Convert results to tensor
    result_tensor = torch.empty(
        (batch_size, N_sim, len(params_init)), dtype=torch.float32, device=device
    )
    for i, j, x in results:
        result_tensor[i, j, :] = torch.tensor(x, dtype=torch.float32, device=device)

    return result_tensor


# optimize OU likelihood with PyTorch Adam
def ou_optimize_torch(
    params_init,
    mode,
    expr_list_torch,
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
    log_alpha_clamp=None,
    root_mode="stationary",
):
    """
    Optimize the OU-Gaussian likelihood with PyTorch Adam.
    Returns OU parameters for each batch and sim.
    Used for initializing OU parameters for ELBO with expression data.

    params_init: OU parameters (alpha, sigma2, theta0, theta1, ..., theta_n)
    mode: 1 for H0, 2 for H1
    expr_list_torch: list of expression data (batch_size, N_sim, n_cells)
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    device: device to use
    max_iter: maximum number of iterations
    learning_rate: learning rate for Adam
    wandb_flag: whether to use wandb
    gene_names: list of gene names
    window: number of recent iterations to check for convergence
    tol: convergence tolerance
    kkt: whether to use KKT condition for OU likelihood
    """       
    batch_size, N_sim, _ = expr_list_torch[0].shape
    n_trees = len(expr_list_torch)

    # Initialize OU parameters for torch optimization
    ou_params = [
        params_init[:, :, 0:1].clone().detach(),
        params_init[:, :, 1:].clone().detach()
    ] # [alpha, others]

    # Set requires_grad based on kkt
    ou_params[0] = ou_params[0].requires_grad_(True) # optimize alpha
    if not kkt:
        ou_params[1] = ou_params[1].requires_grad_(True) # optimize all

    best_loss = torch.full(
        (batch_size, N_sim), float("inf"), dtype=params_init.dtype, device=device
    )
    _log_alpha_clamp = (
        log_alpha_clamp if log_alpha_clamp is not None else DEFAULT_LOG_ALPHA_CLAMP
    )
    with torch.no_grad():
        lo, hi = _log_alpha_clamp
        ou_params[0].clamp_(float(lo), float(hi))

    # Track best parameters and loss
    best_params = [
        p.clone().detach() for p in ou_params
    ] # [alpha, others]

    optimizer = torch.optim.Adam(ou_params, lr=learning_rate)

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros(
        (batch_size, N_sim), dtype=torch.bool, device=device
    )
    nan_streak = torch.zeros(
        (batch_size, N_sim), dtype=torch.long, device=device
    )
    nan_patience = 50  # mark as failed after this many consecutive NaN iterations

    for n in range(max_iter):
        optimizer.zero_grad()

        # initialize loss matrix and store indices of not yet converged genes
        loss_matrix = torch.zeros(batch_size, N_sim, n_trees, device=device)
        active_batch = (~converged_mask).any(dim=1)

        # get loss for each tree for not yet converged genes
        for i in range(n_trees):
            expr_tensor = expr_list_torch[i][active_batch, :, :] # only use not yet converged genes
            ou_params_tensor = [
                p[active_batch, :, :] for p in ou_params
            ] # only optimize not yet converged genes
            
            sigma2_q = torch.zeros_like(expr_tensor) # trace term = 0
            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]

            if kkt:
                loss, sigma, theta = ou_neg_log_lik_torch_kkt(
                    ou_params_tensor,
                    sigma2_q,
                    mode,
                    expr_tensor,
                    diverge,
                    share,
                    epochs,
                    beta,
                    device,
                    root_mode=root_mode,
                )
            else:
                loss = ou_neg_log_lik_torch(
                    ou_params_tensor,
                    sigma2_q,
                    mode,
                    expr_tensor,
                    diverge,
                    share,
                    epochs,
                    beta,
                    device,
                    root_mode=root_mode,
                )
            loss_matrix[active_batch, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        #average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(
        #    torch.tensor(n_trees, device=device, dtype=loss.dtype)
        #)  # (batch_size, N_sim)
        average_loss = loss_matrix.sum(dim=2) # (batch_size, N_sim)

        # update best params for not yet converged genes
        with torch.no_grad():
            mask = (average_loss < best_loss) & ~converged_mask & torch.isfinite(average_loss)
            best_loss = torch.where(
                mask,
                average_loss.clone().detach(),
                best_loss
            ) # update best loss

            # update sigma and theta from kkt (guard against NaN from lstsq)
            if kkt:
                other_params = torch.cat(
                    (sigma.unsqueeze(-1).clone().detach(), # (B,S,1)
                    theta.clone().detach()), dim=-1 # (B,S,n_regimes)
                )
                nan_free = ~torch.isnan(other_params).any(dim=-1, keepdim=True)
                if mode == 1:
                    current = ou_params[1][active_batch, :, :2].clone()
                    ou_params[1][active_batch, :, :2] = torch.where(nan_free, other_params, current)
                else:
                    current = ou_params[1][active_batch, :, :].clone()
                    ou_params[1][active_batch, :, :] = torch.where(nan_free, other_params, current)

            best_params = [
                torch.where(
                    mask.unsqueeze(-1),  # (B,S,1)
                    ou_params[i].clone().detach(), # (B,S,n)
                    best_params[i]
                ) for i in range(len(ou_params))
            ] # update best sigma and theta

            if wandb_flag:
                wandb.log({
                        "iter": n,
                        f"{gene_names[0]}_h{mode-1}_ou_loss": best_loss[0, 0], 
                })

        # Store loss history
        loss_history.append(average_loss.clone().detach())

        # Early exit for persistent NaN/inf losses.
        # Cumulative (not consecutive) so genes with intermittent NaN/inf
        # which the old streak logic reset on every finite iteration and
        # thus never terminated still fail out before wasting max_iter.
        with torch.no_grad():
            is_bad = ~torch.isfinite(average_loss)
            nan_streak = torch.where(is_bad & ~converged_mask, nan_streak + 1, nan_streak)
            nan_failed = (nan_streak >= nan_patience) & ~converged_mask
            if nan_failed.any():
                converged_mask = converged_mask | nan_failed
                if converged_mask.all():
                    break

        # Check for convergence
        if n >= window:
            # Check if loss has stabilized within window
            start_loss = loss_history[-window]
            denom = start_loss.abs().clamp_min(1.0)
            relative_decrease = torch.abs(start_loss - best_loss) / denom
            
            # Mark as newly converged if variance is below tolerance
            newly_converged = (relative_decrease < tol) & ~converged_mask

            # update converged mask if found newly converged genes
            if newly_converged.any():
                converged_mask = converged_mask | newly_converged # update converged mask
                if converged_mask.all(): # break if all genes have converged
                    break

        # backpropagate for not yet converged genes, skipping non-finite
        # losses (NaN or inf) so one bad gene can't poison the shared Adam
        # state for the whole batch
        active_loss_flat = average_loss[~converged_mask]
        valid_mask = torch.isfinite(active_loss_flat)
        if valid_mask.any():
            loss = active_loss_flat[valid_mask].sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                lo, hi = _log_alpha_clamp
                ou_params[0].clamp_(float(lo), float(hi))

    # warning if not all genes have converged
    print(f"\nChecking convergence for h{mode-1} OU init...")
    nan_failed_genes = (best_loss == float("inf"))
    non_converged = ~converged_mask
    slow_genes = non_converged & ~nan_failed_genes
    if nan_failed_genes.any():
        print(f"\nWARNING: {nan_failed_genes.sum().item()} gene(s) failed (persistent NaN loss):")
        for i in range(batch_size):
            for j in range(N_sim):
                if nan_failed_genes[i, j]:
                    print(f"   {gene_names[i]}: NaN loss (skipped after {nan_patience} iters)")
    if slow_genes.any():
        print(f"\nWARNING: {slow_genes.sum().item()}/{batch_size} gene(s) did not converge:")
        print(f"   Gene Name | Relative Decrease | Final Loss")
        print(f"   {'-' * 25}")
        for i in range(batch_size):
            for j in range(N_sim):
                if slow_genes[i, j]:
                    rd = relative_decrease[i, j].item() if 'relative_decrease' in locals() else float('nan')
                    print(
                        f"   {gene_names[i]}: {rd:.2e} | " +
                        f"{best_loss[i, j].item():.2e}"
                    )
        print(f"\nRecommendations for non-converged genes:")
        print(f"   - Increase max_iter (current: {max_iter})")
        print(f"   - Increase learning_rate (current: {learning_rate})")
        print(f"   - Increase tol (current: {tol})")
        print(f"   - Decrease window (current: {window})")
        print(f"   - Check data quality for these genes")

    best_params = torch.cat(
        [p for p in best_params], dim=-1
     ) # (batch_size, N_sim, n_params)

    return best_params, best_loss


# optimize ELBO with Adam (OU)
def Lq_optimize_torch_OU(
    params,
    mode,
    x_tensor_list,
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
    em,
    prior,
    kkt,
    nb,
    library_list_tensor,
    const,
    fix_alpha=False,
    step_callback=None,
    grad_clip_norm=None,
    log_alpha_clamp=None,
    log_r_clamp=None,
    log_s2_clamp=None,
    seed_per_call=None,
    ou_lambda_mode=1,
    root_mode="stationary",
):
    """
    Optimize ELBO with PyTorch Adam.

    Numerical-safety knobs:
    - grad_clip_norm: if given, apply torch.nn.utils.clip_grad_norm_ to
      each parameter tensor with this max norm. Bounds the size of any
      single bad MC step.
    - log_alpha_clamp: (lo, hi) tuple. Clamp log_alpha post-step to this
      range. Prevents alpha -> 0 (BM degeneracy -> KKT NaN) and alpha -> inf
      (V -> 0 -> singular Cholesky). Defaults to (-5, 5), except when
      fix_alpha=True.
    - log_r_clamp: (lo, hi) tuple for NB dispersion log_r clamp. Applied
      pre-loop, post-step, and pre-KKT. Defaults to (-5, 10) -> r in
      [~0.007, ~22026]. Upper end allows near-Poisson regimes for low-noise
      high-count genes (housekeeping, ribosomal). log_r has no gradient
      signal in the Poisson limit, so Adam is self-limiting at high r.
    - log_s2_clamp: (lo, hi) tuple for variational log-variance clamp.
      Defaults to (-10, 10) -> std in [~0.007, ~148]. Keeps q-variance
      strictly positive and bounded for the MC expectations.
    - seed_per_call: integer seed applied via torch.manual_seed at the
      start of this call. Stabilises the MC sampling against batch-order
      effects (a gene then gets the same MC sequence regardless of which
      batch it lands in).

    params: List of (batch_size, N_sim, all_param_dim)
            [Lq_params tree1, Lq_params tree2, ..., Lq_params treeN, logr, shared OU_params]
    mode: 1 for H0, 2 for H1
    x_tensor: list of (batch_size, N_sim, n_cells)
    gene_names: list of gene names
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    max_iter: maximum number of iterations
    learning_rate: learning rate for Adam
    device: device to use
    wandb_flag: whether to use wandb
    window: number of recent iterations to check for convergence
    tol: convergence tolerance
    approx: approximation method for Poisson likelihood expectation
    em: "e" for E-step, "m" for M-step, None for both together
    prior: L2 regularization strength for OU alpha
    kkt: whether to use KKT condition for OU likelihood
    nb: whether to use negative binomial likelihood
    lib: library size normalization
    const: whether to include constant terms in likelihood
    ou_lambda_mode: Pagel's lambda mode for the OU process.
        0 -> lambda fixed at 0 (star-tree / no phylogenetic signal),
        1 -> lambda fixed at 1 (standard OU, default),
        2 -> lambda optimized via sigmoid(params[ou_lambda_idx]).
        Mode 2 also adds ou_lambda to the trainable params and to the
        returned best_params tuple.

    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    if seed_per_call is not None:
        torch.manual_seed(int(seed_per_call))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed_per_call))

    batch_size, N_sim, _ = params[0].shape
    n_trees = len(x_tensor_list)

    has_ou_lambda_param = len(params) == n_trees + 3
    return_ou_lambda = has_ou_lambda_param or ou_lambda_mode != 1
    base_params = params[:-2] if has_ou_lambda_param else params[:-1]
    if has_ou_lambda_param:
        ou_lambda_init = params[-2].clone().detach()
    else:
        ou_lambda_init = torch.ones((batch_size, N_sim), dtype=params[0].dtype, device=device)

    # Initialize parameters for optimization.
    # Layout: [Lq tree1, ..., Lq treeN, logr, Pagel lambda, alpha, other OU]
    params_tensor = [
        p.clone().detach() for p in base_params
    ] + [
        ou_lambda_init,
        params[-1][:, :, 0:1].clone().detach(),
        params[-1][:, :, 1:].clone().detach()
    ]
    log_r_idx = n_trees
    ou_lambda_idx = n_trees + 1
    log_alpha_idx = n_trees + 2
    other_ou_idx = n_trees + 3

    # Set requires_grad based on EM mode
    if em == "e":
        # E-step: optimize ELBO with fixed OU parameters
        for i in range(log_r_idx + 1):
            params_tensor[i] = params_tensor[i].requires_grad_(True)
    elif em == "m":
        # M-step: optimize OU parameters with fixed ELBO
        if ou_lambda_mode == 2:
            params_tensor[ou_lambda_idx] = params_tensor[ou_lambda_idx].requires_grad_(True)
        if not fix_alpha:
            params_tensor[log_alpha_idx] = params_tensor[log_alpha_idx].requires_grad_(True) # alpha
        if not kkt:
            params_tensor[other_ou_idx] = params_tensor[other_ou_idx].requires_grad_(True) # other OU
    else:
        # optimize both ELBO and OU parameters
        for i in range(log_r_idx + 1):
            params_tensor[i] = params_tensor[i].requires_grad_(True)
        if ou_lambda_mode == 2:
            params_tensor[ou_lambda_idx] = params_tensor[ou_lambda_idx].requires_grad_(True)
        if not fix_alpha:
            params_tensor[log_alpha_idx] = params_tensor[log_alpha_idx].requires_grad_(True)
        if fix_alpha:
            params_tensor[log_alpha_idx] = params_tensor[log_alpha_idx].requires_grad_(False) # freeze alpha
        if not kkt:
            params_tensor[other_ou_idx] = params_tensor[other_ou_idx].requires_grad_(True) # other OU
    
    _log_alpha_clamp = None
    if not fix_alpha:
        _log_alpha_clamp = (
            log_alpha_clamp
            if log_alpha_clamp is not None
            else DEFAULT_LOG_ALPHA_CLAMP
        )
    _log_r_lo, _log_r_hi = (
        log_r_clamp if log_r_clamp is not None else DEFAULT_LOG_R_CLAMP
    )
    _log_s2_lo, _log_s2_hi = (
        log_s2_clamp if log_s2_clamp is not None else DEFAULT_LOG_S2_CLAMP
    )

    def _clamp_lq_log_s2_(param_list):
        with torch.no_grad():
            for lq in param_list[:n_trees]:
                n_cells = lq.shape[-1] // 2
                lq[..., n_cells:2 * n_cells].clamp_(_log_s2_lo, _log_s2_hi)

    # Pre-loop clamps are domain bounds
    with torch.no_grad():
        if nb:
            params_tensor[log_r_idx].clamp_(_log_r_lo, _log_r_hi)
        if _log_alpha_clamp is not None:
            lo, hi = _log_alpha_clamp
            params_tensor[log_alpha_idx].clamp_(float(lo), float(hi))
    _clamp_lq_log_s2_(params_tensor)

    # Track best parameters for all parameter types
    best_params = [p.clone().detach() for p in params_tensor]
    best_loss = torch.full(
        (batch_size, N_sim), float("inf"), dtype=params[0].dtype, device=device
    )
    optimizer = torch.optim.Adam(
        [p for p in params_tensor if p.requires_grad], lr=learning_rate
    )

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros(
        (batch_size, N_sim), dtype=torch.bool, device=device
    )
    nan_streak = torch.zeros(
        (batch_size, N_sim), dtype=torch.long, device=device
    )
    nan_patience = 50  # mark as failed after this many consecutive NaN iterations

    tracing = step_callback is not None
    if tracing:
        TRACE.enabled = True

    for n in range(max_iter):
        optimizer.zero_grad()
        active_batch = (~converged_mask).any(dim=1)  # (batch_size,)

        # get loss for each tree for not yet converged genes
        active_loss_list = []
        # Capture pre-forward param snapshot for trace (post-clamp value used
        # in this step's forward pass)
        if tracing:
            TRACE.reset()
            TRACE.write("step", n)
            TRACE.write("log_r_pre_forward", float(params_tensor[log_r_idx].reshape(-1)[0].detach().cpu()))
            TRACE.write("log_alpha_pre_forward", float(params_tensor[log_alpha_idx].reshape(-1)[0].detach().cpu()))

        for i in range(n_trees):
            if kkt:
                loss, sigma, theta = Lq_neg_log_lik_torch(
                    params_tensor[i][active_batch],
                    params_tensor[log_r_idx][active_batch],
                    [params_tensor[log_alpha_idx][active_batch], params_tensor[other_ou_idx][active_batch]],
                    mode,
                    x_tensor_list[i][active_batch],
                    diverge_list_torch[i],
                    share_list_torch[i],
                    epochs_list_torch[i],
                    beta_list_torch[i],
                    device,
                    approx,
                    prior,
                    kkt,
                    nb,
                    library_list_tensor[i],
                    const,
                    fix_alpha=fix_alpha,
                    ou_lambda=params_tensor[ou_lambda_idx][active_batch],
                    ou_lambda_mode=ou_lambda_mode,
                    log_s2_clamp=(_log_s2_lo, _log_s2_hi),
                    root_mode=root_mode,
                )
            else:
                loss = Lq_neg_log_lik_torch(
                    params_tensor[i][active_batch],
                    params_tensor[log_r_idx][active_batch],
                    [params_tensor[log_alpha_idx][active_batch], params_tensor[other_ou_idx][active_batch]],
                    mode,
                    x_tensor_list[i][active_batch],
                    diverge_list_torch[i],
                    share_list_torch[i],
                    epochs_list_torch[i],
                    beta_list_torch[i],
                    device,
                    approx,
                    prior,
                    kkt,
                    nb,
                    library_list_tensor[i],
                    const,
                    fix_alpha=fix_alpha,
                    ou_lambda=params_tensor[ou_lambda_idx][active_batch],
                    ou_lambda_mode=ou_lambda_mode,
                    log_s2_clamp=(_log_s2_lo, _log_s2_hi),
                    root_mode=root_mode,
                )
            active_loss_list.append(loss)

        active_joint_loss = torch.stack(active_loss_list, dim=-1).sum(dim=-1)  # (num_active, N_sim)

        # update best params for not yet converged genes
        with torch.no_grad():
            joint_loss = torch.full((batch_size, N_sim), float("inf"), dtype=params[0].dtype, device=device)
            joint_loss[active_batch] = active_joint_loss.clone().detach()

            mask = (joint_loss < best_loss) & ~converged_mask
            best_loss = torch.where(
                mask,
                joint_loss,
                best_loss
            ) # update best loss

            # save best params (skip params_tensor[-1] when kkt: sigma/theta
            # are reconstructed after the loop from the best alpha/Lq params)
            for i, param in enumerate(params_tensor):
                if kkt and i == other_ou_idx:
                    continue  # skip sigma/theta, reconstruct later
                current_mask = mask.unsqueeze(-1) if param.dim() > 2 else mask
                best_params[i] = torch.where(
                    current_mask,  # (B,S,1) or (B,S)
                    param.clone().detach(), # (B,S,all_param_dim)
                    best_params[i]
                ) # update best Lq params

            if wandb_flag:
                # plot loss of first gene in batch
                if em is None:
                    wandb.log({
                            "iter": n,
                            f"{gene_names[0]}_ou{mode}_elbo_loss": best_loss[0, 0],
                    })
                else:
                    wandb.log({
                            "iter": n,
                            f"{gene_names[0]}_ou{mode}_{em}_loss": best_loss[0, 0],
                    })

        # Store loss history
        loss_history.append(joint_loss.clone().detach())

        # Early exit for persistent NaN/inf losses
        with torch.no_grad():
            is_bad = ~torch.isfinite(joint_loss)
            nan_streak = torch.where(is_bad & ~converged_mask, nan_streak + 1, nan_streak)
            nan_failed = (nan_streak >= nan_patience) & ~converged_mask
            if nan_failed.any():
                converged_mask = converged_mask | nan_failed
                if converged_mask.all():
                    break

        # backpropagate for not yet converged genes, skipping non-finite
        # losses so one bad gene can't poison the shared Adam state
        active_loss_flat = active_joint_loss[~converged_mask[active_batch]]
        valid_mask = torch.isfinite(active_loss_flat)
        if valid_mask.any():
            loss = active_loss_flat[valid_mask].sum()
            loss.backward()

            if tracing:
                for pname, pidx in [("log_r", log_r_idx), ("log_alpha", log_alpha_idx), ("lq0", 0)]:
                    pt = params_tensor[pidx]
                    if pt.grad is not None:
                        n_nan = int(torch.isnan(pt.grad).sum().detach().cpu())
                        n_inf = int(torch.isinf(pt.grad).sum().detach().cpu())
                        # Largest |grad| among FINITE entries (gives a sense of
                        # gradient magnitude even when a few entries are non-finite)
                        finite_mask = torch.isfinite(pt.grad)
                        if finite_mask.any():
                            g_abs_max = float(pt.grad[finite_mask].abs().max().detach().cpu())
                        else:
                            g_abs_max = float("nan")
                        TRACE.write(f"grad_{pname}_n_nan_raw", n_nan)
                        TRACE.write(f"grad_{pname}_n_inf_raw", n_inf)
                        TRACE.write(f"grad_{pname}_abs_max_finite", g_abs_max)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params_tensor if p.grad is not None],
                    max_norm=float(grad_clip_norm),
                )

            if tracing:
                # capture grads + adam state BEFORE step (Adam mutates m,v
                # in-place during step). Index 0,0 since trace is single-gene.
                grad_log_r = params_tensor[log_r_idx].grad
                grad_log_alpha = params_tensor[log_alpha_idx].grad
                grad_lq0 = params_tensor[0].grad
                if grad_log_r is not None:
                    TRACE.write("grad_log_r", float(grad_log_r.reshape(-1)[0].detach().cpu()))
                    n_nan_g = int(torch.isnan(grad_log_r).sum().detach().cpu())
                    TRACE.write("grad_log_r_n_nan", n_nan_g)
                if grad_log_alpha is not None:
                    TRACE.write("grad_log_alpha", float(grad_log_alpha.reshape(-1)[0].detach().cpu()))
                if grad_lq0 is not None:
                    TRACE.write("grad_lq_max_abs", float(grad_lq0.abs().max().detach().cpu()))
                    nan_g, inf_g = int(torch.isnan(grad_lq0).sum().detach().cpu()), int(torch.isinf(grad_lq0).sum().detach().cpu())
                    TRACE.write("grad_lq_n_nan", nan_g)
                    TRACE.write("grad_lq_n_inf", inf_g)

                # Adam state for log_r
                state = optimizer.state.get(params_tensor[log_r_idx], {})
                if "exp_avg" in state:
                    TRACE.write("adam_m_log_r", float(state["exp_avg"].reshape(-1)[0].detach().cpu()))
                if "exp_avg_sq" in state:
                    TRACE.write("adam_v_log_r", float(state["exp_avg_sq"].reshape(-1)[0].detach().cpu()))

            optimizer.step()

            if tracing:
                # capture POST-Adam, PRE-clamp value of log_r
                TRACE.write("log_r_post_adam_pre_clamp", float(params_tensor[log_r_idx].reshape(-1)[0].detach().cpu()))
                # NaN-genesis: log whether each param tensor has any non-finite
                # entries AFTER the optimizer step (before any clamp / sanitize)
                for pname, pidx in [("log_r", log_r_idx), ("log_alpha", log_alpha_idx), ("lq0", 0)]:
                    pt = params_tensor[pidx]
                    n_nan = int(torch.isnan(pt).sum().detach().cpu())
                    n_inf = int(torch.isinf(pt).sum().detach().cpu())
                    TRACE.write(f"param_{pname}_n_nan_post_step", n_nan)
                    TRACE.write(f"param_{pname}_n_inf_post_step", n_inf)
                # Also log post-step log_alpha value for NaN-genesis analysis
                TRACE.write("log_alpha_post_step", float(params_tensor[log_alpha_idx].reshape(-1)[0].detach().cpu()))

        # clamp log_r post-step to prevent NaN/overflow
        if nb:
            with torch.no_grad():
                params_tensor[log_r_idx].clamp_(_log_r_lo, _log_r_hi)

        if _log_alpha_clamp is not None:
            lo, hi = _log_alpha_clamp
            with torch.no_grad():
                params_tensor[log_alpha_idx].clamp_(float(lo), float(hi))
        _clamp_lq_log_s2_(params_tensor)

        if tracing:
            TRACE.write("log_r_post_clamp", float(params_tensor[log_r_idx].reshape(-1)[0].detach().cpu()))
            try:
                step_callback(n, dict(TRACE.diag))
            except Exception as e:
                print(f"[trace] step_callback raised at step {n}: {e}")

        # check convergence in window
        if n >= window:
            # Check if loss has stabilized within window
            start_loss = loss_history[-window]
            denom = torch.maximum(
                start_loss.abs(), torch.tensor(1.0, dtype=loss.dtype, device=device)
            )
            relative_decrease = torch.abs(start_loss - best_loss) / denom

            # Mark as newly converged if variance is below tolerance
            newly_converged = (relative_decrease < tol) & ~converged_mask

            # update converged mask if found newly converged genes
            if newly_converged.any():
                converged_mask = converged_mask | newly_converged # update converged mask
                if converged_mask.all(): # break if all genes have converged
                    break

    # warning if not all genes have converged
    print(f"\nChecking convergence for ou{mode} ELBO...")
    nan_failed_genes = (best_loss == float("inf"))
    non_converged = ~converged_mask
    slow_genes = non_converged & ~nan_failed_genes
    if nan_failed_genes.any():
        print(f"\nWARNING: {nan_failed_genes.sum().item()} gene(s) failed (persistent NaN loss):")
        for i in range(batch_size):
            for j in range(N_sim):
                if nan_failed_genes[i, j]:
                    print(f"   {gene_names[i]}: NaN loss (skipped after {nan_patience} iters)")
    if slow_genes.any():
        print(f"\nWARNING: {slow_genes.sum().item()}/{batch_size} gene(s) did not converge:")
        print(f"   Gene Name | Relative Decrease | Final Loss")
        print(f"   {'-' * 25}")
        for i in range(batch_size):
            for j in range(N_sim):
                if slow_genes[i, j]:
                    rd = relative_decrease[i, j].item() if 'relative_decrease' in locals() else float('nan')
                    print(
                        f"   {gene_names[i]}: {rd:.2e} | " +
                        f"{best_loss[i, j].item():.2e}"
                    )
        print(f"\nRecommendations for non-converged genes:")
        print(f"   - Increase max_iter (current: {max_iter})")
        print(f"   - Increase learning_rate (current: {learning_rate})")
        print(f"   - Increase tol (current: {tol})")
        print(f"   - Decrease window (current: {window})")
        print(f"   - Check data quality for these genes")

    # Pre-KKT clamps are domain bounds only; no NaN-to-zero parameter repair.
    with torch.no_grad():
        if nb:
            best_params[log_r_idx].clamp_(_log_r_lo, _log_r_hi)
        if _log_alpha_clamp is not None:
            lo, hi = _log_alpha_clamp
            best_params[log_alpha_idx].clamp_(float(lo), float(hi))
    _clamp_lq_log_s2_(best_params)

    # Reconstruct sigma/theta from best alpha/Lq params via KKT analytical solution.
    if kkt:
        with torch.no_grad():
            for i in range(n_trees):
                _, sigma, theta = Lq_neg_log_lik_torch(
                    best_params[i],
                    best_params[log_r_idx],
                    [best_params[log_alpha_idx], best_params[other_ou_idx]],
                    mode,
                    x_tensor_list[i],
                    diverge_list_torch[i],
                    share_list_torch[i],
                    epochs_list_torch[i],
                    beta_list_torch[i],
                    device,
                    approx,
                    prior,
                    kkt,
                    nb,
                    library_list_tensor[i],
                    const,
                    fix_alpha=fix_alpha,
                    ou_lambda=best_params[ou_lambda_idx],
                    ou_lambda_mode=ou_lambda_mode,
                    log_s2_clamp=(_log_s2_lo, _log_s2_hi),
                    root_mode=root_mode,
                )
            other_params = torch.cat(
                (sigma.unsqueeze(-1), theta), dim=-1
            )
            if mode == 1:
                best_params[other_ou_idx][..., :2] = other_params
            else:
                best_params[other_ou_idx] = other_params

    all_ou = torch.cat((best_params[log_alpha_idx], best_params[other_ou_idx]), dim=-1)
    if return_ou_lambda:
        if ou_lambda_mode == 0:
            ou_lambda = torch.zeros_like(best_params[ou_lambda_idx])
        elif ou_lambda_mode == 1:
            ou_lambda = torch.ones_like(best_params[ou_lambda_idx])
        elif ou_lambda_mode == 2:
            ou_lambda = torch.sigmoid(best_params[ou_lambda_idx])
        else:
            raise ValueError(f"Invalid Pagel lambda mode: {ou_lambda_mode}")
        best_params = best_params[:n_trees] + [best_params[log_r_idx], ou_lambda, all_ou]
    else:
        best_params = best_params[:n_trees] + [best_params[log_r_idx], all_ou]

    if step_callback is not None:
        TRACE.enabled = False
        TRACE.reset()

    return best_params, best_loss


# optimize ELBO with Adam (BM)
def Lq_optimize_torch_BM(
    params,
    mode,
    x_tensor_list,
    gene_names,
    share_list_torch,
    max_iter,
    learning_rate,
    device,
    wandb_flag,
    window,
    tol,
    approx,
    nb,
    library_list_tensor,
    const,
):
    """
    Optimize ELBO with PyTorch Adam.

    params: List of (batch_size, all_param_dim)
            [Lq_params tree1, Lq_params tree2, ..., Lq_params treeN, logr, pagel_lambda]
    mode: 0 for star tree, 1 for original tree, 2 for optimizing pagel's lambda
    x_tensor: list of (batch_size, n_cells)
    gene_names: list of gene names
    share_list_torch: list of tensors
    max_iter: maximum number of iterations
    learning_rate: learning rate for Adam
    device: device to use
    wandb_flag: whether to use wandb
    window: number of recent iterations to check for convergence
    tol: convergence tolerance
    approx: approximation method for Poisson/NB likelihood expectation
    nb: whether to use negative binomial likelihood
    lib: library size normalization
    const: whether to include constant terms in likelihood
    
    Returns: (batch_size, ) numpy array of params and losses
    """
    batch_size, _ = params[0].shape
    pagel_lambda = params[-1].clone().detach()
    n_trees = len(x_tensor_list)

    _log_r_lo, _log_r_hi = DEFAULT_LOG_R_CLAMP
    _log_s2_lo, _log_s2_hi = DEFAULT_LOG_S2_CLAMP

    def _clamp_bm_params_(param_list):
        with torch.no_grad():
            if nb:
                param_list[-2].clamp_(_log_r_lo, _log_r_hi)
            for lq in param_list[:n_trees]:
                n_cells = lq.shape[-1] // 2
                lq[..., n_cells:2 * n_cells].clamp_(_log_s2_lo, _log_s2_hi)

    # optimize q parameters
    for i in range(len(params)-1):
        params[i] = params[i].requires_grad_(True)

    # Fix pagel's lambda
    if mode == 0: # star tree
        params[-1] = torch.zeros_like(pagel_lambda, dtype=pagel_lambda.dtype, device=device)
    elif mode == 1: # original tree
        params[-1] = torch.ones_like(pagel_lambda, dtype=pagel_lambda.dtype, device=device)
    else: # optimize lambda
        params[-1] = params[-1].requires_grad_(True)

    _clamp_bm_params_(params)

    optimizer = torch.optim.Adam([p for p in params if p.requires_grad], lr=learning_rate)
    
    # Track best parameters for gradient-based params
    best_params = [p.clone().detach() for p in params[:-1]] + [
        torch.zeros((batch_size, 3), dtype=params[0].dtype, device=device)
    ] # q, (lambda, mu, sigma)
    best_loss = torch.full((batch_size, ), float("inf"), dtype=params[0].dtype, device=device)

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros((batch_size, ), dtype=torch.bool, device=device)
    nan_streak = torch.zeros((batch_size, ), dtype=torch.long, device=device)
    nan_patience = 50  # mark as failed after this many consecutive NaN iterations

    for n in range(max_iter):
        optimizer.zero_grad()
        active_batch = ~converged_mask
        active_loss_list = []

        # get loss for each tree for not yet converged genes
        for i in range(n_trees):
            x_tensor = x_tensor_list[i][active_batch, :]
            Lq_params = params[i][active_batch, :]
            r_param = params[-2][active_batch]

            # Map lambda to strict (0, 1) bound safely
            if mode == 2:
                safe_lambda = torch.sigmoid(params[-1][active_batch])
            else:
                safe_lambda = params[-1][active_batch]

            share = share_list_torch[i]
            lib = library_list_tensor[i] # (n_cells,)

            loss, mu, sigma = Lq_neg_log_lik_torch(
                Lq_params, r_param, safe_lambda, 0, x_tensor, # mode=0 for BM likelihood
                None, share, None, None, device, approx, None, None, nb, lib, const,
                log_s2_clamp=(_log_s2_lo, _log_s2_hi),
            )
            active_loss_list.append(loss) # list of (b,)

        active_joint_loss = torch.stack(active_loss_list, dim=-1).sum(dim=-1) # (b, t) -> (b,)

        # update best params for not yet converged genes
        with torch.no_grad():
            joint_loss = torch.full((batch_size,), float("inf"), dtype=params[0].dtype, device=device)
            joint_loss[active_batch] = active_joint_loss.clone().detach()

            # update improved loss
            mask = (joint_loss < best_loss) & ~converged_mask
            best_loss = torch.where(mask, joint_loss, best_loss)

            # update improved q params
            for i, param in enumerate(params[:-1]):
                current_mask = mask.unsqueeze(-1) if param.dim() > 1 else mask
                best_params[i] = torch.where(
                    current_mask,  # (B, 1) or (B, )
                    param.clone().detach(), # (B, all_param_dim) or (B, )
                    best_params[i]
                )
            
            # update improved bm params
            if mode == 2: # update lambda
                best_params[-1][mask, 0] = torch.sigmoid(params[-1][mask].clone().detach())
            else:
                best_params[-1][mask, 0] = params[-1][mask].clone().detach()
            
            # update improved mu and sigma from active outputs
            best_params[-1][mask, 1] = mu[mask[active_batch]].clone().detach()
            best_params[-1][mask, 2] = sigma[mask[active_batch]].clone().detach()

            if wandb_flag:
                # plot loss of first gene in batch
                wandb.log({
                        "iter": n,
                        f"{gene_names[0]}_bm{mode}_elbo_loss": best_loss[0].item(), 
                })
        
        # Store loss history
        loss_history.append(joint_loss.clone().detach())

        # Early exit for persistent NaN/inf losses
        with torch.no_grad():
            is_bad = ~torch.isfinite(joint_loss)
            nan_streak = torch.where(is_bad & ~converged_mask, nan_streak + 1, nan_streak)
            nan_failed = (nan_streak >= nan_patience) & ~converged_mask
            if nan_failed.any():
                converged_mask = converged_mask | nan_failed
                if converged_mask.all():
                    break

        # check convergence in window
        if n >= window:
            # Check if loss has stabilized within window
            start_loss = loss_history[-window]
            denom = torch.maximum(
                start_loss.abs(), torch.tensor(1.0, dtype=loss.dtype, device=device)
            )
            relative_decrease = torch.abs(start_loss - best_loss) / denom

            # Mark as newly converged if variance is below tolerance
            newly_converged = (relative_decrease < tol) & ~converged_mask

            # update converged mask if found newly converged genes
            if newly_converged.any():
                converged_mask = converged_mask | newly_converged # update converged mask
                if converged_mask.all(): # break if all genes have converged
                    break

        # backpropagate for not yet converged genes from active gene outputs
        active_loss_flat = active_joint_loss[~converged_mask[active_batch]]
        # skipping non-finite losses so one bad gene can't poison others
        valid_mask = torch.isfinite(active_loss_flat)
        if not valid_mask.any():
            continue
        loss = active_loss_flat[valid_mask].sum()
        loss.backward()
        optimizer.step()

        _clamp_bm_params_(params)

    # warning if not all genes have converged
    print(f"\nChecking convergence for bm{mode} ELBO...")
    nan_failed_genes = (best_loss == float("inf"))
    non_converged = ~converged_mask
    slow_genes = non_converged & ~nan_failed_genes
    if nan_failed_genes.any():
        print(f"\nWARNING: {nan_failed_genes.sum().item()} gene(s) failed (persistent NaN loss):")
        for i in range(batch_size):
            if nan_failed_genes[i]:
                print(f"   {gene_names[i]}: NaN loss (skipped after {nan_patience} iters)")
    if slow_genes.any():
        print(f"\nWARNING: {slow_genes.sum().item()}/{batch_size} gene(s) did not converge:")
        print(f"   Gene Name | Relative Decrease | Final Loss")
        print(f"   {'-' * 25}")
        for i in range(batch_size):
            if slow_genes[i]:
                rd = relative_decrease[i].item() if 'relative_decrease' in locals() else float('nan')
                print(
                    f"   {gene_names[i]}: {rd:.2e} | " +
                    f"{best_loss[i].item():.2e}"
                )
        print(f"\nRecommendations for non-converged genes:")
        print(f"   - Increase max_iter (current: {max_iter})")
        print(f"   - Increase learning_rate (current: {learning_rate})")
        print(f"   - Increase tol (current: {tol})")
        print(f"   - Decrease window (current: {window})")
        print(f"   - Check data quality for these genes")

    _clamp_bm_params_(best_params)

    return best_params, best_loss


# optimize BM with Adam
def optimize_torch_BM(
    params,
    mode,
    x_tensor_list,
    gene_names,
    share_list_torch,
    max_iter,
    learning_rate,
    device,
    wandb_flag,
    window,
    tol,
    const,
):
    """
    Optimize BM with PyTorch Adam.

    params: pagel_lambda (batch_size, )
    mode: 0 for star tree, 1 for original tree, 2 for optimizing pagel's lambda
    x_tensor: list of (batch_size, n_cells)
    gene_names: list of gene names
    share_list_torch: list of tensors
    max_iter: maximum number of iterations
    learning_rate: learning rate for Adam
    device: device to use
    wandb_flag: whether to use wandb
    window: number of recent iterations to check for convergence
    tol: convergence tolerance
    const: whether to include constant terms in likelihood
    
    Returns: (batch_size, ) numpy array of params and losses
    """
    batch_size = params.shape[0]
    n_trees = len(x_tensor_list)

    # Fix pagel's lambda
    if mode == 0 or mode == 1: # star tree
        params *= 0.0 # set lambda to 0
        if mode == 1: # original tree
            params += 1.0 # set lambda to 1
    
        # analytical solution with fixed lambda
        joint_loss = torch.zeros_like(params)

        for i in range(n_trees):
            x_tensor = x_tensor_list[i]
            n_cells = x_tensor.shape[-1]

            loss, mu, sigma = bm_neg_log_lik_torch_kkt(
                params, torch.zeros_like(x_tensor), x_tensor, share_list_torch[i], device
            )  
            if const:
                loss = loss + n_cells/2 * (1 + torch.log(2 * torch.tensor(torch.pi, device=device)))
            joint_loss = joint_loss + loss

        params = torch.stack((params, mu, sigma), dim=-1)
        return params, joint_loss

    # optimize lambda (caller passes logit init, e.g. -2 -> sigmoid(-2)~=0.12)
    params.requires_grad_(True) # pagel_lambda
    optimizer = torch.optim.Adam([params], lr=learning_rate)

    # Track best parameters for all parameter types
    best_results = torch.zeros((batch_size, 3), dtype=params.dtype, device=device)
    best_loss = torch.full((batch_size, ), float("inf"), dtype=params.dtype, device=device)

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros((batch_size, ), dtype=torch.bool, device=device)
    nan_streak = torch.zeros((batch_size, ), dtype=torch.long, device=device)
    nan_patience = 50  # mark as failed after this many consecutive NaN iterations

    for n in range(max_iter):
        optimizer.zero_grad()
        active_batch = ~converged_mask
        active_loss_list = []

        for i in range(n_trees):
            x_tensor = x_tensor_list[i][active_batch, :]
            n_cells = x_tensor.shape[-1]
            bm_params = torch.sigmoid(params[active_batch])

            loss, mu, sigma = bm_neg_log_lik_torch_kkt(
                bm_params, torch.zeros_like(x_tensor), x_tensor, share_list_torch[i], device
            )  
            if const:
                loss = loss + n_cells/2 * (1 + torch.log(2 * torch.tensor(torch.pi, device=device)))
            active_loss_list.append(loss)

        active_joint_loss = torch.stack(active_loss_list, dim=-1).sum(dim=-1)

        with torch.no_grad():
            # Reconstruct the full-batch loss for accurate indexing
            joint_loss_full = torch.full((batch_size,), float("inf"), dtype=params.dtype, device=device)
            joint_loss_full[active_batch] = active_joint_loss.clone().detach()

            mask = (joint_loss_full < best_loss) & ~converged_mask
            best_loss = torch.where(mask, joint_loss_full, best_loss)
            actual_lambda = torch.sigmoid(params)

            best_results[mask, 0] = actual_lambda[mask].clone().detach()
            best_results[mask, 1] = mu[mask[active_batch]].clone().detach()
            best_results[mask, 2] = sigma[mask[active_batch]].clone().detach()

            if wandb_flag:
                wandb.log({
                        "iter": n,
                        f"{gene_names[0]}_bm{mode}_elbo_loss": best_loss[0].item(), 
                })
        
        loss_history.append(joint_loss_full.clone().detach())

        # Early exit for persistent NaN/inf losses (cumulative; see note
        # in ou_optimize_torch).
        with torch.no_grad():
            is_bad = ~torch.isfinite(joint_loss_full)
            nan_streak = torch.where(is_bad & ~converged_mask, nan_streak + 1, nan_streak)
            nan_failed = (nan_streak >= nan_patience) & ~converged_mask
            if nan_failed.any():
                converged_mask = converged_mask | nan_failed
                if converged_mask.all():
                    break

        if n >= window:
            start_loss = loss_history[-window]
            denom = torch.maximum(
                start_loss.abs(), torch.tensor(1.0, dtype=joint_loss_full.dtype, device=device)
            )
            relative_decrease = torch.abs(start_loss - best_loss) / denom

            newly_converged = (relative_decrease < tol) & ~converged_mask

            if newly_converged.any():
                converged_mask = converged_mask | newly_converged
                if converged_mask.all():
                    break

        active_loss_flat = active_joint_loss[~converged_mask[active_batch]]
        valid_mask = torch.isfinite(active_loss_flat)
        if not valid_mask.any():
            continue
        loss = active_loss_flat[valid_mask].sum()
        loss.backward()
        optimizer.step()

    # warning if not all genes have converged
    print(f"\nChecking convergence for bm{mode} ELBO...")
    nan_failed_genes = (best_loss == float("inf"))
    non_converged = ~converged_mask
    slow_genes = non_converged & ~nan_failed_genes
    if nan_failed_genes.any():
        print(f"\nWARNING: {nan_failed_genes.sum().item()} gene(s) failed (persistent NaN loss):")
        for i in range(batch_size):
            if nan_failed_genes[i]:
                print(f"   {gene_names[i]}: NaN loss (skipped after {nan_patience} iters)")
    if slow_genes.any():
        print(f"\nWARNING: {slow_genes.sum().item()}/{batch_size} gene(s) did not converge:")
        print(f"   Gene Name | Relative Decrease | Final Loss")
        print(f"   {'-' * 25}")
        for i in range(batch_size):
            if slow_genes[i]:
                rd = relative_decrease[i].item() if 'relative_decrease' in locals() else float('nan')
                print(
                    f"   {gene_names[i]}: {rd:.2e} | " +
                    f"{best_loss[i].item():.2e}"
                )
        print(f"\nRecommendations for non-converged genes:")
        print(f"   - Increase max_iter (current: {max_iter})")
        print(f"   - Increase learning_rate (current: {learning_rate})")
        print(f"   - Increase tol (current: {tol})")
        print(f"   - Decrease window (current: {window})")
        print(f"   - Check data quality for these genes")

    return best_results, best_loss
