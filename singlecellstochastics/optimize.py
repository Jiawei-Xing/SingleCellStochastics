from joblib import Parallel, delayed
import numpy as np
import torch
from scipy.optimize import minimize
import wandb

from .likelihood import ou_neg_log_lik_numpy, ou_neg_log_lik_numpy_kkt
from .likelihood import ou_neg_log_lik_torch, ou_neg_log_lik_torch_kkt
from .elbo import Lq_neg_log_lik_torch


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
    kkt
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

    # Track best parameters and loss
    best_params = [
        p.clone().detach() for p in ou_params
    ] # [alpha, others]

    best_loss = torch.full(
        (batch_size, N_sim), float("inf"), dtype=params_init.dtype, device=device
    )
    optimizer = torch.optim.Adam(ou_params, lr=learning_rate)

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros(
        (batch_size, N_sim), dtype=torch.bool, device=device
    )

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
                    device
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
                    device
                )
            loss_matrix[active_batch, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(
            torch.tensor(n_trees, device=device, dtype=loss.dtype)
        )  # (batch_size, N_sim)

        # update best params for not yet converged genes
        with torch.no_grad():
            mask = (average_loss < best_loss) & ~converged_mask
            best_loss = torch.where(
                mask,
                average_loss.clone().detach(),
                best_loss
            ) # update best loss

            # update sigma and theta from kkt
            if kkt:
                other_params = torch.cat(
                    (sigma.unsqueeze(-1).clone().detach(), # (B,S,1)
                    theta.clone().detach()), dim=-1 # (B,S,n_regimes)
                )
                if mode == 1:
                    ou_params[1][active_batch, :, :2] = other_params
                else:
                    ou_params[1][active_batch, :, :] = other_params

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

        # backpropagate for not yet converged genes
        loss = average_loss[~converged_mask].mean()
        loss.backward()
        optimizer.step()

    # warning if not all genes have converged
    print(f"\nChecking convergence for h{mode-1} OU init...")
    if not converged_mask.all():
        print(f"\n⚠️  WARNING: {(~converged_mask).sum().item()}/{batch_size} genes did not converge:")
        print(f"   Gene Name | Relative Decrease | Final Loss")
        print(f"   {'-' * 25}")
        for i in range(batch_size):
            for j in range(N_sim):
                if not converged_mask[i, j]:
                    print(
                        f"   {gene_names[i]}: {relative_decrease[i, j].item():.2e} | " +
                        f"{best_loss[i, j].item():.2e}"
                    )
        print(f"\n💡 Recommendations for non-converged genes:")
        print(f"   - Increase max_iter (current: {max_iter})")
        print(f"   - Increase learning_rate (current: {learning_rate})")
        print(f"   - Increase tol (current: {tol})")
        print(f"   - Decrease window (current: {window})")
        print(f"   - Check data quality for these genes")

    best_params = torch.cat(
        [p for p in best_params], dim=-1
     ) # (batch_size, N_sim, n_params)

    return best_params, best_loss


# optimize ELBO with Adam
def Lq_optimize_torch(
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
    nb
):
    """
    Optimize ELBO with PyTorch Adam.

    params: List of (batch_size, N_sim, all_param_dim)
            [Lq_params tree1, Lq_params tree2, ..., Lq_params treeN, shared OU_params]
    mode: 1 for H0, 2 for H1
    x_tensor: list of (batch_size, N_sim, n_cells)
    gene_names: list of gene names
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    window: number of recent iterations to check for convergence
    tol: convergence tolerance
    approx: approximation method for Poisson likelihood expectation
    em: "e" for E-step, "m" for M-step, None for both together
    prior: L2 regularization strength for OU alpha
    kkt: whether to use KKT condition for OU likelihood
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    batch_size, N_sim, _ = params[0].shape
    n_trees = len(x_tensor_list)

    # Initialize parameters for optimization
    params_tensor = [
        p.clone().detach() for p in params[:-1]
    ] + [
        params[-1][:, :, 0:1].clone().detach(),
        params[-1][:, :, 1:].clone().detach()
    ] # [Lq tree1, Lq tree2, ..., Lq treeN, r, alpha, other OU]

    # Set requires_grad based on EM mode
    if em == "e":
        # E-step: optimize ELBO with fixed OU parameters
        for i in range(len(params_tensor)-2):
            params_tensor[i] = params_tensor[i].requires_grad_(True)
    elif em == "m":
        # M-step: optimize OU parameters with fixed ELBO
        params_tensor[-2] = params_tensor[-2].requires_grad_(True) # alpha
        if not kkt:
            params_tensor[-1] = params_tensor[-1].requires_grad_(True) # other OU
    else:
        # optimize both ELBO and OU parameters
        for i in range(len(params_tensor)-1):
            params_tensor[i] = params_tensor[i].requires_grad_(True)
        if not kkt:
            params_tensor[-1] = params_tensor[-1].requires_grad_(True) # other OU
    
    # Track best parameters for all parameter types
    best_params = [p.clone().detach() for p in params_tensor]
    best_loss = torch.full(
        (batch_size, N_sim), float("inf"), dtype=params[0].dtype, device=device
    )
    optimizer = torch.optim.Adam(params_tensor, lr=learning_rate)

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros(
        (batch_size, N_sim), dtype=torch.bool, device=device
    )

    for n in range(max_iter):
        optimizer.zero_grad()

        # initialize loss matrix and store indices of not yet converged genes
        loss_matrix = torch.zeros(
            batch_size, N_sim, n_trees, dtype=params[0].dtype, device=device
        )
        active_batch = (~converged_mask).any(dim=1)

        # get loss for each tree for not yet converged genes
        for i in range(n_trees):
            x_tensor = x_tensor_list[i][active_batch, :, :]
            Lq_params = params_tensor[i][active_batch, :, :]
            ou_params = [
                params_tensor[-2][active_batch, :, :],
                params_tensor[-1][active_batch, :, :]
            ] # [alpha, others]

            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]

            if kkt:
                loss, sigma, theta = Lq_neg_log_lik_torch(
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
                    kkt,
                    nb
                )
            else:
                loss = Lq_neg_log_lik_torch(
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
                    kkt,
                    nb
                )
            loss_matrix[active_batch, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(
            torch.tensor(n_trees, device=device, dtype=loss.dtype)
        )  # (batch_size, N_sim)

        # update best params for not yet converged genes
        with torch.no_grad():
            mask = (average_loss < best_loss) & ~converged_mask
            best_loss = torch.where(
                mask,
                average_loss.clone().detach(),
                best_loss
            ) # update best loss

            # update sigma and theta from kkt
            if kkt:
                other_params = torch.cat(
                    (sigma.unsqueeze(-1).clone().detach(), # (B,S,1)
                    theta.clone().detach()), dim=-1 # (B,S,n_regimes)
                )
                if mode == 1:
                    params_tensor[-1][active_batch, :, :2] = other_params
                else:
                    params_tensor[-1][active_batch, :, :] = other_params

            for i, param in enumerate(params_tensor):
                best_params[i] = torch.where(
                    mask.unsqueeze(-1),  # (B,S,1)
                    param.clone().detach(), # (B,S,all_param_dim)
                    best_params[i]
                ) # update best Lq params

            if wandb_flag:
                # plot loss of first gene in batch
                if em is None:
                    wandb.log({
                            "iter": n,
                            f"{gene_names[0]}_h{mode-1}_elbo_loss": best_loss[0, 0], 
                    })
                else:
                    wandb.log({
                            "iter": n,
                            f"{gene_names[0]}_h{mode-1}_{em}_loss": best_loss[0, 0], 
                    })
        
        # Store loss history
        loss_history.append(average_loss.clone().detach())

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

        # backpropagate for not yet converged genes
        loss = average_loss[~converged_mask].mean()
        loss.backward()
        optimizer.step()
    
    # warning if not all genes have converged
    print(f"\nChecking convergence for h{mode-1} ELBO...")
    if not converged_mask.all():
        print(f"\n⚠️  WARNING: {(~converged_mask).sum().item()}/{batch_size} genes did not converge:")
        print(f"   Gene Name | Relative Decrease | Final Loss")
        print(f"   {'-' * 25}")
        for i in range(batch_size):
            for j in range(N_sim):
                if not converged_mask[i, j]:
                    print(
                        f"   {gene_names[i]}: {relative_decrease[i, j].item():.2e} | " +
                        f"{best_loss[i, j].item():.2e}"
                    )
        print(f"\n💡 Recommendations for non-converged genes:")
        print(f"   - Increase max_iter (current: {max_iter})")
        print(f"   - Increase learning_rate (current: {learning_rate})")
        print(f"   - Increase tol (current: {tol})")
        print(f"   - Decrease window (current: {window})")
        print(f"   - Check data quality for these genes")
    
    best_params = [p for p in best_params[:-2]] + [
        torch.cat((best_params[-2], best_params[-1]), dim=-1)
    ] # [Lq tree1, Lq tree2, ..., Lq treeN, all OU]

    return best_params, best_loss
