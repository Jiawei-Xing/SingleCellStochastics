from joblib import Parallel, delayed
import numpy as np
import torch
from scipy.optimize import minimize
import wandb

from .likelihood import ou_neg_log_lik_numpy
from .likelihood import ou_neg_log_lik_torch
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
            ou_neg_log_lik_numpy,
            params_init,
            args=(mode, expr, diverge_list, share_list, epochs_list, beta_list),
            bounds=[(1e-6, None)] * 2 + [(None, None)] * (len(params_init)-2),
            method="L-BFGS-B",
        )
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
):
    """
    Optimize the OU-only likelihood with PyTorch Adam.
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
    """
    ou_params = params_init.clone().detach().requires_grad_(True)
    batch_size, N_sim, _ = expr_list_torch[0].shape
    n_trees = len(expr_list_torch)

    # Track best parameters for all parameter types
    best_params = params_init.clone().detach()
    best_loss = torch.full((batch_size, N_sim), float("inf"), device=device)
    optimizer = torch.optim.Adam([ou_params], lr=learning_rate)
    
    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros((batch_size, N_sim), dtype=torch.bool, device=device)

    for n in range(max_iter):
        optimizer.zero_grad()

        # initialize loss matrix and store indices of not yet converged genes
        loss_matrix = torch.zeros(batch_size, N_sim, n_trees, device=device)
        active_batch = (~converged_mask).any(dim=1)

        # get loss for each tree for not yet converged genes
        for i in range(n_trees):
            expr_tensor = expr_list_torch[i][active_batch, :, :] # only use not yet converged genes
            ou_params_tensor = ou_params[active_batch, :, :] # only optimize not yet converged genes
            
            sigma2_q = torch.zeros_like(expr_tensor) # trace term = 0
            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]

            loss = ou_neg_log_lik_torch(
                ou_params_tensor,
                sigma2_q,
                mode,
                expr_tensor,
                diverge,
                share,
                epochs,
                beta,
                device=device
            )
            loss_matrix[active_batch, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(
            torch.tensor(n_trees, device=device, dtype=torch.float32)
        )  # (batch_size, N_sim)

        # update best params for not yet converged genes
        with torch.no_grad():
            mask = (average_loss < best_loss) & ~converged_mask
            best_loss[mask] = average_loss[mask].clone().detach()
            best_params[mask] = ou_params[mask].clone().detach()

            if wandb_flag:
                wandb.log(
                    {f"{gene_names[0]}_h{mode-1}_ou_loss": best_loss[0, 0], "iter": n}
                )

        # Store loss history
        loss_history.append(average_loss.clone().detach())

        # Check for convergence
        if n >= window:
            # Check if loss has stabilized within window
            start_loss = loss_history[-window]
            denom = torch.maximum(start_loss.abs(), torch.tensor(1.0, device=device))
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
    em
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
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    # Determine which parameters should have gradients based on EM mode
    if em == "e":
        # E-step: optimize ELBO with fixed OU parameters
        requires_grad_mask = [True] * (len(params) - 1) + [False]
    elif em == "m":
        # M-step: optimize OU parameters with fixed ELBO
        requires_grad_mask = [False] * (len(params) - 1) + [True]
    else:
        # optimize both ELBO and OU parameters
        requires_grad_mask = [True] * len(params)
    
    # Create params_tensor with appropriate gradient requirements
    params_tensor = [
        p.clone().detach().requires_grad_(requires_grad) 
        for p, requires_grad in zip(params, requires_grad_mask)
    ]  # list of (batch_size, N_sim, param_dim)

    batch_size, N_sim, _ = params_tensor[0].shape
    n_trees = len(x_tensor_list)
    
    # Track best parameters for all parameter types
    if em is None:
        # For full optimization, only track OU parameters (original behavior)
        best_params = params[-1].clone().detach()  # (batch_size, N_sim, ou_param_dim)
    else:
        # For EM steps, track all parameters
        best_params = [p.clone().detach() for p in params]
    
    best_loss = torch.full((batch_size, N_sim), float("inf"), device=device)
    optimizer = torch.optim.Adam(params_tensor, lr=learning_rate)

    # Track loss history for convergence checking
    loss_history = []
    converged_mask = torch.zeros((batch_size, N_sim), dtype=torch.bool, device=device)

    for n in range(max_iter):
        optimizer.zero_grad()

        # initialize loss matrix and store indices of not yet converged genes
        loss_matrix = torch.zeros(batch_size, N_sim, n_trees, device=device)
        active_batch = (~converged_mask).any(dim=1)

        # get loss for each tree for not yet converged genes
        for i in range(n_trees):
            x_tensor = x_tensor_list[i][active_batch, :, :]
            Lq_params = params_tensor[i][active_batch, :, :]
            ou_params = params_tensor[-1][active_batch, :, :]

            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]

            loss = Lq_neg_log_lik_torch(
                Lq_params,
                ou_params,
                mode,
                x_tensor,
                diverge,
                share,
                epochs,
                beta,
                device=device,
                approx=approx
            )
            loss_matrix[active_batch, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(
            torch.tensor(n_trees, device=device, dtype=torch.float32)
        )  # (batch_size, N_sim)

        # update best params for not yet converged genes
        with torch.no_grad():
            # s2, alpha, sigma2 should be positive (instead of clamping, optimize square)
            #for i in range(n_trees - 1):
            #    n_cells = x_tensor_list[i].shape[-1]
            #    if (params_tensor[i][:, :, n_cells:] < 1e-6).any():
            #        params_tensor[i][:, :, n_cells:].clamp_(min=1e-6)
            #if (params_tensor[-1][:, :, :2] < 1e-6).any():
            #    params_tensor[-1][:, :, :2].clamp_(min=1e-6)

            # update best params
            mask = (average_loss < best_loss) & ~converged_mask
            best_loss[mask] = average_loss[mask].clone().detach()
            
            if em is None:
                # For full optimization, only update OU parameters
                best_params[mask] = params_tensor[-1][mask].clone().detach()
            else:
                # For EM steps, update all parameters
                for i, param in enumerate(params_tensor):
                    best_params[i][mask] = param[mask].clone().detach()

            if wandb_flag:
                # plot loss of first gene in batch
                if em is None:
                    wandb.log(
                        {f"{gene_names[0]}_h{mode-1}_all_loss": best_loss[0, 0], "iter": n}
                    )
                else:
                    wandb.log(
                        {f"{gene_names[0]}_h{mode-1}_{em}_loss": best_loss[0, 0], "iter": n}
                    )
        
        # Store loss history
        loss_history.append(average_loss.clone().detach())

        # check convergence in window   
        if n >= window:
            # Check if loss has stabilized within window
            start_loss = loss_history[-window]
            denom = torch.maximum(start_loss.abs(), torch.tensor(1.0, device=device))
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

    if em is None:
        return (
            best_params.clone().detach().cpu().numpy(),
            best_loss.clone().detach().cpu().numpy(),
        )
    else:
        # For EM steps, return all best parameters as a list
        best_params_list = [p.clone().detach() for p in best_params]
        return (
            best_params_list,
            best_loss.clone().detach(),
        )
