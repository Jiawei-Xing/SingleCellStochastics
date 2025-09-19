
import time
from typing import Dict
import torch
from Bio import Phylo

from .ornstein_uhlenbeck import oup_neg_log_likelihood


def adam_optimize_ou_parameters(
    tree: Phylo.BaseTree.Tree,
    alpha_init: torch.Tensor,
    sigma_init: torch.Tensor,
    theta_dict_init:  Dict[str, torch.Tensor],
    root_expression: torch.Tensor,
) -> float:
    """
    Optimize OU parameters using the Adam optimizer to minimize negative log-likelihood.
    
    Args:
        tree: A Biopython `Tree` object with assigned regimes and tip read_counts.
        alpha_init: Initial value for selective strength parameter.
        sigma2_init: Initial value for variance parameter.
        theta_dict_init: A dictionary mapping regime labels to initial optimal expression values (theta).
        root_expression: Expression value at the root node.
    
    Returns:
        optimal_neg_log_lik: The optimal negative log-likelihood after optimization (float).
    """
    # Set requires_grad=True for optimization
    alpha = alpha_init.detach().clone().requires_grad_(True)
    sigma = sigma_init.detach().clone().requires_grad_(True)
    theta_dict = {regime: theta.detach().clone().requires_grad_(True) for regime, theta in theta_dict_init.items()}
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([alpha, sigma] + list(theta_dict.values()), lr=0.01)
    
    # Convergence criteria
    max_not_improved_steps = 100  # Number of steps without improvement to allow
    convergence = 1e-6  # Convergence threshold
    
    # Track time
    start_time = time.time()
    print_freq = 100  # How many steps between progress updates

    # Optimization loop
    prev_negll = None
    not_improved_steps = 0
    step = 1
    while True:
        optimizer.zero_grad()
        neg_log_lik = oup_neg_log_likelihood(tree, alpha, sigma, theta_dict, root_expression)
        neg_log_lik.backward()
        optimizer.step()

        cur_negll = neg_log_lik.item()

        # Print progress
        if step == 1 or step % print_freq == 0:
            print(f"Step {step}, Negative Log-Likelihood: {cur_negll}, alpha: {alpha.item()}, sigma: {sigma.item()}, thetas: {[theta.item() for theta in theta_dict.values()]}")

        # Check for improvement
        if prev_negll and abs(cur_negll) - abs(prev_negll) < convergence:
            not_improved_steps += 1
        else:
            not_improved_steps = 0

        if not_improved_steps >= max_not_improved_steps:
            break

        prev_negll = cur_negll
        step += 1
    
    # Print total time
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds over {step} steps.")
    
    optimal_neg_log_lik = ou_neg_log_likelihood(tree, alpha, sigma, theta_dict, root_expression)
    
    return optimal_neg_log_lik, alpha.detach(), sigma.detach(), {regime: theta.detach() for regime, theta in theta_dict.items()}
    