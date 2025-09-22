import time
from typing import Dict
import torch
from Bio import Phylo

from .ornstein_uhlenbeck import oup_neg_log_likelihood


def print_state(neg_log_lik, alpha, sigma, theta_dict):
    """
    Utility function to print the current state of the optimization.
    """
    print(f"\nNeg log likelihood (or neg elbo) = {neg_log_lik.item()}")
    print(f"\talpha: {alpha.item()}")
    print(f"\tsigma: {sigma.item()}")
    print(
        f"\tthetas: {{ {', '.join(f'{k}: {v.item()}' for k, v in theta_dict.items())} }}"
    )


def adam_optimize_ou_parameters(
    tree: Phylo.BaseTree.Tree,
    alpha_init: torch.Tensor,
    sigma_init: torch.Tensor,
    theta_dict_init: Dict[str, torch.Tensor],
    origin_expression: torch.Tensor,
    poisson_logl_mode: str = "deterministic",
) -> float:
    """
    Optimize OU parameters using the Adam optimizer to minimize negative log-likelihood.

    Args:
        tree: A Biopython `Tree` object with assigned regimes and tip read_counts.
        alpha_init: Initial value for selective strength parameter.
        sigma2_init: Initial value for variance parameter.
        theta_dict_init: A dictionary mapping regime labels to initial optimal expression values (theta).
        origin_expression: Expression value assumed at the origin of the experiment.
        poisson_logl_mode (str): Mode for Poisson sampling, either "deterministic", "stochastic", or "variational".

    Returns:
        optimal_neg_log_lik: The optimal negative log-likelihood after optimization (float).
    """
    # Set requires_grad=True for optimization
    alpha = alpha_init.detach().clone().requires_grad_(True)
    sigma = sigma_init.detach().clone().requires_grad_(True)
    theta_dict = {
        regime: theta.detach().clone().requires_grad_(True)
        for regime, theta in theta_dict_init.items()
    }

    # Setup variational parameters as needed
    if poisson_logl_mode == "variational":
        num_tips = len(tree.get_terminals())
        variational_means = torch.ones(num_tips, requires_grad=True)
        variational_std_devs = torch.ones(num_tips, requires_grad=True)
        params = (
            [alpha, sigma]
            + list(theta_dict.values())
            + [variational_means, variational_std_devs]
        )
    else:
        variational_means = None
        variational_std_devs = None
        params = [alpha, sigma] + list(theta_dict.values())

    # Use Adam optimizer
    optimizer = torch.optim.Adam(params, lr=0.1)

    # Print initial state
    initial_neg_log_lik = oup_neg_log_likelihood(
        tree,
        alpha,
        sigma,
        theta_dict,
        origin_expression,
        poisson_logl_mode=poisson_logl_mode,
        variational_means=variational_means,
        variational_log_stds=variational_std_devs,
    )
    print()
    print("Initial state:")
    print_state(initial_neg_log_lik, alpha, sigma, theta_dict)
    if poisson_logl_mode == "variational":
        print(f"\tvariational_means: {[v.item() for v in variational_means]}")
        print(f"\tvariational_std_devs: {[v.item() for v in torch.exp(variational_std_devs)]}")

    # Convergence criteria
    max_not_improved_steps = 100  # Number of steps without improvement to allow
    convergence = 1e-6  # Convergence threshold

    # Track time
    start_time = time.time()
    log_freq = 100  # How many steps between progress updates

    # Optimization loop
    prev_negll = None
    not_improved_steps = 0
    step = 1
    print()
    print("Running optimization:")
    while True:
        optimizer.zero_grad()
        neg_log_lik = oup_neg_log_likelihood(
            tree,
            alpha,
            sigma,
            theta_dict,
            origin_expression,
            poisson_logl_mode=poisson_logl_mode,
            variational_means=variational_means,
            variational_log_stds=variational_std_devs,
        )
        neg_log_lik.backward()
        optimizer.step()

        cur_negll = neg_log_lik.item()

        # Print progress
        if step % log_freq == 0:
            print(
                f"Step {step}, Negative log-likelihood (or -elbo): {cur_negll}, alpha: {alpha.item()}, sigma: {sigma.item()}, thetas: {[theta.item() for theta in theta_dict.values()]}"
            )
            if poisson_logl_mode == "variational":
                print(f"\tvariational_means: {[v.item() for v in variational_means]}")
                print(f"\tvariational_std_devs: {[v.item() for v in torch.exp(variational_std_devs)]}")

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
    print(
        f"Optimization completed in {end_time - start_time:.2f} seconds over {step} steps."
    )

    optimal_neg_log_lik = oup_neg_log_likelihood(
        tree,
        alpha,
        sigma,
        theta_dict,
        origin_expression,
        poisson_logl_mode=poisson_logl_mode,
        variational_means=variational_means,
        variational_log_stds=variational_std_devs,
    )
    print()
    print("Final state:")
    print_state(optimal_neg_log_lik, alpha, sigma, theta_dict)
    if poisson_logl_mode == "variational":
        print(f"\tvariational_means: {[v.item() for v in variational_means]}")
        print(f"\tvariational_std_devs: {[v.item() for v in torch.exp(variational_std_devs)]}")

    return (
        optimal_neg_log_lik,
        alpha.detach(),
        sigma.detach(),
        {regime: theta.detach() for regime, theta in theta_dict.items()},
    )
