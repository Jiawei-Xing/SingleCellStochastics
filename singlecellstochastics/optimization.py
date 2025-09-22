import time
from typing import Dict
from collections import deque
import torch
from Bio import Phylo

from .ornstein_uhlenbeck import oup_neg_log_likelihood


def print_state(neg_log_lik, alpha, sigma, theta_dict, log_path):
    """
    Utility function to print the current state of the optimization.
    """
    with open(log_path, "a") as f:
        f.write(f"\nNegative log-likelihood (or -elbo) = {neg_log_lik.item()}\n")
        f.write(f"\talpha: {alpha.item()}\n")
        f.write(f"\tsigma: {sigma.item()}\n")
        f.write(f"\tthetas: {{ {', '.join(f'{k}: {v.item()}' for k, v in theta_dict.items())} }}\n")


def adam_optimize_ou_parameters(
    tree: Phylo.BaseTree.Tree,
    alpha_init: torch.Tensor,
    sigma_init: torch.Tensor,
    theta_dict_init: Dict[str, torch.Tensor],
    origin_expression: torch.Tensor,
    log_path: str,
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
        log_path: Path to a log file to record optimization progress (str).
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
        variational_log_std_devs = torch.ones(num_tips, requires_grad=True)

        params = (
            [alpha, sigma]
            + list(theta_dict.values())
            + [variational_means, variational_log_std_devs]
        )
    else:
        variational_means = None
        variational_log_std_devs = None
        params = [alpha, sigma] + list(theta_dict.values())

    # Use Adam optimizer
    lr = 0.01
    optimizer = torch.optim.Adam(params, lr=lr)

    # Print initial state
    initial_neg_log_lik = oup_neg_log_likelihood(
        tree,
        alpha,
        sigma,
        theta_dict,
        origin_expression,
        poisson_logl_mode=poisson_logl_mode,
        variational_means=variational_means,
        variational_log_stds=variational_log_std_devs,
    )
    if log_path:
        with open(log_path, "a") as f:
            f.write("\nInitial state:\n")
    print_state(initial_neg_log_lik, alpha, sigma, theta_dict, log_path)
    if poisson_logl_mode == "variational":
        with open(log_path, "a") as f:
            f.write(f"\tvariational_means: {[v.item() for v in variational_means]}\n")
            f.write(f"\tvariational_std_devs: {[v.item() for v in torch.exp(variational_log_std_devs)]}\n")

    # Convergence criteria
    window_size = 10  # Number of past steps to consider for convergence
    convergence = 0.001  # Convergence threshold
    convergence_check_freq = 10  # How often to update the window and check for convergence
    
    # Log adam setup
    with open(log_path, "a") as f:
        f.write(f"\nSetting up Adam optimizer with:\n")
        f.write(f"\tlearning rate: {lr}\n")
        f.write(f"\tconvergence threshold: {convergence}\n")
        f.write(f"\tconvergence_window_size: {window_size}\n")
        f.write(f"\tconvergence_check_frequency: {convergence_check_freq}\n")

    # Track time
    start_time = time.time()
    log_freq = 10  # How many steps between progress updates

    # Optimization loop
    log = []
    window = deque()
    window_sum = 0.0
    step = 1
    with open(log_path, "a") as f:
        f.write("\nRunning optimization:\n")
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
            variational_log_stds=variational_log_std_devs,
        )
        neg_log_lik.backward()
        optimizer.step()

        cur_negll = neg_log_lik.item()

        # Print progress
        if step % log_freq == 0:
            log.append(f"Step {step}, Negative log-likelihood (or -elbo): {cur_negll}, alpha: {alpha.item()}, sigma: {sigma.item()}, thetas: {[theta.item() for theta in theta_dict.values()]}")
            if poisson_logl_mode == "variational":
                log.append(f"\tvariational_means: {[v.item() for v in variational_means]}")
                log.append(f"\tvariational_std_devs: {[v.item() for v in torch.exp(variational_log_std_devs)]}")

            # Write to file periodically
            if step % (10 * log_freq) == 0:
                with open(log_path, "a") as f:
                    for message in log:
                        f.write(message + "\n")
                log = []

        # Check for improvement
        if step % convergence_check_freq == 0:
            window.append(cur_negll)
            window_sum += cur_negll
            if len(window) > window_size:
                window_sum -= window.popleft()
            window_avg = window_sum / len(window)
            if len(window) == window_size and abs(window_avg - cur_negll) < convergence:
                break

        prev_negll = cur_negll
        step += 1

    # Print total time
    end_time = time.time()
    with open(log_path, "a") as f:
        for message in log:
            f.write(message + "\n")
        f.write(
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
        variational_log_stds=variational_log_std_devs,
    )
    with open(log_path, "a") as f:
        f.write("\nFinal state:")
    print_state(optimal_neg_log_lik, alpha, sigma, theta_dict, log_path)
    if poisson_logl_mode == "variational":
        with open(log_path, "a") as f:
            f.write(f"\tvariational_means: {[v.item() for v in variational_means]}\n")
            f.write(f"\tvariational_std_devs: {[v.item() for v in torch.exp(variational_log_std_devs)]}")

    return (
        optimal_neg_log_lik,
        alpha.detach(),
        sigma.detach(),
        {regime: theta.detach() for regime, theta in theta_dict.items()},
    )
