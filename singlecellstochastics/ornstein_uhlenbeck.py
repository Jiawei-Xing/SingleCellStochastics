
from collections import deque
import numpy as np
from scipy.stats import multivariate_normal
from Bio import Phylo
from typing import Dict, List, Tuple
import torch
from torch.distributions import MultivariateNormal, Poisson
import torch.nn.functional as F

from .tree_utils import collect_tip_data
from .transform import transform_latent_expression_values

def get_ou_expr_one_branch(
    parent_expr: float,
    optim: float,
    alpha: float,
    branch_length: float,
    sigma2: float
) -> float:
    """
    Simulate gene expression for a single branch under the Ornstein-Uhlenbeck (OU) process.

    Args:
        parent_expr (float): Expression value at the parent node.
        optim (float): Optimal expression value (theta) for the OU process.
        alpha (float): Selective strength parameter for the OU process.
        branch_length (float): Length of the branch.
        sigma2 (float): Variance parameter for the OU process.

    Returns:
        float: Simulated expression value at the child node.
    """
    mean = optim + (parent_expr - optim) * np.exp(-alpha * branch_length)
    var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * branch_length))
    std = np.sqrt(np.maximum(var, 1e-10))
    new_expr = np.random.normal(loc=mean, scale=std)
    return new_expr


def compute_ou_covariance(
    tree: Phylo.BaseTree.Tree,
    dist_to_root: Dict[str, float],
    alpha: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the OU covariance matrix among tips in a tree.

    Args:
        tree: Biopython Tree object.
        dist_to_root: dict mapping tip names to their distances from the root.
        alpha: torch scalar (requires_grad=True)
        sigma2: torch scalar (requires_grad=True)
    
    Returns:
        cov: torch.Tensor of shape (n_tips, n_tips)
    """
    tips = tree.get_terminals()
    tip_names = [tip.name for tip in tips]
    n = len(tips)
    
    cov_matrix = torch.zeros((n, n), dtype=alpha.dtype, device=alpha.device)

    for i in range(n):
        for j in range(n):
            t_i = torch.tensor(dist_to_root[tip_names[i]], dtype=torch.float32)
            t_j = torch.tensor(dist_to_root[tip_names[j]], dtype=torch.float32)

            if i == j:
                cov_matrix[i, j] = (sigma2 / (2.0 * alpha)) * (1.0 - torch.exp(-2.0 * alpha * t_i))
            else:
                mrca = tree.common_ancestor(tips[i], tips[j])
                t_mrca = torch.tensor(tree.distance(tree.root, mrca), dtype=alpha.dtype, device=alpha.device)
                cov_matrix[i, j] = (sigma2 / (2.0 * alpha)) * torch.exp(
                    -alpha * (t_i + t_j - 2.0 * t_mrca)
                ) * (1.0 - torch.exp(-2.0 * alpha * t_mrca))
    
    return cov_matrix


def compute_ou_mean_with_regimes(
    tree: Phylo.BaseTree.Tree,
    alpha: torch.Tensor,
    root_expression: torch.Tensor,
    theta_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute mean tip values under OU process using BFS from root to tips.
    Avoids recalculating internal node values multiple times.
    """
    node_mu = {tree.root: root_expression}
    queue = deque([tree.root])
    
    while queue:
        parent = queue.popleft()
        parent_mu = node_mu[parent]
        
        for child in parent.clades:
            t = torch.tensor(child.branch_length if child.branch_length else 0.0, dtype=torch.float32)
            regime = getattr(child, "regime")
            theta = theta_dict[regime]
            node_mu[child] = theta + (parent_mu - theta) * torch.exp(-alpha * t)
            queue.append(child)
    
    # Collect tip values
    tips = tree.get_terminals()
    tip_values = torch.stack([node_mu[tip] for tip in tips])
    return tip_values


def oup_neg_log_likelihood(
    tree: Phylo.BaseTree.Tree,
    alpha: torch.Tensor,
    sigma: torch,
    theta_dict: Dict[str, torch.Tensor],
    root_expression: torch.Tensor,
    transformation: str = "softplus",
    poisson_logl_mode: str = "deterministic",
) -> float:
    """
    Compute the log-likelihood of an OU process on a phylogenetic tree
    with regime-specific theta values.
    
    Args:
        tree (Tree): A Biopython `Tree` object with assigned regimes and tip read_counts.
        alpha (float): Selective strength parameter for the OU process.
        sigma (float): Sigma standard deviation parameter for the OU process.
        theta_dict (dict): A dictionary mapping regime labels to optimal expression values (theta).
        root_expression (float): Expression value at the root node.
        transformation (str): Transformation to apply to mean before Poisson, either "softplus" or "exp".
        poisson_sampling_mode (str): Mode for Poisson sampling, either "deterministic", "stochastic", or "variational".

    Returns:
        log_lik: Log-likelihood of observed tip read_counts.
    """
    # Extract tip data
    tip_names, y, dist_to_root = collect_tip_data(tree)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    # Square sigma to get sigma^2 (this prevents negative values when optimizing over sigma^2 directly)
    sigma2 = sigma ** 2

    # Compute covariance matrix
    cov = compute_ou_covariance(tree, dist_to_root, alpha, sigma2)

    # Compute mean vector
    mean = compute_ou_mean_with_regimes(tree, alpha, root_expression, theta_dict)

    # Multivariate normal log-likelihood
    mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)
    ou_log_lik = mvn.log_prob(y_t)
    neg_log_lik = -ou_log_lik
    
    # Add softplus and poisson to log-likelihood
    # Directly use OU expected means as latent expression values
    if poisson_logl_mode == "none":
        pass
    elif poisson_logl_mode == "deterministic":
        lambda_tip = transform_latent_expression_values(mean, transformation)    
        poisson_log_lik = Poisson(rate=lambda_tip).log_prob(y_t).sum()
        neg_log_lik = neg_log_lik - poisson_log_lik
    # Stochastically sample latent expression values from OU distribution at tips and average trasformation/poisson over samples
    elif poisson_logl_mode == "stochastic":
        n_samples = 1000
        X_samples = mvn.rsample((n_samples,))
        lambda_s = transform_latent_expression_values(X_samples, transformation)
        poisson_log_lik_per_sample = Poisson(rate=lambda_s).log_prob(y_t).sum(dim=-1)
        mean_poisson_log_lik = poisson_log_lik_per_sample.mean()
        neg_log_lik = -mean_poisson_log_lik
    elif poisson_logl_mode == "variational":
        ValueError("Variational mode not implemented yet")
    
    return neg_log_lik




