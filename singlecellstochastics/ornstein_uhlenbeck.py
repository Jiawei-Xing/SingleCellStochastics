
import numpy as np
from scipy.stats import multivariate_normal
from Bio import Phylo
from typing import Dict, List, Tuple

from .tree_utils import collect_tip_data

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
    alpha: float,
    sigma2: float
) -> np.ndarray:
    """
    Compute the OU covariance matrix among tips.
    
    Args:
        tree (Tree): A Biopython `Tree` object.
        dist_to_root (dict): A dictionary mapping tip names to their distances from the root.
        alpha (float): Selective strength parameter for the OU process.
        sigma2 (float): Variance parameter for the OU process.

    Returns:
        cov: n x n covariance matrix
    """
    tips = tree.get_terminals()
    tip_names = [tip.name for tip in tips]
    n = len(tips)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = (sigma2/(2*alpha)) * (1 - np.exp(-2*alpha*dist_to_root[tip_names[i]]))
            else:
                mrca = tree.common_ancestor(tips[i], tips[j])
                t_mrca = tree.distance(tree.root, mrca)
                cov[i, j] = (sigma2/(2*alpha)) * np.exp(
                    -alpha * (
                        dist_to_root[tip_names[i]]
                        + dist_to_root[tip_names[j]]
                        - (2 * t_mrca)
                    )
                ) * (1 - np.exp(-2*alpha*t_mrca))
    return cov


def compute_ou_mean_with_regimes(
    tree: Phylo.BaseTree.Tree, 
    alpha: float, 
    root_expression: float, 
    theta_dict: Dict[str, float],
) -> np.ndarray:
    """
    Compute mean tip values accounting for regime changes along each path
    from the root to that tip.
    
    Args:
        tree (Tree): A Biopython `Tree` object with assigned regimes.
        alpha (float): Selective strength parameter for the OU process.
        root_expression (float): Expression value at the root node.
        theta_dict (dict): A dictionary mapping regime labels to optimal expression values (theta).
        
    Returns:
        mean: Array of mean expression values at the tips.
    """
    tips = tree.get_terminals()
    mean = {}
    for tip in tips:
        path = tree.get_path(tip)  # list of clades from root (excluding root)
        mu = root_expression
        prev = tree.root
        for node in path:
            t = prev.branch_length if prev.branch_length else 0.0
            regime = getattr(node, "regime")
            theta = theta_dict[regime]
            mu = theta + (mu - theta) * np.exp(-alpha * t)
            prev = node
        mean[tip.name] = mu
    return np.array([mean[tip.name] for tip in tips])


def ou_neg_log_likelihood(
    tree: Phylo.BaseTree.Tree,
    alpha: float,
    sigma2: float,
    theta_dict: Dict[str, float],
    root_expression: float
) -> float:
    """
    Compute the log-likelihood of an OU process on a phylogenetic tree
    with regime-specific theta values.
    
    Args:
        tree (Tree): A Biopython `Tree` object with assigned regimes and tip read_counts.
        alpha (float): Selective strength parameter for the OU process.
        sigma2 (float): Variance parameter for the OU process.
        theta_dict (dict): A dictionary mapping regime labels to optimal expression values (theta).
        root_expression (float): Expression value at the root node.

    Returns:
        log_lik: Log-likelihood of observed tip read_counts.
    """
    # Extract tip data
    tip_names, y, dist_to_root = collect_tip_data(tree)

    # Compute covariance matrix
    cov = compute_ou_covariance(tree, dist_to_root, alpha, sigma2)

    # Compute mean vector
    mean = compute_ou_mean_with_regimes(tree, alpha, root_expression, theta_dict)

    # Multivariate normal log-likelihood
    log_lik = float(multivariate_normal.logpdf(y, mean=mean, cov=cov, allow_singular=False))
    neg_log_lik = -log_lik
    
    return neg_log_lik