
from Bio import Phylo
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson

from .transform import transform_latent_expression_values


def get_poisson_sampled_read_counts(
    tree: Phylo.BaseTree.Tree
) -> None:
    """
    Simulate observed read counts at the tips of the tree by sampling from a Poisson distribution
    with the latent expression values as the rate parameter.
    
    Args:
        tree (Tree): A Biopython `Tree` object with clamped expression values.
        
    Returns:
        None
    """
    for node in tree.get_terminals():
        node.read_count = np.random.poisson(node.expr)
        
        
def expected_log_poisson_mc(
    q: Normal, 
    y_t: torch.Tensor,
    transformation: str = "softplus",
    n_samples: int = 1000
) -> torch.Tensor:
    """
    Monte Carlo estimate of E_q[log p(y|X)] where p(y|X) is Poisson(rate=transform(X))
    
    Args:
        q: torch.distributions.Normal representing variational distribution
        y_t: observed counts at tips, shape (n_tips,)
        n_samples: number of Monte Carlo samples
    
    Returns:
        expected_log_p_y: scalar torch.Tensor
    """
    Z_samples = q.rsample((n_samples,))
    lambda_s = transform_latent_expression_values(Z_samples, transformation)
    log_p_y_given_x = Poisson(rate=lambda_s).log_prob(y_t).sum(dim=-1)
    expected_log_p_y = log_p_y_given_x.mean()
    return expected_log_p_y



