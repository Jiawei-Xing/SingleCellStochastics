from collections import deque
import numpy as np
from scipy.stats import multivariate_normal
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree, Clade
from typing import Dict, List, Tuple, Optional
import torch
from torch.distributions import MultivariateNormal, Poisson, Normal
import torch.nn.functional as F

from .transform import transform_latent_expression_values
from .poisson import expected_log_poisson_mc


def get_ou_expr_one_branch(
    parent_expr: float, optim: float, alpha: float, branch_length: float, sigma2: float
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


def preprocess_tree(tree: Tree) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Precompute tip distances and MRCA (most recent common ancestor) distances
    for a given phylogenetic tree.

    Args:
        tree (Tree): A Biopython Phylo tree object.

    Returns:
        tip_names (List[str]): Ordered list of terminal (tip) names.
        tip_dist (torch.Tensor): 1-D tensor of shape (n_tips,) containing
            the root-to-tip distances for each tip.
        mrca_dist (torch.Tensor): 2-D tensor of shape (n_tips, n_tips)
            where mrca_dist[i, j] is the root-to-MRCA distance
            between tips i and j.
    """
    # Precompute distances from root to every node
    root_dist = {}

    def dfs(node: Clade, dist: float) -> None:
        root_dist[node] = dist + (node.branch_length or 0.0)
        for clade in node.clades:
            dfs(clade, root_dist[node])

    dfs(tree.root, 0.0)

    tips = tree.get_terminals() # Should match order in mean calculation
    n = len(tips)
    tip_names = [tip.name for tip in tips]

    # Precompute MRCA distance matrix
    mrca_dist = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i, n):
            mrca = tree.common_ancestor(tips[i], tips[j])
            dist = root_dist[mrca]
            mrca_dist[i, j] = dist
            mrca_dist[j, i] = dist

    tip_dist = torch.tensor([root_dist[t] for t in tips], dtype=torch.float32)
    return tip_names, tip_dist, mrca_dist


def compute_ou_covariance(
    tip_names: List[str],
    tip_dist: torch.Tensor,
    mrca_dist: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the OU covariance matrix among tips in a tree.

    Args:
        tip_names: Ordered list of terminal (tip) names corresponding to tip_dist and mrca_dist order.
        tip_dist: 1-D torch.Tensor of shape (n_tips,) containing root-to-tip distances.
        mrca_dist: 2-D torch.Tensor of shape (n_tips, n_tips) containing root-to-MRCA distances for each pair of tips.
        alpha: Alpha for the OU process as a torch.tensor with (requires_grad=True)
        sigma: Sigma for the OU process as a torch.tendor with (requires_grad=True)

    Returns:
        cov: torch.Tensor of shape (n_tips, n_tips)
    """
    n = len(tip_names)

    # Initialize covariance matrix
    cov_matrix = torch.zeros((n, n), dtype=torch.float32)

    # Square sigma to get sigma^2 (this prevents negative values when optimizing over sigma^2 directly)
    sigma2 = sigma**2

    # Precompute other reused calculations
    two_alpha = 2.0 * alpha
    sigma2_over_2alpha = sigma2 / two_alpha

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # If same tip for both i and j, use variance formula
                cov_matrix[i, j] = sigma2_over_2alpha * (
                    1.0 - torch.exp(-two_alpha * tip_dist[i])
                )
            else:
                # If different tips for i and j, use covariance formula
                cov = (
                    sigma2_over_2alpha
                    * torch.exp(-alpha * (tip_dist[i] + tip_dist[j] - (2.0 * mrca_dist[i, j])))
                    * (1.0 - torch.exp(-two_alpha * mrca_dist[i, j]))
                )
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov  # Symmetric matrix

    return cov_matrix


def compute_ou_covariance_fast(
    tip_dist: torch.Tensor,
    mrca_dist: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Ornstein–Uhlenbeck (OU) covariance matrix among tree tips
    using precomputed root-to-tip and root-to-MRCA distances.

    Args:
        tip_dist (torch.Tensor): 1-D tensor of shape (n_tips,) containing
            root-to-tip distances.
        mrca_dist (torch.Tensor): 2-D tensor of shape (n_tips, n_tips) containing
            root-to-MRCA distances for each pair of tips.
        alpha (torch.Tensor): Scalar OU strength parameter (requires_grad=True
            if used in optimization).
        sigma (torch.Tensor): Scalar OU diffusion parameter (requires_grad=True
            if used in optimization).

    Returns:
        torch.Tensor: OU covariance matrix of shape (n_tips, n_tips).
    """
    two_alpha = 2.0 * alpha
    sigma2_over_2alpha = (sigma ** 2) / two_alpha

    d_i = tip_dist.view(-1, 1)   # column vector
    d_j = tip_dist.view(1, -1)   # row vector
    m = mrca_dist

    # Pairwise covariance
    cov = sigma2_over_2alpha * torch.exp(-alpha * (d_i + d_j - 2 * m)) \
          * (1.0 - torch.exp(-two_alpha * m))

    # Replace diagonal with variance formula
    diag = sigma2_over_2alpha * (1.0 - torch.exp(-two_alpha * tip_dist))
    cov[torch.arange(cov.size(0)), torch.arange(cov.size(0))] = diag
    return cov


def compute_ou_mean_with_regimes(
    tree: Phylo.BaseTree.Tree,
    alpha: torch.Tensor,
    origin_expression: torch.Tensor,
    theta_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute mean tip values under OU process using BFS from origin to root to tips.
    Avoids recalculating internal node values multiple times.
    """
    # Evolve from origin to root node, if any branch length exists
    root_branch_length = tree.root.branch_length if tree.root.branch_length else 0.0
    if root_branch_length > 0.0:
        t = torch.tensor(root_branch_length, dtype=torch.float32)
        regime = getattr(tree.root, "regime")
        theta = theta_dict[regime]
        root_expression = theta + (origin_expression - theta) * torch.exp(-alpha * t)
    else:
        root_expression = origin_expression
    
    # Standard BFS from root down
    node_mu = {tree.root: root_expression}
    queue = deque([tree.root])

    while queue:
        parent = queue.popleft()
        parent_mu = node_mu[parent]

        for child in parent.clades:
            t = torch.tensor(
                child.branch_length if child.branch_length else 0.0, dtype=torch.float32
            )
            regime = getattr(child, "regime")
            theta = theta_dict[regime]
            node_mu[child] = theta + (parent_mu - theta) * torch.exp(-alpha * t)
            queue.append(child)

    # Collect tip values
    tips = tree.get_terminals() # Should match order in cov calculation
    tip_values = torch.stack([node_mu[tip] for tip in tips])
    return tip_values


def oup_neg_log_likelihood(
    tree: Phylo.BaseTree.Tree,
    y_t: torch.Tensor,
    tip_names: List[str],
    tip_dist: torch.Tensor,
    mrca_dist: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch,
    theta_dict: Dict[str, torch.Tensor],
    transformation: str = "softplus",
    poisson_logl_mode: str = "deterministic",
    variational_means: torch.Tensor = None,
    variational_log_stds: torch.Tensor = None,
    null_regime: str = "0"
) -> float:
    """
    Compute the log-likelihood of an OU process on a phylogenetic tree
    with regime-specific theta values.

    Args:
        tree: Biopython `Tree` object with assigned regimes and tip read_counts.
        y_t (torch.Tensor): Observed tip read counts as a 1-D tensor of shape (n_tips,).
        tip_names (List[str]): Ordered list of terminal (tip) names corresponding to tip_dist and mrca_dist order.
        tip_dist (torch.Tensor): 1-D tensor of shape (n_tips,) containing root-to-tip distances.
        mrca_dist (torch.Tensor): 2-D tensor of shape (n_tips, n_tips) containing root-to-MRCA distances for each pair of tips.
        alpha (float): Selective strength parameter for the OU process.
        sigma (float): Sigma standard deviation parameter for the OU process.
        theta_dict (dict): A dictionary mapping regime labels to optimal expression values (theta).
        transformation (str): Transformation to apply to mean before Poisson, either "softplus" or "exp".
        poisson_logl_mode (str): Mode for Poisson sampling, either "deterministic", "stochastic", or "variational".
        variational_means (torch.Tensor): Mean parameters for variational distribution (required if poisson_logl_mode="variational").
        variational_log_stds (torch.Tensor): Log standard deviation parameters for variational distribution (required if poisson_logl_mode="variational").
        null_regime (str): Regime label for the null hypothesis (used to initialize origin expression from input thetas). Default is "0".

    Returns:
        log_lik: Log-likelihood of observed tip read_counts.
    """
    # Compute covariance matrix
    cov = compute_ou_covariance(tip_names, tip_dist, mrca_dist, alpha, sigma)

    # Compute mean vector
    origin_expression = theta_dict[null_regime]
    mean = compute_ou_mean_with_regimes(tree, alpha, origin_expression, theta_dict)

    # Multivariate normal log-likelihood
    mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)

    # Decide whether we use the variational elbo through mean-field gaussian assumption or attempt direct log-likelihood computation
    if poisson_logl_mode == "variational":
        if variational_means is None or variational_log_stds is None:
            raise ValueError(
                "variational_means and variational_log_stds must be provided when poisson_logl_mode='variational'."
            )
        if variational_means.shape[0] != len(y_t) or variational_log_stds.shape[
            0
        ] != len(y_t):
            raise ValueError(
                "variational_means and variational_log_stds must match number of tips."
            )

        std_q = torch.exp(variational_log_stds)
        q = Normal(variational_means, std_q)

        # Calculate E_q[log p(y|z)] for the transformation/poisson contribution using z samples from q(z)
        log_p_y = expected_log_poisson_mc(q, y_t, transformation=transformation)

        # Calculate E_q[log p(X)] for the OU contribution in closed form
        n_tips = len(y_t)
        log_2pi = torch.log(
            torch.tensor(2.0 * torch.pi, dtype=cov.dtype, device=cov.device)
        )
        log_det_cov = torch.logdet(cov)
        diff = variational_means - mean
        Sigma_ou_inv = torch.linalg.inv(cov)
        trace_term = torch.sum(std_q**2 * torch.diagonal(Sigma_ou_inv))
        quad_term = diff @ Sigma_ou_inv @ diff
        log_p_z = -0.5 * (n_tips * log_2pi + log_det_cov + trace_term + quad_term)

        # Compute -E_q[log q(z)] for entropy of variational distribution in closed form
        log_q_z = q.entropy().sum()

        elbo = log_p_y + log_p_z + log_q_z
        neg_log_lik = -elbo
    else:
        ou_log_lik = mvn.log_prob(y_t)
        neg_log_lik = -ou_log_lik

        # Add softplus and poisson to log-likelihood
        if poisson_logl_mode == "none":
            # Do not add any transformation/poisson, just return the OU negative log-likelihood
            pass
        elif poisson_logl_mode == "deterministic":
            # Directly use OU expected means as latent expression values
            lambda_tip = transform_latent_expression_values(mean, transformation)
            poisson_log_lik = Poisson(rate=lambda_tip).log_prob(y_t).sum()
            neg_log_lik = neg_log_lik - poisson_log_lik
        elif poisson_logl_mode == "stochastic":
            # Stochastically sample latent expression values from OU distribution at tips and average trasformation/poisson over samples
            n_samples = 1000
            X_samples = mvn.rsample((n_samples,))
            lambda_s = transform_latent_expression_values(X_samples, transformation)
            poisson_log_lik_per_sample = (
                Poisson(rate=lambda_s).log_prob(y_t).sum(dim=-1)
            )
            mean_poisson_log_lik = poisson_log_lik_per_sample.mean()
            neg_log_lik = neg_log_lik - mean_poisson_log_lik
        else:
            raise ValueError(
                "poisson_logl_mode must be one of 'none', 'deterministic', 'stochastic', or 'variational'."
            )

    return neg_log_lik
