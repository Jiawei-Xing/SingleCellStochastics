import numpy as np


def get_bm_expr_one_branch(
    parent_expr: float, branch_length: float, sigma2: float
) -> float:
    """
    Simulate gene expression for a single branch under the Brownian Motion (BM) process.

    Args:
        parent_expr (float): Expression value at the parent node.
        branch_length (float): Length of the branch.
        sigma2 (float): Variance parameter for the BM process.

    Returns:
        float: Simulated expression value at the child node.
    """
    var = sigma2 * branch_length
    std = np.sqrt(np.maximum(var, 1e-10))
    new_expr = np.random.normal(loc=parent_expr, scale=std)
    return new_expr
