
from Bio import Phylo
import math
import torch
import torch.nn.functional as F


def clamp_latent_gene_expression_at_tips(
    tree: Phylo.BaseTree.Tree, 
    method: str = "clamp"
) -> None:
    """
    Clamp or transform negative latent gene expression values at the tips of the tree.
    
    Args:
        tree (Tree): A Biopython `Tree` object with simulated expression values.
        method (str): Method to handle negative expressions. Options are "clamp" (set to zero) or "softplus" (apply softplus transformation). Default is "clamp".
        
    Returns:
        None
    """
    for node in tree.get_terminals():
        if method == "clamp":
            node.expr = max(0, node.expr)
        elif method == "softplus":
            node.expr = math.log(1 + math.exp(node.expr))
        else:
            raise ValueError("method input must be clamp or softplus")
        

def transform_latent_expression_values(
    mean: torch.Tensor,
    transformation: str = "softplus"
) -> torch.Tensor:
    """
    Transform latent expression values to be non-negative using specified transformation.
    
    Args:
        mean (torch.Tensor): Latent expression values.
        transformation (str): Transformation method. Options are "exp" (exponential) or "softplus". Default is "softplus".
        
    Returns:
        torch.Tensor: Transformed non-negative expression values.
    """
    if transformation == "exp":
        return torch.exp(mean)
    elif transformation == "softplus":
        return F.softplus(mean)
    else:
        raise ValueError("transformation must be either 'exp' or 'softplus'")