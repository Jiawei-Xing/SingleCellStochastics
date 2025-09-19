
from Bio import Phylo
import math


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