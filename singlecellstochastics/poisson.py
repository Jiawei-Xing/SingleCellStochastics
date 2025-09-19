
from Bio import Phylo
import numpy as np

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