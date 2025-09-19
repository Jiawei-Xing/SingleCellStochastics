from Bio.Phylo.BaseTree import Clade, Tree
from Bio import Phylo
import random


def birth_tree(
    n_tips: int,
    outfile: str,
    make_ultrametric: bool = False
) -> None:
    """
    Birth tree with n_tips, assign names to tips and internal nodes,
    and write to outfile in Newick format with branch lengths. Note:
    this is not a realistic tree simulation, just a simple birth process
    to get a tree.
    
    Args:
        n_tips: Number of tips in the tree (int).
        outfile: File path to write the Newick tree (str).
        make_ultrametric: If True, adjust branch lengths to make the tree ultrametric (bool).
    
    Returns:
        None
    """
    # Initialize tips
    tips = [Clade(branch_length=1.0, name=f"tip{i+1}") for i in range(n_tips)]
    next_internal = 1
    nodes = tips[:]
    while len(nodes) > 1:
        # Randomly pick two nodes to merge
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        # Create new internal node
        internal = Clade(branch_length=1.0, name=f"node{next_internal}", clades=[a, b])
        next_internal += 1
        nodes.append(internal)
    # The last node is the root
    tree = Tree(root=nodes[0])
    
    # Optionally make ultrametric by adjusting tip branch lengths
    if make_ultrametric:
        max_height = max([tree.distance(tree.root, tip) for tip in tree.get_terminals()])
        for tip in tree.get_terminals():
            tip.branch_length += max_height - tree.distance(tree.root, tip)
    
    Phylo.write(tree, outfile, "newick")
