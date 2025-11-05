import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cells", type=str, required=True, help="File path of clonal cells")
parser.add_argument("-d", "--division", type=str, required=True, help="File path of cell division history")
parser.add_argument("-m", "--migration", type=str, required=True, help="File path of cell migrations")
parser.add_argument("-p", "--proportion", type=float, default=1.0, help="Sample proportion of each clone")
parser.add_argument("-s", "--sample", type=str, default="sampled_cells.txt", help="File path of sampled cells")
parser.add_argument("-t", "--tree", type=str, default="tree.nwk", help="File path of output tree")
parser.add_argument("-r", "--regime", type=str, default="regime.csv", help="File path of output regime")
parser.add_argument("-r2", "--regime_mrca", type=str, default="regime_mrca.csv", help="File path of output regime in mrca format")
args = parser.parse_args()

# read input parameters
cell_file = args.cells
division_file = args.division
migration_file = args.migration
p = args.proportion
sample_file = args.sample
tree_file = args.tree
regime_file = args.regime
regime_mrca_file = args.regime_mrca

# each node has label, generation, regime, parent, and children
class Node:
    def __init__(self, label):
        self.label = label
        self.gen = None
        self.regime = None
        self.parent = None
        self.children = []
        
# select nodes for tree from sampled cells to root retrospectively
def selectNodes(keepNodes, root, nodesTree={}):
    # keep nodes for tree
    for k in keepNodes:
        if k.label not in nodesTree:
            nodesTree[k.label] = k

    # recurse to keep parents until reaching the root
    if root in keepNodes:
        return nodesTree
    else:
        return selectNodes(keepNodes=[k.parent for k in keepNodes], root=root, nodesTree=nodesTree)
    
# build tree from root with selected nodes prospectively
def buildTree(node, nodesTree):
    # accumulate branch length for nodes with only one child
    def compress_single_child_path(n):
        branch_length = 1
        while len(n.children) == 1:  # Continue down single-child branches
            branch_length += 1  # Increment occurrence count
            n = n.children[0]
        return f"{buildTree(n, nodesTree=nodesTree)}:{str(branch_length)}"
    
    # Base case: if the node is a leaf, return its label
    if not node.children:
        return node.label

    # Recursive case: build the subtree for each child
    subtree = []
    for child in node.children:
        if len(child.children) == 1:
            # Compress single-child branch and add occurrence as branch length
            subtree.append(compress_single_child_path(child))
        else:
            subtree.append(buildTree(child, nodesTree=nodesTree) + ":1")  # Standard recursive call

    # Combine all child subtrees for the current node
    return f"({','.join(subtree)}){node.label}"

# read clone cells
random.seed(42)
samples = {}
with open(cell_file) as f:
    for line in f:
        # get clone size
        if line.startswith("Clone size: "):
            n = int(line.strip().split(": ")[-1])
        
        # sample cells from clone
        elif line[0].isdigit():
            cells = line.strip().strip(',').split('\t')[-1].split(',')
            cells = ["c" + cell for cell in cells]
            site = line.split('\t')[0]
            mutations = line.split('\t')[1]        
            samples[(site, mutations)] = random.sample(cells, round(n * p))

# output sampled cells
with open(sample_file, "w") as f:
    for clone, cells in samples.items():
        f.write(f"{','.join(clone)}: {','.join(cells)}\n")
        
# read migration cells
migration = {}
with open(migration_file) as f:
    for line in f:
        site = line.split(':')[0].split()[1]
        cell = "c" + line.split()[-1].strip().strip(',')
        migration[cell] = site

# read parent:child pairs
nodes = {}
with open(division_file) as f:
    for line in f:
        gen = line.strip().split()[0]
        parent = "c" + line.strip().split()[1]
        child = "c" + line.strip().split()[2]
        
        # add nodes to dict
        if parent not in nodes:
            nodes[parent] = Node(parent)
        if child not in nodes:
            nodes[child] = Node(child)

        # add node info
        nodes[parent].children.append(nodes[child])
        nodes[child].parent = nodes[parent]
        nodes[parent].gen = int(gen)
        
        # parent node regime
        if not nodes[parent].regime: # regime not defined in childhood (root)
            nodes[parent].regime = migration[parent]
        
        # child node regime
        if child in migration: # migration happened
            nodes[child].regime = migration[child]
        else: # inherit from parent
            nodes[child].regime = nodes[parent].regime

# flatten leaves from samples
leaves = [nodes[cell] for sample in samples.values() for cell in sample]
cells = [cell for sample in samples.values() for cell in sample]

# select and keep leaves and ancestors for tree
nodesTree = selectNodes(keepNodes=leaves, root=nodes['c0'])

# only keep children in tree
for node in nodesTree.values():
    node.children = [n for n in node.children if n.label in nodesTree]

# build newick tree from root
nwk = buildTree(node=nodes['c0'], nodesTree=nodesTree) + ";"
with open(tree_file, 'w') as f:
    f.write(nwk)

# output regime file with node and regime
with open(regime_file, 'w') as f:
    f.write("node_name,regime\n")
    for node in nodesTree.values():
        if not node.children:  # leaf nodes
            f.write(f"{node.label},{node.regime}\n")
        elif len(node.children) > 1:  # internal nodes with more than one child
            f.write(f"{node.label},{node.regime}\n")
    f.write(f"c0,{nodes['c0'].regime}\n")  # always include root

# regime file with cell pairs and regimes of their mrca
mrca = []
with open(regime_mrca_file, 'w') as f:
    f.write('node,node2,regime\n')
    # regime leaf cells
    for cell in cells:
        f.write(f"{cell},,{nodes[cell].regime}\n")
        
    # regime of internal nodes
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            m, n = cells[i], cells[j]
            # MRCA of cell pair
            while True:
                m = nodes[m].parent.label
                n = nodes[n].parent.label
                if m == n:
                    break
                    
            # prevent duplication
            if m in mrca: 
                continue
            else:
                f.write(f"{cells[i]},{cells[j]},{nodes[m].regime}\n")
                mrca.append(m)

