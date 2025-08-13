import random
import argparse

# sample cells from clones
parser = argparse.ArgumentParser()
parser.add_argument("--cell_file", type=str, required=True)
parser.add_argument("--history_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("-p", type=float, default=1.0)
args = parser.parse_args()

cell_file = args.cell_file
history_file = args.history_file
output_file = args.output_file
p = args.p

# read clone cells
samples = {}
with open(cell_file) as f1:
    for line in f1:
        # get clone size
        if line.startswith("Clone size: "):
            n = int(line.strip().split(": ")[-1])
        
        # sample cells from clone
        elif line[0].isdigit():
            cells = line.strip().split('\t')[-1].split(',')[:-1]
            site = line.split('\t')[0]
            mutations = line.split('\t')[1]        
            samples[(site, mutations)] = random.sample(cells, round(n * p))

# select sampled cells from history and build newick tree
class Node:
    def __init__(self, label=None):
        self.label = label
        self.children = []
        self.parent = None
        self.gen = 0

# read parent:child pairs
nodes = {}
with open(history_file) as f2:
    for line in f2:
        gen = line.split("\t")[0]
        parent = line.split("\t")[1]
        child = line.strip().split("\t")[2]

        if parent not in nodes:
            nodes[parent] = Node(parent)    
        if child not in nodes:
            nodes[child] = Node(child)

        nodes[parent].children.append(nodes[child])
        nodes[child].parent = nodes[parent]
        nodes[parent].gen = gen

# flatten leaves from samples
leaves = [nodes[cell] for sample in samples.values() for cell in sample]

def selectNodes(keepNodes, nodesTree={}, root=nodes['0']):
    # keep nodes
    for k in keepNodes:
        if k.label not in nodesTree:
            nodesTree[k.label] = k

    # recurse to keep parents until reaching the root
    if root in keepNodes:
        return nodesTree
    else:
        return selectNodes(keepNodes=[k.parent for k in keepNodes], nodesTree=nodesTree)

# select and keep leaves and ancestors
nodesTree = selectNodes(keepNodes=leaves)

for node in nodesTree.values():
    node.children = [n for n in node.children if n.label in nodesTree]

def buildTree(node, nodesTree=nodesTree):
    # Helper function to accumulate branch length for nodes with only one child
    def compress_single_child_path(n):
        branch_length = 1
        while len(n.children) == 1:  # Continue down single-child branches
            branch_length += 1  # Increment occurrence count
            n = n.children[0]
        return buildTree(n) + ":" + str(branch_length)
    
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
            subtree.append(buildTree(child) + ":1")  # Standard recursive call

    # Combine all child subtrees for the current node
    return "(" + ",".join(subtree) + ")" + node.label

# build newick tree from root
nwk = buildTree(node=nodes['0']) + ";"
with open(output_file, 'w') as f:
    f.write(nwk)

