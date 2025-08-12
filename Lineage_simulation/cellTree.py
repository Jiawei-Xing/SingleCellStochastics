import random

# set sample proportion
p = 1.0
samples = {}

with open("cloneCells.txt") as f1:
    for line in f1:
        
        # get clone size
        if line.startswith("Clone size: "):
            n = int(line.strip().split(": ")[-1])
        
        # sample cells from clone
        elif line[0] in map(str, range(10)):
            cells = line.strip().split('\t')[-1].split(',')[:-1]
            mutations = line.split('\t')[1]
            samples[mutations] = random.sample(cells, round(n * p))

# output sampled cells for each clone
with open("sampledCells.txt", 'w') as f2:
    for m, s in samples.items():
        f2.write(m + "\t" + ','.join(s) + "\n")


""" # add {child:parent} pairs for each generation into division
division = []
with open("cellDivisionHistory.txt") as f3:
    for line in f3:
        generation = int(line.split('\t')[0])
        if len(division) < generation + 1:
            division.append({})
        else:
            parent = line.split('\t')[1]
            child = line.strip().split('\t')[-1]
            division[generation][child] = parent

# flatten cells in samples; they are tree leaves
nodes = [cell for sample in samples.values() for cell in sample]
#divisionTree = []
pairTree = {}

# loop over division reversely (generations from new to old)
for i in range(len(division) - 1, -1, -1):

    # only keep children (and their parents) present in nodes; they are used for tree building
    #divisionTree.append([{child: division[i][child]} for child in division[i] if child in nodes])
    pairs =  [(division[i][child], child) for child in division[i] if child in nodes]
    pairTree += pairs

    # update parents as new nodes
    #nodes = []
    #for pair in divisionTree[-1]:
    #    nodes += list(pair.values())
    nodes = [pair[0] for pair in pairs]
 """


# each node has a label, children, and a parent
class Node:
    def __init__(self, label=None):
        self.label = label
        self.children = []
        self.parent = None

# read parent:child pairs
nodes = {}
with open("cellDivisionHistory.txt") as f3:
    for line in f3:

        parent = line.split("\t")[1]
        child = line.strip().split("\t")[2]

        if parent not in nodes:
            nodes[parent] = Node(parent)    
        if child not in nodes:
            nodes[child] = Node(child)

        nodes[parent].children.append(nodes[child])
        nodes[child].parent = nodes[parent]

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

def buildTree(node, nodesTree=nodesTree):
    # recurse until reaching leaves
    if node.children == []:
        return node.label
    else:
        subtree = [buildTree(child) for child in node.children if child.label in nodesTree]
        return "(" + ",".join(subtree) + ")" + node.label

# build newick tree from root
nwk = buildTree(node=nodes['0'])
print(nwk)
