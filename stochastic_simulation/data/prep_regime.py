from Bio import Phylo
import argparse

parser = argparse.ArgumentParser(description="Preparing regime files from tree and simple cell lineage.")
parser.add_argument("-t", "--tree", type=str, required=True, help="Newick tree file")
parser.add_argument("-c", "--cells", type=str, required=True, help="Lineage file with cells of other regimes")
parser.add_argument("-r", "--regime", type=str, default="regime.csv", help="Regime file")
args = parser.parse_args()

# get leaf of mrca
def get_leaf(node):
    if node.is_terminal():
        return node.name
    else:
        return get_leaf(node.clades[0])

# read input tree and leaf cell lineage
tree = Phylo.read(args.tree, 'newick')
lineage = {}
with open(args.cells) as f1:
    for line in f1:
        regime = line.split()[0]
        cells = line.strip().split()[1].split(",")
        for cell in cells:
            lineage[cell] = regime

# output leaf in regime file
with open(args.regime, 'w') as f2:
    f2.write(f"node,node2,regime\n")
    for cell in tree.get_terminals():
        f2.write(f"{cell.name},,{lineage[cell.name]}\n")

# read cell lineage line by line in order
output = {}
with open(args.cells) as f3:
    for line in f3:
        lineage = {}
        regime = line.split()[0]
        cells = line.strip().split()[1].split(",")
        for cell in cells:
            lineage[cell] = regime

        for node in tree.find_clades():
            # internal node
            if node.is_terminal() or len(node.clades) == 1:
                continue

            # leaf of mrca
            leaf1 = get_leaf(node.clades[0])
            leaf2 = get_leaf(node.clades[1])

            # if both leaf in lineage, add to output
            if leaf1 in lineage and leaf2 in lineage:
                output[(leaf1, leaf2)] = lineage[leaf1]

# add to regime file
with open(args.regime, 'a') as f4:
    for (leaf1, leaf2) in output:
        f4.write(f"{leaf1},{leaf2},{output[(leaf1, leaf2)]}\n")

