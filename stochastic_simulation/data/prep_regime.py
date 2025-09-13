from Bio import Phylo
import argparse

parser = argparse.ArgumentParser(description="Preparing regime files from tree and simple cell lineage.")
parser.add_argument("-t", "--tree", type=str, required=True, help="Newick tree file")
parser.add_argument("-p", "--primary", type=str, default="0", help="Primary regime from root")
parser.add_argument("-c", "--cells", type=str, required=True, help="Lineage file with cells of other regimes")
parser.add_argument("-r", "--regime", type=str, default="regime.csv", help="Regime file")
args = parser.parse_args()

# get leaf of mrca
def get_leaf(node):
    if node.is_terminal():
        return node.name
    else:
        return get_leaf(node.clades[0])

# read input tree and cell lineage
tree = Phylo.read(args.tree, 'newick')
lineage = {}
with open(args.cells) as f1:
    for line in f1:
        regime = line.split()[0]
        cells = line.strip().split()[1].split(",")
        for cell in cells:
            lineage[cell] = regime

for cell in tree.get_terminals():
    if cell.name not in lineage:
        lineage[cell.name] = args.primary

# regime file
with open(args.regime, 'w') as f2:
    f2.write(f"node1,node2,regime\n")
    # leaf
    for cell in tree.get_terminals():
        f2.write(f"{cell.name},,{lineage[cell.name].split('_')[0]}\n")

    # internal node
    for node in tree.find_clades():
        if node.is_terminal() or len(node.clades) == 1:
            continue

        leaf1 = get_leaf(node.clades[0])
        leaf2 = get_leaf(node.clades[1])
        if lineage[leaf1] == lineage[leaf2]:
            f2.write(f"{leaf1},{leaf2},{lineage[leaf1].split('_')[0]}\n")
        else:
            f2.write(f"{leaf1},{leaf2},{args.primary}\n")

