import sys

counts = {}
file = sys.argv[1]

# read count matrix (cell by gene)
with open(file) as f:
    for line in f:
        if line[0] == "\t":
            n = len(line.strip().split())
            continue
        counts[line.split()[0]] = line.strip().split()

# output long format for evogenex
with open(f"{file.split('/')[-1][:-4]}_long.csv", 'w') as f:
    f.write("gene,species,replicate,exprval\n")
    for i in range(1, n + 1):
        for cell, value in counts.items():
            f.write(f"{i},{cell},R1,{value[i]}\n")
