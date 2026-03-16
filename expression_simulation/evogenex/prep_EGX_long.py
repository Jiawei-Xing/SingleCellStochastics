import sys

counts = {}
in_file = sys.argv[1]
out_file = sys.argv[2]

# read count matrix (cell by gene)
with open(in_file) as f:
    for line in f:
        if line[0] == "\t":
            n = len(line.strip().split())
            continue
        counts[line.split()[0]] = line.strip().split()

# output long format for evogenex
with open(out_file, 'w') as f:
    f.write("gene,species,replicate,exprval\n")
    for i in range(1, n + 1):
        for cell, value in counts.items():
            f.write(f"{i},{cell},R1,{value[i]}\n")
