import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import os
import sys

regime_file = sys.argv[1]
expr_file = sys.argv[2]

# cells in metastatic regime
met = []
with open(regime_file) as f:
    for line in f:
        leaf1, leaf2, regime = line.strip().split(",")
        if leaf2 == "" and regime == "1":
            met.append(leaf1)

# read count
count_matrix = pd.read_csv(expr_file, index_col=0, header=0, sep="\t")

# set groups
adata = ad.AnnData(count_matrix)
group = [
    "met" if cell in met else "pri"
    for cell in adata.obs.index
]
adata.obs["group"] = group

# log transformation
sc.pp.log1p(adata)

# Perform DEA using the Wilcoxon method between the "met" and "pri" groups
sc.tl.rank_genes_groups(adata, groupby="group", method="wilcoxon")

# Extract the results and store them in a DataFrame
de_results = adata.uns["rank_genes_groups"]
df_de = pd.DataFrame({
    "gene": de_results["names"]["met"],
    "log2FC": de_results["logfoldchanges"]["met"],
    "p_value": de_results["pvals"]["met"],
    "q_value": de_results["pvals_adj"]["met"],
    "signif": de_results["pvals_adj"]["met"] < 0.05
})

# Save the results to a file
df_de.to_csv("DEA_" + expr_file.split("/")[-1], sep="\t", index=False)
