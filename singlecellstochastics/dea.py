import scanpy as sc
import pandas as pd
import anndata as ad
import argparse
import csv

def run_dea():
    """
    Run DEA using the Wilcoxon method between the "met" and "pri" groups.
    """

    parser = argparse.ArgumentParser(description="Run DEA using the Wilcoxon method between the 'met' and 'pri' groups.")
    parser.add_argument("--regime", type=str, required=True, help="Path to the regime file")
    parser.add_argument("--test", type=str, required=True, help="Regime to test against primary")
    parser.add_argument("--expr", type=str, required=True, help="Path to the expression file")
    parser.add_argument("--outdir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    regime_file = args.regime
    test = args.test
    expr_file = args.expr
    outdir = args.outdir

    # cells in metastatic regime
    met = []
    with open(regime_file) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)

        if header == ["node", "node2", "regime"]:
            for row in csv_file:
                leaf1, leaf2, regime = row
                if leaf2 == "" and regime == test:
                    met.append(leaf1)
        elif header == ["node_name", "regime"]:
            for row in csv_file:
                leaf1, regime = row
                if regime == test:
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
    df_de.to_csv(outdir + "DEA_" + expr_file.split("/")[-1], sep="\t", index=False)


if __name__ == "__main__":
    run_dea()
