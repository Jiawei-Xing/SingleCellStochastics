"""
Convert run-diff-test output into run-reconst input format.

Usage:
    python prep_reconst_input.py \
        --chi_sq  <result_chi-squared.tsv> \
        --q_params <result_h1_q-mean-std_0.tsv> \
        --expr    <readcounts.tsv> \
        --gene    <gene_col_name> \
        --out_vi  <vi_params.tsv> \
        --out_ou  <ou_params.tsv> \
        [--hypothesis h0|h1]
"""
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi_sq", required=True, help="Chi-squared result TSV from run-diff-test")
    parser.add_argument("--q_params", required=True, help="q-mean-std TSV from run-diff-test")
    parser.add_argument("--expr", required=True, help="Original readcounts TSV (cells x genes)")
    parser.add_argument("--gene", required=True, help="Gene column name to extract")
    parser.add_argument("--out_vi", required=True, help="Output: VI params TSV for run-reconst")
    parser.add_argument("--out_ou", required=True, help="Output: OU params TSV for run-reconst")
    parser.add_argument("--hypothesis", default="h1", choices=["h0", "h1"],
                        help="Which hypothesis params to use (default: h1)")
    args = parser.parse_args()

    h = args.hypothesis  # "h0" or "h1"

    # --- VI params ---
    # q_params: rows=genes, cols=cells (first half: means, second half: stds)
    df_q = pd.read_csv(args.q_params, sep="\t", index_col=0)
    n_cells = len(df_q.columns) // 2
    cells = df_q.columns[:n_cells].tolist()

    df_q.index = df_q.index.astype(str)
    gene_row = df_q.loc[args.gene]
    q_means = gene_row.values[:n_cells]
    q_stds = gene_row.values[n_cells:]

    # Read counts from original expression file, reindexed to match q_params cell order
    df_expr = pd.read_csv(args.expr, sep="\t", index_col=0)
    read_counts = df_expr.loc[cells, args.gene].values

    df_vi = pd.DataFrame({
        'q_mean': q_means,
        'q_std': q_stds,
        'read_count': read_counts,
    }, index=cells)
    df_vi.index.name = 'cell_name'
    df_vi.to_csv(args.out_vi, sep="\t")
    print(f"Wrote {args.out_vi}")

    # --- OU params ---
    df_chi = pd.read_csv(args.chi_sq, sep="\t")
    df_chi['gene'] = df_chi['gene'].astype(str)
    gene_row = df_chi[df_chi['gene'] == args.gene].iloc[0]

    alpha = gene_row[f'{h}_alpha']
    sigma = gene_row[f'{h}_sigma']

    if h == "h0":
        # H0: single shared theta across all regimes
        theta_cols = [c for c in df_chi.columns if c.startswith('h1_theta')]
        regimes = [c.replace('h1_theta', '') for c in theta_cols]
        theta = gene_row['h0_theta']
        rows = [{'regime': r, 'alpha': alpha, 'sigma': sigma, 'theta': theta}
                for r in regimes]
    else:
        # H1: regime-specific thetas
        theta_cols = [c for c in df_chi.columns if c.startswith('h1_theta')]
        regimes = [c.replace('h1_theta', '') for c in theta_cols]
        rows = [{'regime': r, 'alpha': alpha, 'sigma': sigma, 'theta': gene_row[col]}
                for r, col in zip(regimes, theta_cols)]

    df_ou = pd.DataFrame(rows).set_index('regime')
    df_ou.to_csv(args.out_ou, sep="\t")
    print(f"Wrote {args.out_ou}")


if __name__ == "__main__":
    main()
