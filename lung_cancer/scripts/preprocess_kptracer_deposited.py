"""Prepare VINE and LAVOUS inputs from the deposited KP-Tracer dataset."""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build representative KP-Tracer clone inputs for VINE and LAVOUS."
    )
    parser.add_argument(
        "--datadir",
        default=os.environ.get(
            "DEPOSITED_DATA_DIR",
            "/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/real_data/Yang_KP-Tracer",
        ),
        help="Directory containing deposited KP-Tracer character, AnnData, and metadata files.",
    )
    parser.add_argument("--outdir", default="data/vine", help="Directory for VINE inputs.")
    parser.add_argument(
        "--lavous-outdir",
        default="data",
        help="Directory for LAVOUS readcount and library-factor inputs.",
    )
    parser.add_argument("--prefix", default="3724_NT_All", help="Output filename prefix.")
    parser.add_argument(
        "--min-cell-fraction",
        type=float,
        default=0.10,
        help="Keep genes present in at least this fraction of representative cells.",
    )
    parser.add_argument(
        "--character-file",
        default="3724_NT_All_character_matrix.txt",
        help="Character matrix filename under --datadir.",
    )
    parser.add_argument(
        "--adata-file",
        default="adata_processed.combined.h5ad",
        help="AnnData filename under --datadir. raw.X is expected to contain integer UMIs.",
    )
    parser.add_argument(
        "--meta-file",
        default="KPTracer_meta.csv",
        help="Metadata CSV filename under --datadir. Must contain a Tumor column.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.lavous_outdir, exist_ok=True)

    character_path = os.path.join(args.datadir, args.character_file)
    adata_path = os.path.join(args.datadir, args.adata_file)
    meta_path = os.path.join(args.datadir, args.meta_file)

    character = pd.read_csv(character_path, sep="\t", index_col=0)
    adata = ad.read_h5ad(adata_path)
    meta = pd.read_csv(meta_path, index_col=0)

    common_cells = character.index.intersection(adata.obs_names)
    if common_cells.empty:
        raise ValueError("No cells are shared between character matrix and AnnData.")

    character = character.loc[common_cells].copy()
    adata = adata[common_cells].copy()
    print(f"Cells with both barcode and expression: {len(common_cells)}")

    mutation_cols = [col for col in character.columns if col.startswith("r")]
    if not mutation_cols:
        raise ValueError("No mutation columns starting with 'r' found in character matrix.")
    if "total_counts" not in adata.obs:
        raise ValueError("AnnData obs must contain total_counts for representative-cell selection.")
    if "Tumor" not in meta.columns:
        raise ValueError("Metadata must contain a Tumor column for VINE migration labels.")
    if adata.raw is None:
        raise ValueError("AnnData raw.X is required because it stores the integer UMI counts.")

    character["mutation_pattern"] = character[mutation_cols].astype(str).agg("_".join, axis=1)
    adata.obs["mutation_pattern"] = character["mutation_pattern"]

    rep_cells = (
        adata.obs.dropna(subset=["mutation_pattern"])
        .groupby("mutation_pattern")["total_counts"]
        .idxmax()
    )
    print(f"Representative cells (unique clones): {len(rep_cells)}")

    missing_meta = pd.Index(rep_cells.values).difference(meta.index)
    if len(missing_meta):
        raise ValueError(f"{len(missing_meta)} representative cells are missing from metadata.")

    vine_matrix = character.loc[rep_cells.values, mutation_cols].replace("-", "-1")
    vine_matrix_path = os.path.join(
        args.outdir, f"{args.prefix}_character_matrix_representative.tsv"
    )
    vine_matrix.to_csv(vine_matrix_path, sep="\t")
    print(f"Saved character matrix: {vine_matrix.shape} -> {vine_matrix_path}")

    tissue = meta.loc[rep_cells.values, "Tumor"].astype(str).str.replace(
        "3724_NT_", "", regex=False
    )
    state_df = pd.DataFrame({"cell": rep_cells.values, "state": tissue.values})
    state_path = os.path.join(args.outdir, f"{args.prefix}_statelabels.csv")
    state_df.to_csv(state_path, index=False, header=False)
    print(f"Saved state labels: {state_df.shape} -> {state_path}")
    print(state_df["state"].value_counts().to_string())

    adata_rep = adata[rep_cells.values]
    raw = adata_rep.raw.X
    raw_counts = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
    counts = raw_counts.astype(int)

    min_cells = int(args.min_cell_fraction * len(rep_cells))
    gene_mask = (counts > 0).sum(axis=0) >= min_cells
    counts = counts[:, gene_mask]

    expr_df = pd.DataFrame(
        counts,
        index=adata_rep.obs_names,
        columns=adata_rep.raw.var_names[gene_mask],
    )
    expr_path = os.path.join(args.lavous_outdir, f"{args.prefix}_readcounts.tsv")
    expr_df.to_csv(expr_path, sep="\t")
    print(f"Saved expression matrix: {expr_df.shape} -> {expr_path}")

    raw_totals = adata_rep.obs["total_counts"].values.astype(np.float64)
    median_total = np.median(raw_totals)
    if median_total <= 0:
        raise ValueError("Median total_counts is non-positive.")
    lib_factors = raw_totals / median_total

    lib_df = pd.DataFrame(
        lib_factors,
        index=rep_cells.values,
        columns=["library_size"],
    )
    lib_path = os.path.join(args.lavous_outdir, f"{args.prefix}_library.tsv")
    lib_df.to_csv(lib_path, sep="\t", header=False)
    print(f"Saved library size factors: {lib_df.shape} -> {lib_path}")
    print(f"Library factor range: {lib_factors.min():.3f} - {lib_factors.max():.3f}")


if __name__ == "__main__":
    main()
