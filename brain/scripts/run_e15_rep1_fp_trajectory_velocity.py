#!/usr/bin/env python
"""Run trajectory inference for E15 rep1 FP-lineage cells.

RNA velocity is reported as unavailable unless spliced/unspliced layers are
present, because velocity cannot be inferred from mature RNA counts alone.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components


FP_ORDER = ["FP_Rgl1", "FP_NbFP", "FP_NbDA", "FP_DA", "FP_NbGLU", "FP_GLUFP"]
CELL_TYPE_ORDER = ["Rgl1", "NbFP", "NbDA", "DA", "NbGLU", "GLUFP"]
FP_EDGES = [
    ("FP_Rgl1", "FP_NbFP"),
    ("FP_NbFP", "FP_NbDA"),
    ("FP_NbDA", "FP_DA"),
    ("FP_NbFP", "FP_NbGLU"),
    ("FP_NbGLU", "FP_GLUFP"),
]
FP_COLORS = {
    "FP_Rgl1": "#000000",
    "FP_NbFP": "#8C8C8C",
    "FP_NbDA": "#FF7F00",
    "FP_DA": "#E41A1C",
    "FP_NbGLU": "#33A02C",
    "FP_GLUFP": "#1F78B4",
    "Rgl1": "#000000",
    "NbFP": "#8C8C8C",
    "NbDA": "#FF7F00",
    "DA": "#E41A1C",
    "NbGLU": "#33A02C",
    "GLUFP": "#1F78B4",
}
FP_LABELS = {
    "FP_Rgl1": "Rgl1",
    "FP_NbFP": "Nb_FP",
    "FP_NbDA": "Nb_DA",
    "FP_DA": "DA",
    "FP_NbGLU": "Nb_GLU",
    "FP_GLUFP": "GLU_FP",
    "Rgl1": "Rgl1",
    "NbFP": "Nb_FP",
    "NbDA": "Nb_DA",
    "DA": "DA",
    "NbGLU": "Nb_GLU",
    "GLUFP": "GLU_FP",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="outputs/e15_rep1_fp_trajectory_velocity",
        help="Output directory.",
    )
    parser.add_argument(
        "--prefix",
        default="e15_rep1_fp_cells",
        help="Input/output filename prefix.",
    )
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--root-regime", default="FP_Rgl1")
    return parser.parse_args()


def read_inputs(outdir, prefix):
    outdir = Path(outdir)
    counts_path = outdir / f"{prefix}_counts.mtx"
    obs_path = outdir / f"{prefix}_obs.tsv"
    genes_path = outdir / f"{prefix}_genes.tsv"
    for path in (counts_path, obs_path, genes_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    counts = mmread(counts_path).tocsr()
    obs = pd.read_csv(obs_path, sep="\t")
    genes = pd.read_csv(genes_path, sep="\t")
    if counts.shape != (obs.shape[0], genes.shape[0]):
        raise ValueError(
            "Count matrix shape does not match obs/genes: "
            f"{counts.shape} vs {obs.shape[0]} cells and {genes.shape[0]} genes"
        )
    if "Cell.BC" not in obs.columns:
        raise ValueError("obs table must contain Cell.BC")
    if "gene_name" not in genes.columns:
        raise ValueError("genes table must contain gene_name")

    obs = obs.set_index("Cell.BC", drop=False)
    var = genes.set_index("gene_name", drop=False)
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def choose_root(adata, root_regime):
    regimes = adata.obs["fp_regime"].astype(str)
    root_mask = regimes == root_regime
    if not root_mask.any():
        root_mask = regimes == regimes.iloc[0]
    root_indices = np.flatnonzero(root_mask.to_numpy())
    coords = adata.obsm["X_pca"][root_indices, : min(10, adata.obsm["X_pca"].shape[1])]
    centroid = coords.mean(axis=0)
    distances = ((coords - centroid) ** 2).sum(axis=1)
    return int(root_indices[np.argmin(distances)])


def compute_neighbors(adata, n_neighbors, n_pcs):
    n_neighbors = min(n_neighbors, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca")
    n_components, labels = connected_components(
        adata.obsp["connectivities"], directed=False
    )
    if n_components > 1 and n_neighbors < adata.n_obs - 1:
        n_neighbors = min(max(n_neighbors * 2, 50), adata.n_obs - 1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca")
        n_components, labels = connected_components(
            adata.obsp["connectivities"], directed=False
        )
    adata.obs["neighbor_component"] = labels.astype(str)
    return n_neighbors, int(n_components)


def run_trajectory(adata, args):
    adata.layers["counts"] = adata.X.copy()
    adata.obs["fp_regime"] = pd.Categorical(
        adata.obs["fp_regime"].astype(str), categories=FP_ORDER, ordered=True
    )
    adata.obs["fp_branch"] = adata.obs["fp_regime"].astype(str).map(
        {
            "FP_Rgl1": "shared",
            "FP_NbFP": "shared",
            "FP_NbDA": "da",
            "FP_DA": "da",
            "FP_NbGLU": "glu",
            "FP_GLUFP": "glu",
        }
    )

    adata.var["mt"] = adata.var_names.str.lower().str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    n_top_genes = min(args.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)

    n_pcs = min(args.n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if n_pcs < 2:
        raise ValueError("Not enough cells/genes for PCA trajectory inference")
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    used_neighbors, n_components = compute_neighbors(adata, args.n_neighbors, n_pcs)
    sc.tl.umap(adata, min_dist=0.35, spread=1.0)
    sc.tl.diffmap(adata, n_comps=min(15, adata.n_obs - 1))
    adata.uns["iroot"] = choose_root(adata, args.root_regime)
    sc.tl.dpt(adata, n_dcs=min(10, adata.obsm["X_diffmap"].shape[1] - 1))
    adata.uns["trajectory_params"] = {
        "min_cells": args.min_cells,
        "n_top_genes": n_top_genes,
        "n_pcs": n_pcs,
        "n_neighbors": used_neighbors,
        "neighbor_components": n_components,
        "root_regime": args.root_regime,
        "root_cell": str(adata.obs_names[adata.uns["iroot"]]),
    }
    return adata


def save_tables(adata, outdir, prefix):
    obs_out = adata.obs.copy()
    obs_out["umap_1"] = adata.obsm["X_umap"][:, 0]
    obs_out["umap_2"] = adata.obsm["X_umap"][:, 1]
    obs_out["diffmap_1"] = adata.obsm["X_diffmap"][:, 0]
    obs_out["diffmap_2"] = adata.obsm["X_diffmap"][:, 1]
    obs_out.to_csv(outdir / f"{prefix}_trajectory_obs.tsv", sep="\t", index=False)

    agg = {
        "n_cells": ("Cell.BC", "size"),
        "median_pseudotime": ("dpt_pseudotime", "median"),
        "mean_pseudotime": ("dpt_pseudotime", "mean"),
        "min_pseudotime": ("dpt_pseudotime", "min"),
        "max_pseudotime": ("dpt_pseudotime", "max"),
        "median_counts": ("total_counts", "median"),
    }
    if "leaf_id" in obs_out.columns:
        agg["n_clones"] = ("leaf_id", "nunique")
    if "pattern" in obs_out.columns:
        agg["n_patterns"] = ("pattern", "nunique")
    summary = obs_out.groupby(["fp_regime", "Cell.type"], observed=True).agg(**agg)
    front_cols = ["n_cells"]
    if "n_clones" in summary.columns:
        front_cols.append("n_clones")
    if "n_patterns" in summary.columns:
        front_cols.append("n_patterns")
    summary = summary[
        front_cols + [col for col in summary.columns if col not in front_cols]
    ].reset_index()
    summary.to_csv(
        outdir / f"{prefix}_trajectory_regime_summary.tsv", sep="\t", index=False
    )

    edge_rows = []
    medians = obs_out.groupby("fp_regime", observed=True)["dpt_pseudotime"].median()
    for source, target in FP_EDGES:
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "source_median_pseudotime": medians.get(source, np.nan),
                "target_median_pseudotime": medians.get(target, np.nan),
                "delta_median_pseudotime": medians.get(target, np.nan)
                - medians.get(source, np.nan),
            }
        )
    pd.DataFrame(edge_rows).to_csv(
        outdir / f"{prefix}_trajectory_expected_edges.tsv", sep="\t", index=False
    )


def plot_categorical(adata, outdir, prefix, key):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x = adata.obsm["X_umap"][:, 0]
    y = adata.obsm["X_umap"][:, 1]
    values = adata.obs[key].astype(str)
    categories = [c for c in FP_ORDER + CELL_TYPE_ORDER if c in set(values)]
    categories += sorted(set(values) - set(categories))
    for category in categories:
        mask = values == category
        color = FP_COLORS.get(category)
        ax.scatter(
            x[mask],
            y[mask],
            s=28,
            linewidths=0.2,
            edgecolors="white",
            alpha=0.9,
            label=FP_LABELS.get(category, category),
            color=color,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(key)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.2)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_umap_{key}.png", dpi=220)
    plt.close(fig)


def plot_continuous(adata, outdir, prefix, basis, key):
    coords = adata.obsm[f"X_{basis}"]
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=adata.obs[key].astype(float),
        s=28,
        cmap="viridis",
        linewidths=0.2,
        edgecolors="white",
    )
    ax.set_xlabel(f"{basis.upper()} 1")
    ax.set_ylabel(f"{basis.upper()} 2")
    ax.set_title(key)
    fig.colorbar(scatter, ax=ax, label=key)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_{basis}_{key}.png", dpi=220)
    plt.close(fig)


def save_plots(adata, outdir, prefix):
    plot_categorical(adata, outdir, prefix, "fp_regime")
    plot_categorical(adata, outdir, prefix, "Cell.type")
    plot_continuous(adata, outdir, prefix, "umap", "dpt_pseudotime")
    plot_continuous(adata, outdir, prefix, "diffmap", "dpt_pseudotime")


def save_velocity_status(adata, outdir, prefix):
    has_spliced = "spliced" in adata.layers
    has_unspliced = "unspliced" in adata.layers
    status = {
        "velocity_computed": False,
        "has_spliced_layer": has_spliced,
        "has_unspliced_layer": has_unspliced,
        "reason": (
            "RNA velocity was not computed because the available inputs contain "
            "RNA counts/data only. Velocity requires spliced and unspliced count "
            "layers, or raw BAM/FASTQ inputs from which those layers can be generated."
        ),
    }
    (outdir / f"{prefix}_velocity_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    (outdir / f"{prefix}_velocity_status.txt").write_text(status["reason"] + "\n")


def save_summary(adata, outdir, prefix):
    params = adata.uns["trajectory_params"]
    lines = [
        "E15 rep1 FP trajectory inference summary",
        f"cells: {adata.n_obs}",
        f"highly_variable_genes_used: {adata.n_vars}",
        f"root_regime: {params['root_regime']}",
        f"root_cell: {params['root_cell']}",
        f"pca_components: {params['n_pcs']}",
        f"neighbors: {params['n_neighbors']}",
        f"neighbor_components: {params['neighbor_components']}",
        "rna_velocity: not computed; see velocity status file",
    ]
    (outdir / f"{prefix}_analysis_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sc.settings.verbosity = 2

    adata = read_inputs(outdir, args.prefix)
    adata = run_trajectory(adata, args)
    save_tables(adata, outdir, args.prefix)
    save_plots(adata, outdir, args.prefix)
    save_velocity_status(adata, outdir, args.prefix)
    save_summary(adata, outdir, args.prefix)
    adata.write_h5ad(outdir / f"{args.prefix}_trajectory.h5ad")

    print(f"Wrote trajectory outputs to {outdir}")
    print(f"Root cell: {adata.uns['trajectory_params']['root_cell']}")
    print("RNA velocity was not computed; see velocity status output.")


if __name__ == "__main__":
    main()
