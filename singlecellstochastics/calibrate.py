"""Empirical-null calibration for the diff test.

Consumes the chi-squared TSV from `run-diff-test` plus a sidecar meta.json
written alongside it, and produces empirical p-values without re-running the
per-gene H0/H1 optimization. Two modes:

  --sim_all N   pooled null: sample H0 OU params (with replacement) from the
                fitted genes, simulate N null datasets, fit LRT, compare each
                gene's observed LR to the resulting null LR distribution.

  --sim_each N  per-gene null: simulate N null datasets per gene from that
                gene's own H0 fit and compare its observed LR.

The pooled null is the cheap default. The per-gene null is N_gene x more
expensive and is rarely worth it once H0 params are reasonable; provided here
for completeness.
"""

import argparse
import json
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from .preprocess import process_data_OU
from .lrt import likelihood_ratio_test
from .simulate import simulate_null_all, simulate_null_each
from .output import output_results

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Empirical-null calibration for the diff test."
    )
    p.add_argument("--chi", required=True,
                   help="Chi-squared TSV produced by run-diff-test.")
    p.add_argument("--meta", default=None,
                   help="Path to {prefix}_meta.json (default: alongside --chi).")
    p.add_argument("--outdir", default=None,
                   help="Output dir (default: same dir as --chi).")
    p.add_argument("--prefix", default=None,
                   help="Output prefix (default: derived from --chi filename).")

    # Calibration mode (one of)
    p.add_argument("--sim_all", type=int, default=None,
                   help="Pooled empirical null: number of simulations.")
    p.add_argument("--sim_each", type=int, default=None,
                   help="Per-gene empirical null: simulations per gene.")

    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for null simulation (default: 42).")
    p.add_argument("--cache", action="store_true",
                   help="Reuse cached sim_all_null_LRs_{N}.pkl if present.")
    return p.parse_args()


def _load_meta(chi_path, meta_arg):
    if meta_arg is not None:
        meta_path = meta_arg
    else:
        # Replace trailing "_chi-squared.tsv" or ".tsv" with "_meta.json"
        base = chi_path
        for suffix in ("_chi-squared.tsv", ".tsv"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        meta_path = base + "_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found at {meta_path}. "
            "Either pass --meta or rerun the diff test so it writes the sidecar."
        )
    with open(meta_path) as f:
        return json.load(f), meta_path


def _h0_params_from_tsv(chi_df):
    """Convert TSV rows -> (n_genes, 4) array in [log_r, log_alpha, sigma, theta0] form."""
    required = ["h0_r", "h0_alpha", "h0_sigma", "h0_theta"]
    missing = [col for col in required if col not in chi_df.columns]
    if missing:
        raise ValueError(f"Missing required H0 parameter columns in --chi: {missing}")

    h0_r = chi_df["h0_r"].to_numpy(dtype=np.float64)
    h0_alpha = chi_df["h0_alpha"].to_numpy(dtype=np.float64)
    h0_sigma = chi_df["h0_sigma"].to_numpy(dtype=np.float64)
    h0_theta = chi_df["h0_theta"].to_numpy(dtype=np.float64)

    valid = np.isfinite(h0_r) & np.isfinite(h0_alpha) & np.isfinite(h0_sigma) & np.isfinite(h0_theta)
    valid &= (h0_r > 0) & (h0_alpha > 0)
    if not valid.all():
        logger.warning(
            f"{(~valid).sum()}/{len(chi_df)} genes have non-finite or non-positive H0 params; "
            "they will be excluded from the null-simulation parameter pool."
        )

    log_r = np.log(np.where(valid, h0_r, 1.0))
    log_alpha = np.log(np.where(valid, h0_alpha, 1.0))

    h0 = np.stack([log_r, log_alpha, h0_sigma, h0_theta], axis=1)
    return h0[valid], valid


def _normalize_result_columns(chi_df):
    """Support current result files and older files that used LR for delta NLL."""
    if "delta_nll" not in chi_df.columns:
        if "LR" not in chi_df.columns:
            raise ValueError("Result table must contain either 'delta_nll' or legacy 'LR'.")
        chi_df = chi_df.rename(columns={"LR": "delta_nll"})

    if "lrt" not in chi_df.columns:
        insert_at = chi_df.columns.get_loc("delta_nll") + 1
        chi_df.insert(insert_at, "lrt", 2 * chi_df["delta_nll"].to_numpy(dtype=np.float64))

    return chi_df


def _build_lrt_kwargs(meta, device, dtype):
    """Reconstruct the lrt_kwargs needed to run LRT on simulated data."""
    return dict(
        max_iter=meta["iter"], learning_rate=meta["lr"],
        dtype=dtype, device=device, wandb_flag=None,
        window=min(meta["window"], meta["iter"]), tol=meta["tol"],
        approx=meta["approx"],
        em_iter=meta.get("em_iter", 0),
        prior=meta.get("prior", 1.0), init=meta.get("init", False),
        kkt=meta.get("kkt", True), grid=meta.get("grid", 0),
        nb=meta.get("nb", True),
        importance=meta.get("importance", 0), const=meta.get("const", False),
        mix=meta.get("mix", 1.0),
        grad_clip_norm=meta.get("grad_clip_norm", None),
        seed_per_gene=meta.get("seed_per_gene", False),
    )


def run_calibrate():
    start_time = time.time()
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if (args.sim_all is None) == (args.sim_each is None):
        raise SystemExit("Specify exactly one of --sim_all or --sim_each.")

    chi_path = args.chi
    meta, meta_path = _load_meta(chi_path, args.meta)
    logger.info(f"Loaded meta from {meta_path}")

    outdir = args.outdir or os.path.dirname(chi_path) or "."
    if args.prefix is not None:
        prefix = args.prefix
    else:
        base = os.path.basename(chi_path)
        for suffix in ("_chi-squared.tsv", ".tsv"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        prefix = base
    os.makedirs(outdir, exist_ok=True)

    # Read chi-squared output
    chi_df = pd.read_csv(chi_path, sep="\t")
    chi_df = _normalize_result_columns(chi_df)
    chi_df = chi_df.sort_values("ID").reset_index(drop=True)
    logger.info(f"Read {len(chi_df)} genes from {chi_path}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if meta.get("dtype", "float64") == "float32" else torch.float64
    logger.info(f"Using device: {device}")

    # Preprocess (we need tree topology + cells_list to simulate; library_list for sampling)
    tree_files = meta["tree"].split(",")
    gene_files = meta["expression"].split(",")
    regime_files = meta["regime"].split(",")
    library_files = meta["library"].split(",") if meta.get("library") else [None] * len(tree_files)

    (
        tree_list, cells_list, df_list,
        diverge_list, share_list, epochs_list, beta_list,
        diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch,
        regime_list, library_list
    ) = process_data_OU(tree_files, gene_files, regime_files, library_files, meta["null"], device)

    regimes = list(dict.fromkeys(x for sub in regime_list for x in sub))
    n_regimes = len(regimes)

    lrt_kwargs = dict(
        n_regimes=n_regimes,
        diverge_list=diverge_list, share_list=share_list,
        epochs_list=epochs_list, beta_list=beta_list,
        diverge_list_torch=diverge_list_torch, share_list_torch=share_list_torch,
        epochs_list_torch=epochs_list_torch, beta_list_torch=beta_list_torch,
        library_list=library_list,
        **_build_lrt_kwargs(meta, device, dtype),
    )

    nb = meta.get("nb", True)
    delta_nll_obs = chi_df["delta_nll"].to_numpy(dtype=np.float64)

    # === sim_all mode ===
    if args.sim_all:
        cache_path = os.path.join(outdir, f"sim_all_null_LRs_{args.sim_all}.pkl")
        null_LRs = None
        if args.cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    null_LRs = pickle.load(f)
                logger.info(f"Loaded cached null LRs from {cache_path}")
            except Exception as e:
                logger.info(f"Cache load failed ({e}); regenerating.")
                null_LRs = None

        if null_LRs is None:
            ou_params, _ = _h0_params_from_tsv(chi_df)
            logger.info(f"Simulating {args.sim_all} null datasets from {len(ou_params)} H0 params...")
            x_sim = simulate_null_all(
                tree_list, ou_params, args.sim_all, cells_list,
                library_list=library_list, nb=nb,
            )
            x_sim = [np.expand_dims(x, axis=1) for x in x_sim]
            gene_names = ["sim_all"] * args.sim_all
            _, h0_loss_sim, _, h1_loss_sim, _, _ = likelihood_ratio_test(
                x_sim, gene_names=gene_names, batch_start=0, **lrt_kwargs
            )
            null_LRs = h0_loss_sim[:, 0] - h1_loss_sim[:, 0]
            with open(cache_path, "wb") as f:
                pickle.dump(null_LRs, f)
            logger.info(f"Cached null LRs to {cache_path}")

        # Compute empirical p-values
        results_emp = {}
        for i, row in chi_df.iterrows():
            obs_delta_nll = delta_nll_obs[i]
            if not np.isfinite(obs_delta_nll):
                p_emp = np.nan
            else:
                p_emp = (np.sum(null_LRs >= obs_delta_nll) + 1) / (len(null_LRs) + 1)
            results_emp[i] = list(row.values[: -3]) + [p_emp]

        out_path = os.path.join(outdir, f"{prefix}_empirical-all.tsv")
        output_results(results_emp, out_path, regimes)
        logger.info(f"Wrote {out_path}")

    # === sim_each mode ===
    elif args.sim_each:
        ou_params, valid = _h0_params_from_tsv(chi_df)
        # simulate_null_each expects (batch, 1, param_dim)
        h0_params_each = ou_params[:, None, :]
        logger.info(f"Simulating {args.sim_each} per-gene null datasets for {len(ou_params)} genes...")
        x_sim = simulate_null_each(
            tree_list, h0_params_each, args.sim_each, cells_list,
            library_list=library_list, nb=nb,
        )
        gene_names_each = [f"sim_each_{i}" for i in range(len(ou_params))]
        _, h0_loss_sim, _, h1_loss_sim, _, _ = likelihood_ratio_test(
            x_sim, gene_names=gene_names_each, batch_start=0, **lrt_kwargs
        )
        null_LRs_each = h0_loss_sim - h1_loss_sim  # (n_valid_genes, sim_each)

        results_emp = {}
        # Map valid index back to row index
        valid_idx = np.flatnonzero(valid)
        for j, i in enumerate(valid_idx):
            obs_delta_nll = delta_nll_obs[i]
            if not np.isfinite(obs_delta_nll):
                p_emp = np.nan
            else:
                null_row = null_LRs_each[j, :]
                p_emp = (np.sum(null_row >= obs_delta_nll) + 1) / (len(null_row) + 1)
            results_emp[int(i)] = list(chi_df.iloc[i].values[: -3]) + [p_emp]
        # Genes excluded from valid pool keep p=NaN so output_results pushes them to end
        for i in np.flatnonzero(~valid):
            results_emp[int(i)] = list(chi_df.iloc[i].values[: -3]) + [np.nan]

        out_path = os.path.join(outdir, f"{prefix}_empirical-each.tsv")
        output_results(results_emp, out_path, regimes)
        logger.info(f"Wrote {out_path}")

    elapsed = time.time() - start_time
    logger.info(f"Total running time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    run_calibrate()
