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
for completeness. If the diff-test metadata records a multi-theta null regime,
calibration simulates from that H0 partition and refits simulated datasets
under the same nested H0/H1 comparison.
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
    p.add_argument(
        "--chi", required=True, help="Chi-squared TSV produced by run-diff-test."
    )
    p.add_argument(
        "--meta",
        default=None,
        help="Path to {prefix}_meta.json (default: alongside --chi).",
    )
    p.add_argument(
        "--outdir", default=None, help="Output dir (default: same dir as --chi)."
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="Output prefix (default: derived from --chi filename).",
    )

    # Calibration mode (one of)
    p.add_argument(
        "--sim_all",
        type=int,
        default=None,
        help="Pooled empirical null: number of simulations.",
    )
    p.add_argument(
        "--sim_each",
        type=int,
        default=None,
        help="Per-gene empirical null: simulations per gene.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for null simulation (default: 42).",
    )
    p.add_argument(
        "--cache",
        default=None,
        help="Path to a null-LR pickle. If it exists, load and skip "
        "simulation; otherwise simulate and write to this path.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Chunk size for the LRT call on simulated data (default: 1000). "
        "For --sim_each, total fits per chunk = batch * sim_each.",
    )
    p.add_argument(
        "--pool_top_drop",
        type=float,
        default=0.0,
        help="(--sim_all only) Drop the top fraction of genes by observed lrt "
        "from the H0 parameter pool to reduce contamination from true "
        "positives. Default 0 (no filter); typical 0.05-0.2.",
    )
    p.add_argument(
        "--gpd_tail",
        action="store_true",
        help="(--sim_all only) Fit a Generalized Pareto Distribution to the "
        "upper tail of null LRs and use it to extrapolate small p-values. "
        "Cuts required sims ~5-10x for the same effective resolution.",
    )
    p.add_argument(
        "--gpd_quantile",
        type=float,
        default=0.90,
        help="Quantile threshold above which to fit GPD (default 0.90). "
        "Lower quantile uses more excesses but assumes Pareto regime starts earlier.",
    )
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


def _h0_params_from_tsv(chi_df, h0_regimes=None, pool_top_drop=0.0):
    """Convert TSV rows -> H0 parameter arrays for null simulation.

    Legacy one-theta H0 rows become ``[log_r, log_alpha, sigma, theta0]``.
    Multi-theta H0 rows become ``[log_r, log_alpha, sigma, theta_0, ...]``
    using the supplied ``h0_regimes`` column order.

    If ``pool_top_drop > 0``, also drop the top fraction of genes by observed lrt
    to reduce contamination of the H0 pool by true positives.
    """
    theta_cols = (
        ["h0_theta"]
        if h0_regimes is None
        else [f"h0_theta_{regime}" for regime in h0_regimes]
    )
    required = ["h0_r", "h0_alpha", "h0_sigma"] + theta_cols
    missing = [col for col in required if col not in chi_df.columns]
    if missing:
        raise ValueError(f"Missing required H0 parameter columns in --chi: {missing}")

    h0_r = chi_df["h0_r"].to_numpy(dtype=np.float64)
    h0_alpha = chi_df["h0_alpha"].to_numpy(dtype=np.float64)
    h0_sigma = chi_df["h0_sigma"].to_numpy(dtype=np.float64)
    h0_theta = chi_df[theta_cols].to_numpy(dtype=np.float64)

    valid = np.isfinite(h0_r) & np.isfinite(h0_alpha) & np.isfinite(h0_sigma)
    valid &= np.isfinite(h0_theta).all(axis=1)
    valid &= (h0_r > 0) & (h0_alpha > 0)
    if not valid.all():
        logger.warning(
            f"{(~valid).sum()}/{len(chi_df)} genes have non-finite or non-positive H0 params; "
            "they will be excluded from the null-simulation parameter pool."
        )

    if pool_top_drop > 0:
        if "lrt" not in chi_df.columns:
            raise ValueError("--pool_top_drop > 0 but 'lrt' column missing from --chi.")
        lrt_vals = chi_df["lrt"].to_numpy(dtype=np.float64)
        finite_lrt = np.isfinite(lrt_vals)
        if finite_lrt.any():
            threshold = np.quantile(lrt_vals[finite_lrt], 1.0 - pool_top_drop)
            top_mask = finite_lrt & (lrt_vals >= threshold)
            n_dropped = int((valid & top_mask).sum())
            valid &= ~top_mask
            logger.info(
                f"Pool contamination filter: dropping {n_dropped} genes with lrt >= "
                f"{threshold:.3f} (top {pool_top_drop:.0%})."
            )

    log_r = np.log(np.where(valid, h0_r, 1.0))
    log_alpha = np.log(np.where(valid, h0_alpha, 1.0))

    h0 = np.concatenate(
        [
            log_r[:, None],
            log_alpha[:, None],
            h0_sigma[:, None],
            h0_theta,
        ],
        axis=1,
    )
    return h0[valid], valid


def _fit_gpd_tail(null_LRs, quantile):
    """Fit a GPD to the upper tail of null_LRs above the given quantile.

    Returns (threshold, shape, scale, n_tail) on success, else None.
    """
    from scipy.stats import genpareto

    threshold = float(np.quantile(null_LRs, quantile))
    excesses = null_LRs[null_LRs > threshold] - threshold
    n_tail = len(excesses)
    if n_tail < 50:
        logger.warning(
            f"GPD tail: only {n_tail} excesses above q={quantile}; "
            f"falling back to pure empirical p (need >=50)."
        )
        return None
    try:
        shape, _, scale = genpareto.fit(excesses, floc=0)
    except Exception as e:
        logger.warning(f"GPD fit failed ({e}); falling back to pure empirical p.")
        return None
    logger.info(
        f"GPD tail fit: q={quantile} threshold={threshold:.3f}, "
        f"n_tail={n_tail}, shape={shape:.3f}, scale={scale:.3f}"
    )
    return threshold, shape, scale, n_tail


def _empirical_or_gpd_p(obs, null_LRs, n_valid, gpd_fit):
    """Empirical p; if obs sits above the GPD threshold, use GPD survival."""
    if gpd_fit is None or obs <= gpd_fit[0]:
        return (np.sum(null_LRs >= obs) + 1) / (n_valid + 1)
    from scipy.stats import genpareto

    threshold, shape, scale, n_tail = gpd_fit
    sf = float(genpareto.sf(obs - threshold, shape, loc=0, scale=scale))
    return (n_tail / n_valid) * sf


def _build_lrt_kwargs(meta, device, dtype):
    """Reconstruct the lrt_kwargs needed to run LRT on simulated data."""
    max_iter = meta.get("iter", 10000)
    return dict(
        max_iter=max_iter,
        learning_rate=meta.get("lr", 1e-1),
        dtype=dtype,
        device=device,
        wandb_flag=None,
        window=min(meta.get("window", 200), max_iter),
        tol=meta.get("tol", 1e-4),
        approx=meta.get("approx", "softplus_MC"),
        em_iter=meta.get("em_iter", 0),
        prior=meta.get("prior", 1.0),
        init=meta.get("init", False),
        kkt=meta.get("kkt", True),
        grid=meta.get("grid", 0),
        nb=meta.get("nb", True),
        importance=meta.get("importance", 0),
        const=meta.get("const", False),
        mix=meta.get("mix", 1.0),
        grad_clip_norm=meta.get("grad_clip_norm", None),
        seed_per_gene=meta.get("seed_per_gene", True),
        root_mode=meta.get("root_mode", "stationary"),
    )


def run_calibrate():
    start_time = time.time()
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

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
    chi_df = chi_df.sort_values("ID").reset_index(drop=True)
    logger.info(f"Read {len(chi_df)} genes from {chi_path}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float32 if meta.get("dtype", "float64") == "float32" else torch.float64
    )
    logger.info(f"Using device: {device}")

    # Preprocess (we need tree topology + cells_list to simulate; library_list for sampling)
    tree_files = meta["tree"].split(",")
    gene_files = meta["expression"].split(",")
    regime_files = meta["regime"].split(",")
    null_regime_files = (
        meta["null_regime"].split(",") if meta.get("null_regime") else None
    )
    library_files = (
        meta["library"].split(",") if meta.get("library") else [None] * len(tree_files)
    )

    processed = process_data_OU(
        tree_files,
        gene_files,
        regime_files,
        library_files,
        meta["null"],
        device,
        null_regime_files=null_regime_files,
    )
    if null_regime_files is None:
        (
            tree_list,
            cells_list,
            df_list,
            diverge_list,
            share_list,
            epochs_list,
            beta_list,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            regime_list,
            library_list,
        ) = processed
        null_regimes = None
        null_beta_list = None
        null_beta_list_torch = None
    else:
        (
            tree_list,
            cells_list,
            df_list,
            diverge_list,
            share_list,
            epochs_list,
            beta_list,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            regime_list,
            library_list,
            null_regime_list,
            null_beta_list,
            null_beta_list_torch,
        ) = processed
        null_regimes = list(dict.fromkeys(x for sub in null_regime_list for x in sub))

    regimes = list(dict.fromkeys(x for sub in regime_list for x in sub))
    n_regimes = len(regimes)
    h0_n_regimes = len(null_regimes) if null_regimes is not None else None
    logger.info(f"Alternative theta regimes: {regimes}")
    if null_regimes is None:
        logger.info("Null theta regimes: one shared theta")
    else:
        logger.info(f"Null theta regimes: {null_regimes}")

    lrt_kwargs = dict(
        n_regimes=n_regimes,
        h0_n_regimes=h0_n_regimes,
        h0_beta_list_torch=null_beta_list_torch,
        diverge_list=diverge_list,
        share_list=share_list,
        epochs_list=epochs_list,
        beta_list=beta_list,
        diverge_list_torch=diverge_list_torch,
        share_list_torch=share_list_torch,
        epochs_list_torch=epochs_list_torch,
        beta_list_torch=beta_list_torch,
        library_list=library_list,
        **_build_lrt_kwargs(meta, device, dtype),
    )

    nb = meta.get("nb", True)
    root_mode = meta.get("root_mode", "stationary")
    delta_nll_obs = chi_df["lrt"].to_numpy(dtype=np.float64) / 2.0

    # === sim_all mode ===
    if args.sim_all:
        # Load any existing (possibly partial) cache for resume
        cached_LRs = np.empty(0)
        if args.cache and os.path.exists(args.cache):
            with open(args.cache, "rb") as f:
                cached_LRs = pickle.load(f)
            logger.info(f"Loaded {len(cached_LRs)} cached null LRs from {args.cache}")

        if len(cached_LRs) >= args.sim_all:
            null_LRs = cached_LRs[: args.sim_all]
            logger.info(
                f"Cache has {len(cached_LRs)} >= sim_all={args.sim_all}; skipping simulation."
            )
        else:
            ou_params, _ = _h0_params_from_tsv(
                chi_df,
                h0_regimes=null_regimes,
                pool_top_drop=args.pool_top_drop,
            )
            logger.info(
                f"Simulating {args.sim_all} null datasets from {len(ou_params)} H0 params..."
            )
            x_sim = simulate_null_all(
                tree_list,
                ou_params,
                args.sim_all,
                cells_list,
                library_list=library_list,
                nb=nb,
                root_mode=root_mode,
                diverge_list=diverge_list if null_regimes is not None else None,
                share_list=share_list if null_regimes is not None else None,
                epochs_list=epochs_list if null_regimes is not None else None,
                beta_list=null_beta_list if null_regimes is not None else None,
            )
            x_sim = [np.expand_dims(x, axis=1) for x in x_sim]

            # Resume aligned to batch boundary (drops at most batch-1 cached sims).
            done = (len(cached_LRs) // args.batch) * args.batch
            chunks = [cached_LRs[:done]] if done > 0 else []
            if done > 0:
                logger.info(
                    f"Resuming from sim {done} (cache truncated to batch={args.batch} boundary)."
                )
            for s in range(done, args.sim_all, args.batch):
                e = min(s + args.batch, args.sim_all)
                logger.info(f"  Fitting LRT on sims {s}:{e}...")
                chunk = [x[s:e] for x in x_sim]
                gn = [f"sim_all_{i}" for i in range(s, e)]
                _, h0_loss_sim, _, h1_loss_sim, _, _ = likelihood_ratio_test(
                    chunk, gene_names=gn, batch_start=s, **lrt_kwargs
                )
                chunks.append(h0_loss_sim[:, 0] - h1_loss_sim[:, 0])
                if args.cache:
                    tmp = args.cache + ".tmp"
                    with open(tmp, "wb") as f:
                        pickle.dump(np.concatenate(chunks), f)
                    os.replace(tmp, args.cache)
                    logger.info(
                        f"  Checkpointed {sum(len(c) for c in chunks)} sims to {args.cache}"
                    )
            null_LRs = np.concatenate(chunks)

        # Compute empirical p-values (drop non-finite null sims from denominator)
        finite_mask = np.isfinite(null_LRs)
        n_valid_sims = int(finite_mask.sum())
        n_total_sims = len(null_LRs)
        if n_valid_sims < n_total_sims:
            logger.warning(
                f"Excluded {n_total_sims - n_valid_sims}/{n_total_sims} null sims "
                f"with non-finite LR; using {n_valid_sims} valid sims for p-values."
            )
        else:
            logger.info(f"All {n_total_sims} null sims are finite.")
        null_LRs_finite = null_LRs[finite_mask]

        gpd_fit = (
            _fit_gpd_tail(null_LRs_finite, args.gpd_quantile) if args.gpd_tail else None
        )

        results_emp = {}
        n_tail_used = 0
        for i, row in chi_df.iterrows():
            obs_delta_nll = delta_nll_obs[i]
            if not np.isfinite(obs_delta_nll) or n_valid_sims == 0:
                p_emp = np.nan
            else:
                if gpd_fit is not None and obs_delta_nll > gpd_fit[0]:
                    n_tail_used += 1
                p_emp = _empirical_or_gpd_p(
                    obs_delta_nll, null_LRs_finite, n_valid_sims, gpd_fit
                )
            results_emp[i] = list(row.values[:-3]) + [p_emp]
        if gpd_fit is not None:
            logger.info(f"GPD tail used for {n_tail_used}/{len(chi_df)} genes.")

        out_path = os.path.join(outdir, f"{prefix}_empirical-all.tsv")
        output_results(results_emp, out_path, regimes, null_regimes)
        logger.info(f"Wrote {out_path}")

    # === sim_each mode ===
    elif args.sim_each:
        ou_params, valid = _h0_params_from_tsv(chi_df, h0_regimes=null_regimes)
        n_valid = len(ou_params)

        cached_LRs = None
        if args.cache and os.path.exists(args.cache):
            with open(args.cache, "rb") as f:
                cached_LRs = pickle.load(f)
            logger.info(
                f"Loaded cached null LRs of shape {cached_LRs.shape} from {args.cache}"
            )

        if (
            cached_LRs is not None
            and cached_LRs.ndim == 2
            and cached_LRs.shape == (n_valid, args.sim_each)
        ):
            null_LRs_each = cached_LRs
            logger.info(
                f"Cache complete ({n_valid} genes x {args.sim_each} sims); skipping simulation."
            )
        else:
            # simulate_null_each expects (batch, 1, param_dim)
            h0_params_each = ou_params[:, None, :]
            logger.info(
                f"Simulating {args.sim_each} per-gene null datasets for {n_valid} genes..."
            )
            x_sim = simulate_null_each(
                tree_list,
                h0_params_each,
                args.sim_each,
                cells_list,
                library_list=library_list,
                nb=nb,
                root_mode=root_mode,
                diverge_list=diverge_list if null_regimes is not None else None,
                share_list=share_list if null_regimes is not None else None,
                epochs_list=epochs_list if null_regimes is not None else None,
                beta_list=null_beta_list if null_regimes is not None else None,
            )

            # Resume aligned to batch boundary (drops at most batch-1 cached genes).
            chunks = []
            done = 0
            if (
                cached_LRs is not None
                and cached_LRs.ndim == 2
                and cached_LRs.shape[1] == args.sim_each
            ):
                done = (cached_LRs.shape[0] // args.batch) * args.batch
                if done > 0:
                    chunks.append(cached_LRs[:done])
                    logger.info(
                        f"Resuming from gene {done} (cache truncated to batch={args.batch} boundary)."
                    )
            for s in range(done, n_valid, args.batch):
                e = min(s + args.batch, n_valid)
                logger.info(f"  Fitting LRT on genes {s}:{e}...")
                chunk = [x[s:e] for x in x_sim]
                gn = [f"sim_each_{i}" for i in range(s, e)]
                _, h0_loss_sim, _, h1_loss_sim, _, _ = likelihood_ratio_test(
                    chunk, gene_names=gn, batch_start=s, **lrt_kwargs
                )
                chunks.append(h0_loss_sim - h1_loss_sim)
                if args.cache:
                    tmp = args.cache + ".tmp"
                    with open(tmp, "wb") as f:
                        pickle.dump(np.concatenate(chunks, axis=0), f)
                    os.replace(tmp, args.cache)
                    logger.info(
                        f"  Checkpointed {sum(c.shape[0] for c in chunks)} genes to {args.cache}"
                    )
            null_LRs_each = np.concatenate(chunks, axis=0)  # (n_valid, sim_each)

        results_emp = {}
        # Map valid index back to row index
        valid_idx = np.flatnonzero(valid)
        n_total_sims_each = null_LRs_each.shape[1]
        per_gene_valid = np.isfinite(null_LRs_each).sum(axis=1)
        n_genes_partial = int((per_gene_valid < n_total_sims_each).sum())
        if n_genes_partial > 0:
            logger.warning(
                f"{n_genes_partial}/{len(valid_idx)} genes had non-finite null sims; "
                f"per-gene valid sim counts range "
                f"[{int(per_gene_valid.min())}, {int(per_gene_valid.max())}] "
                f"of {n_total_sims_each}."
            )
        else:
            logger.info(f"All genes have all {n_total_sims_each} null sims finite.")
        for j, i in enumerate(valid_idx):
            obs_delta_nll = delta_nll_obs[i]
            null_row = null_LRs_each[j, :]
            null_row_finite = null_row[np.isfinite(null_row)]
            n_valid = len(null_row_finite)
            if not np.isfinite(obs_delta_nll) or n_valid == 0:
                p_emp = np.nan
            else:
                p_emp = (np.sum(null_row_finite >= obs_delta_nll) + 1) / (n_valid + 1)
            results_emp[int(i)] = list(chi_df.iloc[i].values[:-3]) + [p_emp]
        # Genes excluded from valid pool keep p=NaN so output_results pushes them to end
        for i in np.flatnonzero(~valid):
            results_emp[int(i)] = list(chi_df.iloc[i].values[:-3]) + [np.nan]

        out_path = os.path.join(outdir, f"{prefix}_empirical-each.tsv")
        output_results(results_emp, out_path, regimes, null_regimes)
        logger.info(f"Wrote {out_path}")

    elapsed = time.time() - start_time
    logger.info(f"Total running time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    run_calibrate()
