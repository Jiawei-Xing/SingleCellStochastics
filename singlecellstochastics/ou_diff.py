import torch
import numpy as np
import pandas as pd
import argparse
import wandb
import os
import json
import time
import gc
import logging
import shutil
from .preprocess import process_data_OU
from .lrt import likelihood_ratio_test
from .output import save_result, output_results

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differential expression test: regime-specific OU LRT"
    )
    # Required inputs
    parser.add_argument("--tree", type=str, required=True, help="Newick tree file (comma-separated if multiple clones)")
    parser.add_argument("--expr", type=str, required=True, help="Cell by gene count matrix (comma-separated if multiple clones)")
    parser.add_argument("--regime", type=str, required=True, help="Regime file (comma-separated if multiple clones)")
    parser.add_argument("--null", type=str, required=True, help="Regime for null hypothesis")

    # Output
    parser.add_argument("--outdir", type=str, default="./", help="Output directory (default: ./)")
    parser.add_argument("--prefix", type=str, default="result", help="Prefix for output files (default: result)")

    # Optimization
    parser.add_argument("--batch", type=int, default=1000, help="Batch size for gene processing (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for Adam optimizer (default: 1e-1)")
    parser.add_argument("--iter", type=int, default=10000, help="Max optimization iterations (default: 10000)")
    parser.add_argument("--window", type=int, default=200, help="Convergence check window (default: 200)")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"], help="Torch dtype (default: float64)")

    # Model
    parser.add_argument("--approx", type=str, default="softplus_MC", help="Likelihood approximation method (default: softplus_MC)")
    parser.add_argument("--no_nb", action="store_false", dest="nb", help="Use Poisson instead of NB (default: NB)")
    parser.add_argument("--no_kkt", action="store_false", dest="kkt", help="Disable KKT constraint (default: use KKT)")
    parser.add_argument("--prior", type=float, default=1.0, help="L2 prior on log alpha (default: 1.0)")
    parser.add_argument("--pseudo", type=float, default=0, help="Pseudo count for initial mean (default: 0)")
    parser.add_argument("--const", action="store_true", help="Include constant terms in likelihood")

    # Optional features
    parser.add_argument("--annot", type=str, default=None, help="Gene annotation file")
    parser.add_argument("--library", type=str, default=None, help="Library size file per cell")
    parser.add_argument("--init", action="store_true", help="Initialize with OU optimization")
    parser.add_argument("--em_iter", type=int, default=0, help="Number of EM iterations (default: 0)")
    parser.add_argument("--grid", type=int, default=0, help="Grid search range for alpha (default: 0)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--keep_ckpt", action="store_true", help="Keep checkpoint files after combining results")
    parser.add_argument("--log", type=str, default=None, help="Path to log file")
    parser.add_argument("--wandb", type=str, default=None, help="Wandb run name")
    #parser.add_argument("--selection", type=str, default=None, help="Selection test TSV to reuse OU H1 as diff test H0 (skip H0 optimization)")

    # Importance sampling (empirical-null calibration is a separate CLI: run-calibrate)
    parser.add_argument("--importance", type=int, default=0, help="Number of importance samples (default: 0)")
    parser.add_argument("--mix", type=float, default=1, help="Weight for q(z) in importance sampling mixture (default: 1)")

    # Numerical safety knobs (off by default for backward compat)
    parser.add_argument("--clamp_log_r_before_forward", action="store_true",
                        help="Clamp log_r BEFORE the forward pass (default: post-step). Avoids leaked out-of-range values.")
    parser.add_argument("--grad_clip_norm", type=float, default=None,
                        help="Gradient norm clip per parameter tensor. Bounds bad MC steps. e.g. 10.")
    parser.add_argument("--log_alpha_clamp", type=str, default=None,
                        help="Comma-separated 'lo,hi' for log_alpha clamp post-step (e.g. -3,5). Prevents alpha→0/∞.")
    parser.add_argument("--log_r_clamp", type=str, default=None,
                        help="Comma-separated 'lo,hi' for log_r clamp (e.g. -5,8). Default = -5,5 (legacy). "
                             "Raise upper bound for genes that need Poisson-like dispersion (r >> 1).")
    parser.add_argument("--seed_per_gene", action="store_true",
                        help="Re-seed RNG per-gene-batch from a hash of gene_names. Removes batch-order dependence.")
    parser.add_argument("--s_init_floor", type=float, default=None,
                        help="Cap initial q-std at this value (e.g. 1.0). Reduces MC variance for high-count genes.")
    parser.add_argument("--disable_param_sanitize", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--enable_grad_sanitize", action="store_true",
                        help="Opt in to replacing non-finite gradient entries with 0 before Adam. Default: disabled.")
    parser.add_argument("--disable_grad_sanitize", action="store_true",
                        help=argparse.SUPPRESS)

    return parser.parse_args()


def run_diff_test():
    start_time = time.time()
    args = parse_args()

    # Setup logging
    handlers = [logging.StreamHandler()]
    if args.log:
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if args.wandb:
        wandb.login()
        wandb.init(project="SingleCellStochastics", name=args.wandb)

    # Parse comma-separated file lists
    tree_files = args.tree.split(",")
    gene_files = args.expr.split(",")
    regime_files = args.regime.split(",")
    library_files = args.library.split(",") if args.library else [None] * len(tree_files)
    window = min(args.window, args.iter)

    # Preprocess data
    (
        tree_list, cells_list, df_list,
        diverge_list, share_list, epochs_list, beta_list,
        diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch,
        regime_list, library_list
    ) = process_data_OU(tree_files, gene_files, regime_files, library_files, args.null, device)

    regimes = list(dict.fromkeys(x for sub in regime_list for x in sub))
    n_regimes = len(regimes)
    batch_size = min(args.batch, len(df_list[0].columns))

    # Parse log_alpha_clamp "lo,hi"
    log_alpha_clamp = None
    if args.log_alpha_clamp:
        lo, hi = args.log_alpha_clamp.split(",")
        log_alpha_clamp = (float(lo), float(hi))

    log_r_clamp = None
    if args.log_r_clamp:
        lo, hi = args.log_r_clamp.split(",")
        log_r_clamp = (float(lo), float(hi))

    # Common kwargs for likelihood_ratio_test
    lrt_kwargs = dict(
        n_regimes=n_regimes,
        diverge_list=diverge_list, share_list=share_list,
        epochs_list=epochs_list, beta_list=beta_list,
        diverge_list_torch=diverge_list_torch, share_list_torch=share_list_torch,
        epochs_list_torch=epochs_list_torch, beta_list_torch=beta_list_torch,
        max_iter=args.iter, learning_rate=args.lr,
        dtype=dtype, device=device, wandb_flag=args.wandb,
        window=window, tol=args.tol, approx=args.approx,
        em_iter=args.em_iter, pseudo=args.pseudo,
        prior=args.prior, init=args.init, kkt=args.kkt,
        grid=args.grid, nb=args.nb, library_list=library_list,
        importance=args.importance, const=args.const, mix=args.mix,
        clamp_log_r_before_forward=args.clamp_log_r_before_forward,
        grad_clip_norm=args.grad_clip_norm,
        log_alpha_clamp=log_alpha_clamp,
        log_r_clamp=log_r_clamp,
        seed_per_gene=args.seed_per_gene,
        s_init_floor=args.s_init_floor,
        disable_param_sanitize=args.disable_param_sanitize,
        disable_grad_sanitize=(not args.enable_grad_sanitize) or args.disable_grad_sanitize,
    )

    # Load selection test results for H0 reuse
    sel_df = None

    output_dir = args.outdir
    prefix = args.prefix
    n_genes = len(df_list[0].columns)
    n_batches = (n_genes + batch_size - 1) // batch_size

    # Checkpoint directory for per-batch results + q params
    ckpt_dir = os.path.join(output_dir, f"{prefix}.ckpt")
    if args.resume and os.path.isdir(ckpt_dir):
        logger.info(f"Found checkpoint directory: {ckpt_dir}")
    elif args.resume:
        logger.info(f"No checkpoint directory found, starting from scratch")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Process gene batches
    logger.info(f"Running diff test: {n_genes} genes, {n_batches} batches of {batch_size}")
    for batch_start in range(0, n_genes, batch_size):
        batch_idx = batch_start // batch_size
        batch_genes = df_list[0].columns[batch_start : batch_start + batch_size]

        # Check if batch already completed
        ckpt_file = os.path.join(ckpt_dir, f"batch_{batch_idx}.pt")
        if args.resume and os.path.exists(ckpt_file):
            logger.info(f"Batch {batch_idx+1}/{n_batches} already done, skipping")
            continue

        # Gene names (with optional annotation)
        if args.annot:
            df_annot = pd.read_csv(args.annot, sep="\t", index_col=0, header=None)
            batch_gene_names = df_annot.loc[batch_genes, 0].tolist()
        else:
            batch_gene_names = batch_genes
        logger.info(f"Batch {batch_idx+1}/{n_batches}: genes {batch_start}-{batch_start+len(batch_genes)-1}")

        # Expression data: list of (batch_size, 1, n_cells)
        x_original = [np.expand_dims(df[batch_genes].values.T, axis=1) for df in df_list]

        # LRT (diff test)
        h0_params, h0_loss, h1_params, h1_loss, h0_q, h1_q = likelihood_ratio_test(
            x_original, gene_names=batch_gene_names, batch_start=batch_start, **lrt_kwargs
        )

        # Build results dict for this batch
        batch_results = save_result(batch_start, batch_size, batch_genes,
            h0_params, h1_params, h0_loss, h1_loss, {}, args.approx)

        # Save checkpoint
        ckpt_data = {
            'batch_start': batch_start,
            'batch_genes': list(batch_genes),
            'results': batch_results,
            'h0_q': h0_q,  # list of numpy arrays
            'h1_q': h1_q,
            'h0_params': h0_params,
            'h0_loss': h0_loss,
            'h1_loss': h1_loss,
        }
        torch.save(ckpt_data, ckpt_file)
        logger.info(f"Batch {batch_idx+1}/{n_batches} done ({min(batch_start+batch_size, n_genes)}/{n_genes} genes)")

        # Free memory
        del x_original, h0_params, h1_params, h0_loss, h1_loss

    # === Load all checkpoints and combine ===
    results = {}
    h0_q_params = []
    h1_q_params = []

    for batch_idx in range(n_batches):
        ckpt_file = os.path.join(ckpt_dir, f"batch_{batch_idx}.pt")
        ckpt_data = torch.load(ckpt_file, weights_only=False)

        results.update(ckpt_data['results'])

        # Accumulate q params
        if sel_df is None:
            if h0_q_params:
                h0_q_params = [np.concatenate([h0_q_params[i], ckpt_data['h0_q'][i]], axis=0) for i in range(len(ckpt_data['h0_q']))]
            else:
                h0_q_params = [n for n in ckpt_data['h0_q']]
        if h1_q_params:
            h1_q_params = [np.concatenate([h1_q_params[i], ckpt_data['h1_q'][i]], axis=0) for i in range(len(ckpt_data['h1_q']))]
        else:
            h1_q_params = [n for n in ckpt_data['h1_q']]

    # === Output results ===

    # Chi-squared results
    output_results(results, os.path.join(output_dir, f"{prefix}_chi-squared.tsv"), regimes)

    # Sidecar meta.json — everything run-calibrate needs to reconstruct lrt_kwargs
    meta = {
        "tree": args.tree, "expr": args.expr, "regime": args.regime,
        "library": args.library, "null": args.null,
        "iter": args.iter, "lr": args.lr, "window": args.window, "tol": args.tol,
        "dtype": args.dtype, "approx": args.approx, "nb": args.nb, "kkt": args.kkt,
        "prior": args.prior, "pseudo": args.pseudo, "const": args.const,
        "init": args.init, "em_iter": args.em_iter, "grid": args.grid,
        "importance": args.importance, "mix": args.mix,
        "clamp_log_r_before_forward": args.clamp_log_r_before_forward,
        "grad_clip_norm": args.grad_clip_norm,
        "log_alpha_clamp": args.log_alpha_clamp,
        "log_r_clamp": args.log_r_clamp,
        "seed_per_gene": args.seed_per_gene,
        "s_init_floor": args.s_init_floor,
        "disable_param_sanitize": args.disable_param_sanitize,
        "enable_grad_sanitize": args.enable_grad_sanitize,
    }
    with open(os.path.join(output_dir, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Variational parameters (skip h0_q if reusing from selection)
    if sel_df is None:
        for i in range(len(h0_q_params)):
            df = pd.DataFrame(h0_q_params[i][:,0,:], index=df_list[i].columns, columns=cells_list[i]*2)
            df.to_csv(os.path.join(output_dir, f"{prefix}_h0_q-mean-std_{i}.tsv"), sep="\t")
    for i in range(len(h1_q_params)):
        df = pd.DataFrame(h1_q_params[i][:,0,:], index=df_list[i].columns, columns=cells_list[i]*2)
        df.to_csv(os.path.join(output_dir, f"{prefix}_h1_q-mean-std_{i}.tsv"), sep="\t")

    # Clean up checkpoint directory
    if not args.keep_ckpt:
        shutil.rmtree(ckpt_dir)
    elapsed = time.time() - start_time
    logger.info(f"Results saved to {output_dir}/{prefix}_chi-squared.tsv")
    logger.info(f"For empirical-null calibration: run-calibrate --chi {output_dir}/{prefix}_chi-squared.tsv --sim_all <N>")
    logger.info(f"Total running time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


# Backward compatibility
run_ou_poisson = run_diff_test


if __name__ == "__main__":
    run_diff_test()
