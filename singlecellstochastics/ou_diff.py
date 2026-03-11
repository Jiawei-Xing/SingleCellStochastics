import torch
import numpy as np
import pandas as pd
import argparse
import wandb
import os
import pickle
import time
import gc
from .preprocess import process_data_OU
from .lrt import likelihood_ratio_test
from .simulate import simulate_null_all, simulate_null_each
from .output import save_result, output_results


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
    parser.add_argument("--resume", action="store_true", help="Resume from log file")
    parser.add_argument("--wandb", type=str, default=None, help="Wandb run name")
    parser.add_argument("--selection", type=str, default=None, help="Selection test TSV to reuse OU H1 as diff test H0 (skip H0 optimization)")

    # Empirical null / importance sampling
    parser.add_argument("--sim_all", type=int, default=None, help="Simulations for shared empirical null")
    parser.add_argument("--sim_each", type=int, default=None, help="Simulations for per-gene empirical null")
    parser.add_argument("--importance", type=int, default=0, help="Number of importance samples (default: 0)")
    parser.add_argument("--mix", type=float, default=1, help="Weight for q(z) in importance sampling mixture (default: 1)")

    return parser.parse_args()


def run_diff_test():
    args = parse_args()

    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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

    library_list_tensor = [
        torch.tensor(lib.values.squeeze(), dtype=torch.float32, device=device) for lib in library_list
    ]

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
        importance=args.importance, const=args.const, mix=args.mix
    )

    # Resume from log file
    results = {}
    output_dir = args.outdir
    prefix = args.prefix
    log_file = os.path.join(output_dir, f"{prefix}.log")
    if args.resume and os.path.exists(log_file):
        with open(log_file, 'r') as f_log:
            for line in f_log:
                parts = line.strip().split("\t")
                gene_idx = int(parts[0])
                gene_name = str(parts[1])
                params = list(map(float, parts[2:]))
                results[gene_idx] = [gene_idx, gene_name] + params
    elif args.resume:
        print(f"Log file {log_file} not found. Starting from scratch.")

    # Load selection test results for H0 reuse
    sel_df = None
    sel_q_params = None
    if args.selection:
        sel_df = pd.read_csv(args.selection, sep="\t")
        sel_df = sel_df.set_index("gene")
        print(f"Loaded selection results from {args.selection} ({len(sel_df)} genes)")
        print("Reusing OU H1 from selection as diff test H0 (skipping H0 optimization)")
        # Load OU q params from selection for H0 init
        sel_dir = os.path.dirname(args.selection)
        sel_q_params = []
        for i in range(len(tree_files)):
            q_path = os.path.join(sel_dir, f"ou_q_params_{i}.tsv")
            if os.path.exists(q_path):
                sel_q_params.append(pd.read_csv(q_path, sep="\t", index_col=0))
            else:
                sel_q_params = None
                print(f"Warning: OU q params file {q_path} not found, using fresh init")
                break

    # Process gene batches
    gene_idx_start = max(results.keys()) + 1 if results else 0
    h0_q_params = []
    h1_q_params = []
    results_empirical_each = {}
    ou_params_all = []
    lr_all = []

    for batch_start in range(gene_idx_start, len(df_list[0].columns), batch_size):
        batch_genes = df_list[0].columns[batch_start : batch_start + batch_size]

        # Gene names (with optional annotation)
        if args.annot:
            df_annot = pd.read_csv(args.annot, sep="\t", index_col=0, header=None)
            batch_gene_names = df_annot.loc[batch_genes, 0].tolist()
        else:
            batch_gene_names = batch_genes
        print(f"\ngene batch {batch_start}-{batch_start+len(batch_genes)-1}: {list(batch_gene_names)}", flush=True)

        # Expression data: list of (batch_size, 1, n_cells)
        x_original = [np.expand_dims(df[batch_genes].values.T, axis=1) for df in df_list]

        # Build h0_override from selection TSV if provided
        h0_override = None
        if sel_df is not None:
            batch_sel = sel_df.loc[batch_gene_names]
            # OU params: alpha, sigma, theta0, ... (n_regimes+2 cols)
            ou_param_cols = ["h1_alpha", "h1_sigma"] + [f"h1_theta{i}" for i in range(n_regimes)]
            ou_params = batch_sel[ou_param_cols].values  # (batch, n_regimes+2)
            log_r = np.log(batch_sel["h1_r"].values)  # (batch,)
            loss = batch_sel["h1"].values  # (batch,)
            # Add N_sim=1 dimension
            h0_override = {
                'ou_params': ou_params[:, np.newaxis, :],  # (batch, 1, n_regimes+2)
                'log_r': log_r[:, np.newaxis],  # (batch, 1)
                'loss': loss[:, np.newaxis],  # (batch, 1)
            }
            # Add saved q params for H0 init if available
            if sel_q_params is not None:
                h0_override['q_params'] = [
                    sel_q_params[i].loc[batch_gene_names].values[:, np.newaxis, :]
                    for i in range(len(sel_q_params))
                ]  # list of (batch, 1, 2*n_cells)

        # LRT (diff test)
        h0_params, h0_loss, h1_params, h1_loss, h0_q, h1_q = likelihood_ratio_test(
            x_original, gene_names=batch_gene_names, batch_start=batch_start,
            h0_override=h0_override, **lrt_kwargs
        )

        # Save results
        results = save_result(batch_start, batch_size, batch_genes,
            h0_params, h1_params, h0_loss, h1_loss, results, args.approx)

        # Accumulate variational parameters
        if sel_df is None:
            if h0_q_params:
                h0_q_params = [np.concatenate([h0_q_params[i], h0_q[i]], axis=0) for i in len(h0_q)]
            else:
                h0_q_params = [n for n in h0_q]
        if h1_q_params:
            h1_q_params = [np.concatenate([h1_q_params[i], h1_q[i]], axis=0) for i in len(h1_q)]
        else:
            h1_q_params = [n for n in h1_q]

        # Collect OU params and LR stats
        ou_params_all.append(h0_params[:, 0, :])
        lr = h0_loss - h1_loss
        lr_all.append(lr[:, 0])

        # Per-gene empirical null
        if args.sim_each:
            x_sim = simulate_null_each(tree_list, h0_params, args.sim_each, cells_list)
            _, h0_loss_sim, _, h1_loss_sim, _, _ = likelihood_ratio_test(
                x_sim, gene_names=batch_gene_names, batch_start=batch_start, **lrt_kwargs
            )
            null_LRs = h0_loss_sim - h1_loss_sim
            for i in range(batch_size):
                p_empirical = (sum(null_LRs[i, :] >= lr[i, 0]) + 1) / (len(null_LRs[i, :]) + 1)
                results_empirical_each[batch_start + i] = results[batch_start + i][:-1] + [p_empirical]

        # Log file
        with open(log_file, 'a') as f_log:
            for i in range(batch_size):
                if batch_start + i not in results:
                    continue
                f_log.write("\t".join(list(map(str, results[batch_start + i]))) + "\n")

        # Free memory
        del x_original, h0_params, h1_params, h0_loss, h1_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(0.5)

    # === Output results ===

    # Chi-squared results
    output_results(results, os.path.join(output_dir, f"{prefix}_chi-squared.tsv"), regimes)

    # Variational parameters (skip h0_q if reusing from selection)
    if sel_df is None:
        for i in range(len(h0_q_params)):
            df = pd.DataFrame(h0_q_params[i][:,0,:], index=df_list[i].columns, columns=cells_list[i]*2)
            df.to_csv(os.path.join(output_dir, f"{prefix}_h0_q-mean-std_{i}.tsv"), sep="\t")
    for i in range(len(h1_q_params)):
        df = pd.DataFrame(h1_q_params[i][:,0,:], index=df_list[i].columns, columns=cells_list[i]*2)
        df.to_csv(os.path.join(output_dir, f"{prefix}_h1_q-mean-std_{i}.tsv"), sep="\t")

    # Per-gene empirical null results
    if args.sim_each:
        output_results(results_empirical_each, os.path.join(output_dir, f"{prefix}_empirical-each.tsv"), regimes)

    # Shared empirical null
    if args.sim_all:
        print(f"\nUsing empirical null distribution for all genes ({args.sim_all} simulations)")
        null_LRs = None
        os.makedirs(output_dir, exist_ok=True)
        sim_all_cache_file = os.path.join(output_dir, f"sim_all_null_LRs_{args.sim_all}.pkl")

        # Try cached
        if os.path.exists(sim_all_cache_file):
            try:
                with open(sim_all_cache_file, 'rb') as f:
                    null_LRs = pickle.load(f)
                print(f"\nLoaded cached null LRs from {sim_all_cache_file}")
            except Exception as e:
                print(f"\nFailed to load cached parameters: {e}")

        # Simulate if not cached
        if null_LRs is None:
            ou_params_all = np.concatenate(ou_params_all, axis=0)
            x_sim = simulate_null_all(tree_list, ou_params_all, args.sim_all, cells_list)
            x_sim = [np.expand_dims(x, axis=1) for x in x_sim]
            gene_names = ["sim_all"] * args.sim_all
            _, h0_loss_sim, _, h1_loss_sim, _, _ = likelihood_ratio_test(
                x_sim, gene_names=gene_names, batch_start=batch_start, **lrt_kwargs
            )
            null_LRs = h0_loss_sim[:, 0] - h1_loss_sim[:, 0]
            with open(sim_all_cache_file, 'wb') as f:
                pickle.dump(null_LRs, f)

        # Compute empirical p-values
        lr_all = np.concatenate(lr_all, axis=0)
        results_empirical_all = {}
        for i in range(lr_all.shape[0]):
            p_empirical_all = (sum(null_LRs >= lr_all[i]) + 1) / (len(null_LRs) + 1)
            results_empirical_all[i] = results[i][:-1] + [p_empirical_all]
        output_results(results_empirical_all, os.path.join(output_dir, f"{prefix}_empirical-all.tsv"), regimes)


# Backward compatibility
run_ou_poisson = run_diff_test


if __name__ == "__main__":
    run_diff_test()
