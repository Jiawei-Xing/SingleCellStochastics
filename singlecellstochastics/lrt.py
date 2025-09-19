
import sys
import io
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import torch
from Bio import Phylo

from .tree_utils import read_tree, assign_nodes_to_regimes_from_file, assign_nodes_to_null_regimes, add_read_counts_to_tips
from .input_output import load_read_count_tsv
from .optimization import adam_optimize_ou_parameters
from .stat_utils import calculate_lrt_and_pvalue


def process_gene(
    gene: str, 
    read_count_data: dict, 
    null_tree: Phylo.BaseTree.Tree, 
    alt_tree: Phylo.BaseTree.Tree, 
    root_expression: torch.Tensor,
    null_regime: str,
    poisson_logl_mode: str,
    output_dir: str
) -> None:
    """
    Process a single gene: fit null and alt OU models, compute LRT, and capture all prints to a log file.
    
    Args:
        gene: Gene name (str).
        read_count_data: Dictionary mapping gene names to tip read counts (dict).
        null_tree: Biopython `Tree` object for the null hypothesis (all nodes in one regime).
        alt_tree: Biopython `Tree` object for the alternative hypothesis (nodes in multiple regimes).
        root_expression: Expression value at the root node (torch.Tensor).
        null_regime: Regime label for all nodes under the null hypothesis (str).
        poisson_logl_mode: Mode for Poisson sampling, either "deterministic", "stochastic", or "variational".
        output_dir: Directory to save log files (str).
    
    Returns: 
        None
    """
    log_file = f"{output_dir}/{gene}.log"
    
    # Capture stdout
    buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    
    try:
        print(f"\nProcessing gene {gene}")
        
        # Make copies of trees to avoid interference between threads
        null_tree_copy = copy.deepcopy(null_tree)
        alt_tree_copy = copy.deepcopy(alt_tree)
        
        add_read_counts_to_tips(null_tree_copy, read_count_data[gene])
        add_read_counts_to_tips(alt_tree_copy, read_count_data[gene])
        
        alpha_init = torch.tensor(1.0, dtype=torch.float32)
        sigma2_init = torch.tensor(1.0, dtype=torch.float32)
        theta_dict_init = {"0": torch.tensor(2.0, dtype=torch.float32),
                           "1": torch.tensor(5.0, dtype=torch.float32)}
        theta_dict_init_null = {k: v.clone() for k, v in theta_dict_init.items() if k == null_regime}
        
        # Fit null model
        print("\nFitting null model...")
        null_optimal_negll, null_optimal_alpha, null_optima_sigma2, null_optimal_theta_dict = \
            adam_optimize_ou_parameters(null_tree_copy, alpha_init, sigma2_init, theta_dict_init_null, root_expression, poisson_logl_mode)
        
        # Fit alt model
        print("\nFitting alternative model...")
        alt_optimal_negll, alt_optimal_alpha, alt_optima_sigma2, alt_optimal_theta_dict = \
            adam_optimize_ou_parameters(alt_tree_copy, alpha_init, sigma2_init, theta_dict_init, root_expression, poisson_logl_mode)
        
        # Compute LRT statistic
        null_log_lik = -null_optimal_negll.item()
        alt_log_lik = -alt_optimal_negll.item()
        lrt_statistic, p_value = calculate_lrt_and_pvalue(null_log_lik, alt_log_lik)
        
        print()
        print(f"\tLRT statistic = {lrt_statistic}")
        print(f"\tp-value = {p_value}")
        
    finally:
        # Write captured stdout to log file
        with open(log_file, "w") as f:
            f.write(buf.getvalue())
        sys.stdout = sys_stdout
        

def run_lrt():
    
    parser = argparse.ArgumentParser("Stochastic simulation of gene expression evolution along lineage")
    parser.add_argument("--tree", type=str, required=True, help="File path of input tree")
    parser.add_argument("--regime", type=str, required=True, help="File path of input regime that defined which theta each node belongs to in the OU model")
    parser.add_argument("--expression_data", type=str, required=True, help="File path of input TSV expression data in cells x genes format")
    parser.add_argument("--root_expression", type=int, default=2, help="Starting expression at the root")
    parser.add_argument("--null_regime", type=str, default="0", help="Regime label for all nodes under the null hypothesis")
    parser.add_argument("--poisson_logl_mode", type=str, choices=["none", "deterministic", "stochastic", "variational"], default="variational", 
                        help="Mode for logl calculation if transformation/poisson is added after the OU model, either 'none' (logl for only the OU model), 'deterministic', 'stochastic', or 'variational'.")
    parser.add_argument("--output_dir", type=str, default="./examples/output_results", help="Output directory for log files")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for parallel processing")
    args = parser.parse_args()
    
    # Save some args locally
    null_regime = args.null_regime
    poisson_logl_mode = args.poisson_logl_mode
    output_dir = args.output_dir
    threads = args.threads
    
    # Setup tensors
    root_expression = torch.tensor(float(args.root_expression), dtype=torch.float32)
    
    # Read in the null hypothesis tree without regimes
    null_tree = read_tree(args.tree)
    assign_nodes_to_null_regimes(null_tree, null_regime=args.null_regime)

    # Read in the alternative hypothesis tree with regimes
    alt_tree = read_tree(args.tree)
    assign_nodes_to_regimes_from_file(alt_tree, args.regime)
    
    read_count_data = load_read_count_tsv(args.expression_data)
    num_genes = len(read_count_data.keys())
    
    for i, gene in enumerate(read_count_data.keys(), 1):
        process_gene(gene, 
                        read_count_data, 
                        null_tree,
                        alt_tree, 
                        root_expression,
                        null_regime,
                        poisson_logl_mode,
                        output_dir)
        print(f"Completed {i}/{num_genes}: {gene} genes")
    
    
    # # Parallel processing (Not yet working properly, hence why it is commented out)
    # with ThreadPoolExecutor(max_workers=threads) as executor:
    #     futures = [executor.submit(process_gene, 
    #                             gene, 
    #                             read_count_data, 
    #                             null_tree,
    #                             alt_tree, 
    #                             root_expression,
    #                             output_dir,
    #                             args)
    #             for gene in read_count_data.keys()]
    #     # Wait for completion and raise any exceptions
    #     for i, f in enumerate(as_completed(futures), 1):
    #         gene = futures[f]
    #         try:
    #             f.result()  # raises exceptions if any
    #         except Exception as e:
    #             print(f"Gene {gene} failed: {e}")
    #         print(f"Completed {i}/{num_genes}: {gene} genes")