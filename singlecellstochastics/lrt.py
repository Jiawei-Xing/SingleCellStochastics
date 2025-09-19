
import argparse
import torch

from .tree_utils import read_tree, assign_nodes_to_regimes_from_file, assign_nodes_to_null_regimes, add_read_counts_to_tips
from .input_output import load_read_count_tsv
from .optimization import adam_optimize_ou_parameters
from .stat_utils import calculate_lrt_and_pvalue


def run_lrt():
    
    parser = argparse.ArgumentParser("Stochastic simulation of gene expression evolution along lineage")
    parser.add_argument("--tree", type=str, required=True, help="File path of input tree")
    parser.add_argument("--regime", type=str, required=True, help="File path of input regime that defined which theta each node belongs to in the OU model")
    parser.add_argument("--expression_data", type=str, required=True, help="File path of input TSV expression data in cells x genes format")
    parser.add_argument("--root_expression", type=int, default=2, help="Starting expression at the root")
    parser.add_argument("--null_regime", type=str, default="0", help="Regime label for all nodes under the null hypothesis")
    parser.add_argument("--poisson_logl_mode", type=str, choices=["none", "deterministic", "stochastic", "variational"], default="variational", 
                        help="Mode for logl calculation if transformation/poisson is added after the OU model, either 'none' (logl for only the OU model), 'deterministic', 'stochastic', or 'variational'.")
    args = parser.parse_args()
    
    # Setup tensors
    root_expression = torch.tensor(float(args.root_expression), dtype=torch.float32)
    
    # Read in the null hypothesis tree without regimes
    null_tree = read_tree(args.tree)
    assign_nodes_to_null_regimes(null_tree, null_regime=args.null_regime)

    # Read in the alternative hypothesis tree with regimes
    alt_tree = read_tree(args.tree)
    assign_nodes_to_regimes_from_file(alt_tree, args.regime)
    
    read_count_data = load_read_count_tsv(args.expression_data)
    
    for gene in read_count_data.keys():
        print(f"\nProcessing gene {gene}")
        add_read_counts_to_tips(null_tree, read_count_data[gene])
        add_read_counts_to_tips(alt_tree, read_count_data[gene])
        
        # Decide how to initialize parameters
        alpha_init = torch.tensor(1.0, dtype=torch.float32)
        sigma2_init = torch.tensor(1.0, dtype=torch.float32)
        theta_dict_init = {"0": torch.tensor(2.0, dtype=torch.float32), "1": torch.tensor(5.0, dtype=torch.float32)} # Test example initialization, should be based on regimes in the data
        
        theta_dict_init_null = {k: v.clone() for k, v in theta_dict_init.items() if k == args.null_regime}
        
        # Fit null model
        print("\nFitting null model...")
        null_optimal_negll, null_optimal_alpha, null_optima_sigma2, null_optimal_theta_dict = adam_optimize_ou_parameters(null_tree, alpha_init, sigma2_init, theta_dict_init_null, root_expression, args.poisson_logl_mode)
        
        # Fit alt model
        print("\nFitting alternative model...")
        alt_optimal_negll, alt_optimal_alpha, alt_optima_sigma2, alt_optimal_theta_dict = adam_optimize_ou_parameters(alt_tree, alpha_init, sigma2_init, theta_dict_init, root_expression, args.poisson_logl_mode)
        
        # Compute LRT statistic
        null_log_lik = -null_optimal_negll.item()
        alt_log_lik = -alt_optimal_negll.item()
        lrt_statistic, p_value = calculate_lrt_and_pvalue(null_log_lik, alt_log_lik)
        print(f"\tLRT statistic = {lrt_statistic}")
        print(f"\tp-value = {p_value}")
    
    
    
    
    