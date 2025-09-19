
import argparse

from .tree_utils import read_tree, assign_nodes_to_regimes_from_file, assign_nodes_to_null_regimes, add_read_counts_to_tips
from .input_output import load_read_count_tsv
from .ornstein_uhlenbeck import ou_neg_log_likelihood


def run_lrt():
    
    parser = argparse.ArgumentParser("Stochastic simulation of gene expression evolution along lineage")
    parser.add_argument("--tree", type=str, required=True, help="File path of input tree")
    parser.add_argument("--regime", type=str, required=True, help="File path of input regime")
    parser.add_argument("--expression_data", type=str, required=True, help="File path of input TSV expression data in cells x genes format")
    parser.add_argument("--root_expression", type=int, default=2, help="Starting expression at the root")
    parser.add_argument("--null_regime", type=str, default="0", help="Regime label for all nodes under the null hypothesis")
    parser.add_argument("--out", type=str, default="examples/input_data", help="Output directory")
    args = parser.parse_args()
    
    # Read in the null hypothesis tree without regimes
    null_tree = read_tree(args.tree)
    assign_nodes_to_null_regimes(null_tree, null_regime=args.null_regime)

    # Read in the alternative hypothesis tree with regimes
    alt_tree = read_tree(args.tree)
    assign_nodes_to_regimes_from_file(alt_tree, args.regime)
    
    read_count_data = load_read_count_tsv(args.expression_data)
    print(read_count_data)
    
    for gene in read_count_data.keys():
        add_read_counts_to_tips(null_tree, read_count_data[gene])
        add_read_counts_to_tips(alt_tree, read_count_data[gene])
        
        # Decide how to initialize parameters
        alpha_init = 1.0
        sigma2_init = 1.0
        theta_dict_init = {"0": 2.0, "1": 5.0}  # Test example initialization, should be based on regimes in the data
        
        # Fit null model
        null_neg_log_lik = ou_neg_log_likelihood(null_tree, alpha_init, sigma2_init, theta_dict_init, args.root_expression)
        
        # Fit alt model
        alt_neg_log_lik = ou_neg_log_likelihood(alt_tree, alpha_init, sigma2_init, theta_dict_init, args.root_expression)
        
        # Check results for testing
        print(f"Gene {gene}: Null NLL = {null_neg_log_lik}, Alt NLL = {alt_neg_log_lik}")
    
    
    
    
    