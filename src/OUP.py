#!/usr/bin/env python

from Bio import Phylo
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.special import logsumexp
from statsmodels.stats.multitest import multipletests
import torch
import numpy as np
import pandas as pd
import csv
import argparse
from joblib import Parallel, delayed
import wandb

# wandb
wandb.login()
wandb.init(project="SingleCellStochastics", name="OUP")

# seed
torch.manual_seed(42)  # For GPU version
np.random.seed(42)     # For CPU version

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# input parameters
parser = argparse.ArgumentParser(description="Stochastic OUP model for gene expression evolution")
parser.add_argument("--tree", type=str, required=True, help="Newick tree file (comma-separated if multiple clones)")
parser.add_argument("--expr", type=str, required=True, help="Cell by gene count matrix (comma-separated if multiple clones)")
parser.add_argument("--annot", type=str, default=None, help="Gene annotation file (optional)")
parser.add_argument("--regime", type=str, required=True, help="Regime file (comma-separated if multiple clones)")
parser.add_argument("--null", type=str, required=True, help="Regime for null hypothesis")
parser.add_argument("--output", type=str, default="output", help="Output file")
parser.add_argument("--batch", type=int, default=100, help="Number of genes for batch processing")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
parser.add_argument("--iter", type=int, default=500, help="Max number of iterations for optimization")
parser.add_argument("--sim1", type=int, default=None, help="Number of simulations for empirical null distribution (one distribution for all genes)")
parser.add_argument("--sim2", type=int, default=None, help="Number of simulations for empirical null distribution (one distribution for each gene)")
args = parser.parse_args()

tree_files = args.tree.split(",")
gene_files = args.expr.split(",")
regime_files = args.regime.split(",")
batch_size = args.batch
rnull = args.null
learning_rate = args.lr
max_iter = args.iter
N_sim_all = args.sim1
N_sim_each = args.sim2
annot_file = args.annot
output_file = args.output

# precompute and store input data for later usage
def process_data(tree_files, gene_files, regime_files, rnull):
    '''
    Precompute input data and store for optimization.

    Returns:
    tree_list: list of trees
    cells_list: list of cells
    df_list: list of expression data
    diverge_list: list of divergence matrices (tree)
    share_list: list of share matrices (tree)
    epochs_list: list of depths (tree)
    beta_list: list of active regimes (tree)
    '''
    tree_list = []
    cells_list = []
    df_list = []
    diverge_list = []
    share_list = []
    epochs_list = []
    beta_list = []
    diverge_list_torch = []
    share_list_torch = []
    epochs_list_torch = []
    beta_list_torch = []
    regime_list = []

    for tree_file, gene_file, regime_file in zip(tree_files, gene_files, regime_files):
        # Read the phylogenetic tree
        tree = Phylo.read(tree_file, 'newick')
        cells = sorted([n.name for n in tree.get_terminals()])
        tree_list.append(tree)
        cells_list.append(cells)

        # Read the gene read count matrix, reorder cells as tree leaves
        df = pd.read_csv(gene_file, sep="\t", index_col=0, header=0)
        df.index = df.index.astype(str) # cell names
        df = df.loc[cells] # important to match cell orders!
        df_list.append(df)

        # Edit tree
        total_length = max(tree.depths().values())
        clades = [n for n in tree.find_clades()] # sort by depth-first search
        i = 0
        for clade in clades:
            # Re-scale branch lengths to [0, 1]
            if clade.branch_length is None: 
                clade.branch_length = 0
            else:
                clade.branch_length = clade.branch_length / total_length
            
            # Fix internal node names 
            if clade.name is None: 
                clade.name = "internal_node" + str(i)
                i = i + 1
        
        # Store paths from each node to root (add an extra root)
        #root = [Phylo.BaseTree.Clade(branch_length=0.0, name='root')]
        #paths = {'root': root}
        #paths.update({clade.name: root + tree.get_path(clade) for clade in clades})
        paths = {clade.name: tree.get_path(clade) for clade in clades}

        # Precompute node index mapping for all nodes and cells
        node_names = list(paths.keys())
        node_idx = {name: i for i, name in enumerate(node_names)}
        cell_indices = np.array([node_idx[cell] for cell in cells])

        # Precompute paths sets and reverse lineages for each cell
        cell_paths = [set(a.name for a in paths[cell]) for cell in cells]
        cell_lineages = [[a.name for a in paths[cell]][::-1] for cell in cells]

        # Build MRCA index matrix for all cell pairs
        mrca_idx = np.zeros((len(cells), len(cells)), dtype=int)
        for i in range(len(cells)):
            for j in range(len(cells)):
                # Fast set intersection
                common_ancestors = cell_paths[i] & cell_paths[j]
                # Find the deepest (last in path) ancestor
                for ancestor in cell_lineages[i]:
                    if ancestor in common_ancestors:
                        mrca_idx[i, j] = node_idx[ancestor]
                        break

        # depths (time intervals) for all nodes
        depth = np.array([sum(x.branch_length for x in paths[name]) for name in node_names], dtype=np.float32)        
        share_mat = depth[mrca_idx] # depth of MRCA
        share_torch = torch.tensor(share_mat, dtype=torch.float32, device=device)

        # diverge_mat: depth[a] + depth[b] - 2 * depth[mrca(a, b)]
        depth_a = depth[cell_indices][:, None]  # shape (n_cells, 1)
        depth_b = depth[cell_indices][None, :]  # shape (1, n_cells)
        diverge_mat = depth_a + depth_b - 2 * share_mat
        diverge_torch = torch.tensor(diverge_mat, dtype=torch.float32, device=device)

        # depths of each cell lineage (leaf to root)
        epochs = [np.array([depth[node_idx[x]] for x in lineage]) for lineage in cell_lineages]
        epochs_torch = [torch.tensor(ep, dtype=torch.float32, device=device) for ep in epochs]
        
        # regime for thetas
        with open(regime_file, 'r') as f:
            csv_file = csv.reader(f)
            next(csv_file)
            node_regime = {}
            regimes = set()
            for row in csv_file:
                if row[1] == '':
                    # mrca of a cell with itself
                    cell_idx = cells.index(row[0])
                    node = mrca_idx[cell_idx, cell_idx]
                else:
                    # mrca of two cells
                    cell1_idx = cells.index(row[0])
                    cell2_idx = cells.index(row[1])
                    node = mrca_idx[cell1_idx, cell2_idx]

                # regime for each node
                node_regime[node] = row[2]
                regimes.add(row[2])

        # sort regimes and move rnull to 1st
        regimes = sorted(list(regimes))
        regimes.remove(rnull)
        regimes.insert(0, rnull)
        regime_list.append(regimes)
        regime_idx = {regime: i for i, regime in enumerate(regimes)}

        # beta is binary label for activation of regime along each lineage (leaf to root)
        beta = []
        for lineage in cell_lineages:
            regime_indices = [regime_idx[node_regime[node_idx[x]]] for x in lineage] # active regimes
            beta_matrix = np.eye(len(regimes), dtype=int)[regime_indices] # (n_lineage, n_regimes)
            beta.append(beta_matrix)
        beta_torch = [torch.tensor(beta_matrix, dtype=torch.float32, device=device) for beta_matrix in beta]

        diverge_list.append(diverge_mat)
        share_list.append(share_mat)
        epochs_list.append(epochs)
        beta_list.append(beta)
        diverge_list_torch.append(diverge_torch)
        share_list_torch.append(share_torch)
        epochs_list_torch.append(epochs_torch)
        beta_list_torch.append(beta_torch)

    return tree_list, cells_list, df_list, diverge_list, share_list, epochs_list, beta_list, \
        diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch, regime_list


# calculate theta weight matrix
def theta_weight_W_numpy(alpha, epochs, beta):
    '''
    Compute weight matrix of theta adapted from EvoGeneX.
    Calculated from time intervals and regimes along tree.
    Weights affect the expected optimal value theta at leaves.

    alpha: OU parameter for selection
    epochs: entire time intervals (node to root)
    t: entire time intervals (node to leaf)
    y: adjacent decays by time (node to parent node)
    beta: activated regime along lineage (leaf to root)
    W: theta weight (n_cells, n_regimes)
    '''
    n_cells = len(epochs)
    n_regimes = beta[0].shape[1]
    W = np.zeros((n_cells, n_regimes))
    
    # calculate theta weight for each cell
    for i, ep in enumerate(epochs):
        t = ep[0] - ep          # time intervals from leaf to each node
        y = np.exp(-alpha * t)  # exponential decay by time
        y[:-1] = y[:-1] - y[1:] # adjacent changes y[i] = y[i] - y[i+1], keeping the last (root)
        W[i, :] = np.sum(y[:, None] * beta[i], axis=0) # (1, n_regimes): changes * active regimes
    
    return W


# calculate negative log likelihood of OU
def ou_neg_log_lik_numpy(params, mode, expr_list, diverge_list, share_list, epochs_list, beta_list):
    '''
    Compute two times negative log likelihood of OU along tree adapted from EvoGeneX.
    Use expression data directly as the OU output.
    Assume the same OU parameters for all trees and average likelihoods with log-sum-exp trick.

    params: OU parameters (alpha, sigma2, theta0, theta1, ..., theta_n)
    mode: 1 for H0, 2 for H1
    expr: expression data (n_cells, 1)
    diverge_list: list of divergence matrices (n_cells, n_cells)
    share_list: list of share matrices (n_cells, n_cells)
    epochs_list: list of time intervals (n_cells, 1)
    beta_list: list of regime activation matrices (n_cells, n_regimes)
    V: covariance matrix from trees (n_cells, n_cells)
    W: theta weight (n_cells, n_regimes)
    '''
    alpha, sigma2, theta0 = params[:3]
    n_trees = len(diverge_list)

    # share the same OU parameters for all trees
    loss_list = []
    for i in range(n_trees):
        expr = expr_list[i]
        n_cells = len(expr)
        diverge = diverge_list[i]
        share = share_list[i]
        epochs = epochs_list[i]
        beta = beta_list[i]
        n_regimes = beta[0].shape[1]
        
        # V: covariance matrix from trees (excluding variance sigma2)
        V = (1 / (2 * alpha)) * np.exp(-alpha * diverge) * (1 - np.exp(-2 * alpha * share))

        # diff = observed expression - expected values (n_cells, 1)
        if mode == 1:
            W = np.ones(n_cells) # expected values at leaves are same as theta0 at root
            diff = expr - W * theta0
        elif mode == 2:
            W = theta_weight_W_numpy(alpha, epochs, beta)
            diff = expr - W @ params[-n_regimes:]
        
        # log(det V) + (diff @ V^-1 @ diff) / sigma2 + n_cells * log(sigma2)
        log_det = np.linalg.slogdet(V)[1] # log det V
        exp = (diff @ np.linalg.solve(V, diff)).item() # diff @ V^-1 @ diff
        loss = log_det + exp / sigma2 + n_cells * np.log(sigma2) # -2 * log likelihood
        loss_list.append(loss)

    loss_list = np.array(loss_list)
    average_L = logsumexp(loss_list) - np.log(n_trees)
    return average_L


# optimize OU likelihood with SciPy
def ou_optimize_scipy(params_init, mode, expr_list, diverge_list, share_list, epochs_list, beta_list):
    """
    Optimize OU likelihood with SciPy L-BFGS-B.
    Returns OU parameters for each batch and sim.
    Used for initializing OU parameters for ELBO with expression data.

    params_init: OU parameters (alpha, sigma2, theta0, theta1, ..., theta_n)
    mode: 1 for H0, 2 for H1
    expr: list of expression data (batch_size, N_sim, n_cells)
    """
    batch_size, N_sim, _ = expr_list[0].shape # same for all trees
    
    # optimize OU for batch i, sim j
    def fit_one(i, j):
        expr = [x[i, j, :] for x in expr_list]
        res = minimize(ou_neg_log_lik_numpy, params_init, args=(mode, expr, 
            diverge_list, share_list, epochs_list, beta_list),
            bounds=[(1e-6, None)]*len(params_init), method="L-BFGS-B")
        return i, j, res.x

    # parallel execution
    results = Parallel(n_jobs=-1)(delayed(fit_one)(i, j)
                for i in range(batch_size) for j in range(N_sim))

    # Convert results to tensor
    result_tensor = torch.empty((batch_size, N_sim, len(params_init)), dtype=torch.float32, device=device)
    for i, j, x in results:
        result_tensor[i, j, :] = torch.tensor(x, dtype=torch.float32, device=device)

    return result_tensor


# calculate theta weight matrix with torch
def theta_weight_W_torch(alpha_batch, epochs, beta):
    """
    Same weight matrix with torch. Used for ELBO with torch.

    alpha_batch: (batch_size, N_sim)
    epochs: list of n_cells arrays, each (L,)
    beta: list of n_cells arrays, each (L, n_regimes)
    Returns: (batch_size, N_sim, n_cells, n_regimes)
    """
    batch_size, N_sim = alpha_batch.shape
    W_list = []  # Use a list to avoid in-place operations

    # for each cell
    for i, ep in enumerate(epochs):
        t = ep[0] - ep  # (L,)
        t = t.unsqueeze(0).unsqueeze(0).expand(batch_size, N_sim, -1)  # (batch_size, N_sim, L)
        y = torch.exp(-alpha_batch[:, :, None] * t)  # (batch_size, N_sim, L)

        # Compute differences: y[i] = y[i] - y[i+1] for i < L-1, keep last element
        y_diff = torch.cat([y[:, :, :-1] - y[:, :, 1:], y[:, :, -1:]], dim=2)  # (batch_size, N_sim, L)

        # Compute weighted sum using matrix multiplication
        W_i = torch.matmul(y_diff, beta[i])  # (batch_size, N_sim, n_regimes)
        W_list.append(W_i)
    
    W = torch.stack(W_list, dim=2)  # (batch_size, N_sim, n_cells, n_regimes)
    return W


# calculate expectation of negative OU log likelihood with torch
def ou_neg_log_lik_torch(params_batch, sigma2_q, mode, expr_batch, diverge, share, epochs, beta):
    """
    Same OU likelihood with torch. Used for ELBO with torch.

    params_batch: (batch_size, N_sim, ou_param_dim)
    sigma2_q: (batch_size, N_sim, n_cells)
    expr_batch: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    Returns: (batch_size, N_sim) tensor of losses
    """
    batch_size, N_sim, n_cells = expr_batch.shape

    # Extract parameters
    alpha = params_batch[:, :, 0]  # (batch_size, N_sim)
    sigma2 = params_batch[:, :, 1]  # (batch_size, N_sim)
    thetas = params_batch[:, :, 2:]  # (batch_size, N_sim, n_regimes)

    # Compute V for all batches (broadcast alpha)
    alpha = alpha[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    sigma2 = sigma2[:, :, None, None]  # (batch_size, N_sim, 1, 1)
    diverge = diverge[None, None, :, :]  # (1, 1, n_cells, n_cells)
    share = share[None, None, :, :]      # (1, 1, n_cells, n_cells)
    V = (1 / (2 * alpha)) * torch.exp(-alpha * diverge) * (1 - torch.exp(-2 * alpha * share))  # (batch_size, N_sim, n_cells, n_cells)

    # Compute W and diff
    if mode == 1:
        W = torch.ones((batch_size, N_sim, n_cells), dtype=torch.float32, device=device)
        diff = expr_batch - W * thetas[:, :, 0:1] # (batch_size, N_sim, n_cells)
    elif mode == 2:
        W = theta_weight_W_torch(alpha.squeeze(-1).squeeze(-1), epochs, beta)  # (batch_size, N_sim, n_cells, n_regimes)
        diff = expr_batch - torch.matmul(W, thetas.unsqueeze(-1)).squeeze(-1)  # (batch_size, N_sim, n_cells)

    L = torch.linalg.cholesky(V)  # (batch_size, N_sim, n_cells, n_cells)
    # log|V| = 2 * sum(log diag(L))
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=2, dim2=3)), dim=2)  # (batch_size, N_sim)

    # d^T V^{-1} d = ||L^{-1} d||^2
    d = diff.unsqueeze(-1)  # (batch, N_sim, n_cells, 1)
    y = torch.linalg.solve_triangular(L, d, upper=False)  # y = L^{-1} d (batch, N_sim, n_cells, 1)
    exp = (y.squeeze(-1) ** 2).sum(dim=-1)  # sum(y^2) = y^T y = ||y||^2 (batch, N_sim)

    # trace(V^{-1} Σ) = ||diag(L^{-1} Σ)||_F^2
    S_half = torch.diag_embed(torch.sqrt(sigma2_q))         # Σ^{1/2} = diag(σ_i), shape (..., n, n)
    Y = torch.linalg.solve_triangular(L, S_half, upper=False) # Y = L^{-1} Σ^{1/2}
    tr_term = (Y ** 2).sum(dim=(-2, -1))                      # ||Y||_F^2 = tr(C^{-1} Σ)

    loss = log_det + n_cells * torch.log(sigma2).squeeze(-1).squeeze(-1) + exp / sigma2.squeeze(-1).squeeze(-1) - 2 * tr_term
    return loss  # (batch_size, N_sim)


# calculate negative log likelihood of ELBO with torch
def Lq_neg_log_lik_torch(Lq_params, ou_params, mode, x_tensor, diverge, share, epochs, beta):
    """
    ELBO for approximating model evidence.

    Lq_params: (batch_size, N_sim, 2*n_cells)
    ou_params: (batch_size, N_sim, ou_param_dim)
    mode: 1 for H0, 2 for H1
    x_tensor: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells)
    epochs, beta: list as before
    Returns: (batch_size, N_sim) tensor of losses
    """
    n_cells = x_tensor.shape[-1]
    m = Lq_params[:, :, :n_cells]  # (batch_size, N_sim, n_cells)
    s2 = Lq_params[:, :, n_cells:]  # (batch_size, N_sim, n_cells)
    
    # OU -log lik
    term1 = 0.5 * ou_neg_log_lik_torch(ou_params, s2, mode, m, diverge, share, epochs, beta)  # (batch_size, N_sim)

    # Approximation for E[log(softplus(z))] and E[softplus(z)] TODO: try exponential instead of softplus
    def E_log_softplus(u, sigma2): # Taylor series for E[log]
        result = torch.empty_like(u)
        
        # u < 2, taylor0 + log(exp(z))
        def weighted(u, sigma2):
            log2 = torch.log(torch.tensor(2.0, device=device))
            
            # Taylor expansion at zero
            taylor0 = (torch.log(log2) + (0.5/log2) * u + ((log2 - 1) / (8 * log2**2)) * (u**2 + sigma2))
            
            # Weighting term
            w = 1 / (1 + torch.exp(2 * (u + 2)))
            
            return (1 - w) * taylor0 + w * u
    
        # 2 <= u < 10, log(z + exp(-z))
        def approx(u, sigma2):
            return torch.log(u + torch.exp(-u)) - ((1 - torch.exp(-u))**2) / (2 * (u + torch.exp(-u))**2) * sigma2
        
        # u >= 10, logz
        def log_approx(u, sigma2):
            return torch.log(u) #- sigma2 / (2 * u**2)
        
        result[u < 2] = weighted(u[u < 2], sigma2[u < 2])
        result[(u >= 2) & (u < 10)] = approx(u[(u >= 2) & (u < 10)], sigma2[(u >= 2) & (u < 10)])
        result[u >= 10] = log_approx(u[u >= 10], sigma2[u >= 10])
        
        return result

    def E_softplus(u, sigma2): # Taylor series for E[softplus]
        result = torch.empty_like(u)
        
        # u < 5, taylor(u)
        def taylor(u, sigma2):
            softplus = torch.nn.functional.softplus(u)
            sig = torch.sigmoid(u)
            return softplus + 0.5 * sigma2 * sig * (1 - sig)
        
        # u >= 5, z #+ exp(-z)
        def approx(u, sigma2):
            return u #+ torch.exp(-u + sigma2/2)
        
        result[u < 5] = taylor(u[u < 5], sigma2[u < 5])
        result[u >= 5] = approx(u[u >= 5], sigma2[u >= 5])
        
        return result

    # Poisson -log lik
    term2 = -torch.sum(x_tensor * E_log_softplus(m, s2) - E_softplus(m, s2), dim=2)  # (batch_size, N_sim)

    # -entropy
    term3 = -torch.sum(0.5 * torch.log(s2), dim=2)  # (batch_size, N_sim)
    
    loss = term1 + term2 + term3  # (batch_size, N_sim)
    return loss  # (batch_size, N_sim)


# optimize ELBO with Adam
def Lq_optimize_torch(params, mode, x_tensor_list, gene_names, diverge_list_torch, share_list_torch, \
            epochs_list_torch, beta_list_torch, max_iter=max_iter, learning_rate=learning_rate):
    """
    Optimize ELBO with PyTorch Adam.

    params: (batch_size, N_sim, all_param_dim)
    mode: 1 for H0, 2 for H1
    x_tensor: list of (batch_size, N_sim, n_cells)
    gene_names: list of gene names
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    params_tensor = [p.clone().detach().requires_grad_(True) for p in params] # list of (batch_size, N_sim, param_dim)
    batch_size, N_sim, _ = params_tensor[0].shape
    n_trees = len(x_tensor_list)
    best_params = params[-1].clone().detach() # (batch_size, N_sim, ou_param_dim)
    best_loss = torch.full((batch_size, N_sim), float('inf'), device=device)
    optimizer = torch.optim.Adam(params_tensor, lr=learning_rate)
    
    for n in range(max_iter):
        optimizer.zero_grad()

        # get loss for each tree
        loss_matrix = torch.zeros(batch_size, N_sim, n_trees, device=device)
        for i in range(n_trees):
            x_tensor = x_tensor_list[i]
            Lq_params = params_tensor[i]
            ou_params = params_tensor[-1]
            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]
            loss = Lq_neg_log_lik_torch(Lq_params, ou_params, mode, x_tensor, diverge, share, epochs, beta)
            loss_matrix[:, :, i] = loss

        # average loss across trees (use torch.logsumexp for better numerical stability)
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(torch.tensor(n_trees, device=device, dtype=torch.float32)) # (batch_size, N_sim)

        # update params
        loss = average_loss.sum()
        loss.backward()

        with torch.no_grad():
            # s2, alpha, sigma2 should be positive TODO: try log/square transform
            for i in range(n_trees-1):
                n_cells = x_tensor_list[i].shape[-1]
                if (params_tensor[i][:, :, n_cells:] < 1e-6).any():
                    params_tensor[i][:, :, n_cells:].clamp_(min=1e-6)
            if (params_tensor[-1][:, :, :2] < 1e-6).any():
                params_tensor[-1][:, :, :2].clamp_(min=1e-6)

            # update best params
            mask = average_loss < best_loss 
            best_loss[mask] = average_loss[mask].clone().detach()
            best_params[mask] = params_tensor[-1][mask].clone().detach()

            # plot loss of first gene in batch
            wandb.log({f"{gene_names[0]}_h{mode-1}_loss": best_loss[0, 0], "iter": n})
            
        optimizer.step()
        
    return best_params.clone().detach().cpu().numpy(), best_loss.clone().detach().cpu().numpy()


# simulate null distribution for each gene TODO: sim for each tree separately
def simulate_null_each(trees, h0_params, N_sim, cells):
    """
    Simulate N_sim OU processes along all trees in parallel for a batch of genes.

    h0_params: (batch_size, 1, all_param_dim) numpy array
    Returns: (batch_size, N_sim, n_cells) numpy array
    """
    n_trees = len(trees)
    sims_per_tree = N_sim // n_trees
    remainder = N_sim % n_trees

    x_sim_list = []
    for i, tree in enumerate(trees):
        # get number of simulations for this tree   
        n_sims = sims_per_tree + (1 if i < remainder else 0)
        alpha = np.tile(h0_params[:, 0, 2*len(cells)][:, np.newaxis], (1, n_sims)) # (batch_size, n_sims)
        sigma2 = np.tile(h0_params[:, 0, 2*len(cells)+1][:, np.newaxis], (1, n_sims)) # (batch_size, n_sims)
        theta0 = np.tile(h0_params[:, 0, 2*len(cells)+2][:, np.newaxis], (1, n_sims)) # (batch_size, n_sims)

        # simulate z_values for this tree by OU process
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10)) # (batch_size, n_sims)
        z_root = np.random.normal(loc=theta0, scale=std) # (batch_size, n_sims)
        z_values = {tree.root.name: z_root} # (batch_size, n_sims)

        for clade in tree.get_nonterminals(order='preorder'): # depth first search
            parent_z = z_values[clade.name]
            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = np.array(t, dtype=np.float32)
                mean = theta0 + (parent_z - theta0) * np.exp(-alpha * t) # (batch_size, n_sims)
                var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t)) # (batch_size, n_sims)
                var = np.maximum(var, 1e-10) # (batch_size, n_sims)
                std = np.sqrt(var) # (batch_size, n_sims)
                z_child = np.random.normal(loc=mean, scale=std) # (batch_size, n_sims)
                z_values[child.name] = z_child # (batch_size, n_sims)

        # combine z_values for all cells
        z_sim_matrix = np.stack([z_values[cell] for cell in cells], axis=2)  # (batch_size, n_sim, n_cells)

        # poisson sampling
        lambda_sim = np.log1p(np.exp(z_sim_matrix))  # softplus equivalent
        x_sim = np.random.poisson(lambda_sim) # (batch_size, n_sims, n_cells)
        x_sim_list.append(x_sim) # (batch_size, n_sims, n_cells)

    # combine x_sim for all trees
    x_sim_matrix = np.concatenate(x_sim_list, axis=1)  # (batch_size, N_sims, n_cells)
    return x_sim_matrix


# simulate null distribution for all genes TODO: sim for each tree separately
def simulate_null_all(trees, h0_params, N_sim_all, cells):
    """
    Sample ou params and simulate N_sim_all OU processes along all trees in parallel for all genes.
    
    h0_params: (gene_num, all_param_dim) numpy array
    Returns: (N_sim_all, n_cells) numpy array
    """
    n_trees = len(trees)
    sims_per_tree = N_sim_all // n_trees
    remainder = N_sim_all % n_trees

    # sample ou params for each tree
    n_genes = h0_params.shape[0]
    idx1 = np.random.choice(n_genes, size=sims_per_tree+1, replace=True)
    idx2 = np.random.choice(n_genes, size=sims_per_tree+1, replace=True)
    idx3 = np.random.choice(n_genes, size=sims_per_tree+1, replace=True)
    alpha_sample = h0_params[idx1, 2*len(cells)] # (sims_per_tree+1,)
    sigma2_sample = h0_params[idx2, 2*len(cells)+1] # (sims_per_tree+1,)
    theta0_sample = h0_params[idx3, 2*len(cells)+2] # (sims_per_tree+1,)

    x_sim_list = []
    start_idx = 0
    for i, tree in enumerate(trees):
        # get number of simulations for this tree   
        n_sims = sims_per_tree + (1 if i < remainder else 0)
        alpha = alpha_sample[start_idx:start_idx+n_sims] # (n_sims,)
        sigma2 = sigma2_sample[start_idx:start_idx+n_sims] # (n_sims,)
        theta0 = theta0_sample[start_idx:start_idx+n_sims] # (n_sims,)
        start_idx += n_sims

        # simulate z_values for this tree
        std = np.sqrt(np.maximum(sigma2 / (2 * alpha), 1e-10)) # (n_sims,)
        z_root = np.random.normal(loc=theta0, scale=std) # (n_sims,)
        z_values = {tree.root.name: z_root} # (n_sims,)

        for clade in tree.get_nonterminals(order='preorder'): # depth first search
            parent_z = z_values[clade.name]
            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = np.array(t, dtype=np.float32)
                mean = theta0 + (parent_z - theta0) * np.exp(-alpha * t) # (n_sims,)
                var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t)) # (n_sims,)
                var = np.maximum(var, 1e-10) # (n_sims,)
                std = np.sqrt(var) # (n_sims,)
                z_child = np.random.normal(loc=mean, scale=std) # (n_sims,)
                z_values[child.name] = z_child # (n_sims,)

        # combine z_values for all cells
        z_sim_matrix = np.stack([z_values[cell] for cell in cells], axis=1)  # (n_sim, n_cells)

        # poisson sampling
        lambda_sim = np.log1p(np.exp(z_sim_matrix))  # softplus equivalent
        x_sim = np.random.poisson(lambda_sim) # (n_sim, n_cells)
        x_sim_list.append(x_sim) # (n_sim, n_cells)                

    # combine x_sim for all trees
    x_sim_matrix = np.concatenate(x_sim_list, axis=0)  # (N_sim_all, n_cells)
    return x_sim_matrix
    

# likelihood ratio test
def likelihood_ratio_test(x_original, n_regimes, diverge_list, share_list, epochs_list, beta_list, \
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch, gene_names):
    """
    Hypothesis testing for lineage-specific gene expression change.

    x_original: (batch_size, N_sim, n_cells) numpy array
    diverge_list, share_list, epochs_list, beta_list: list of numpy arrays
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    gene_names: list of gene names
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    x_pseudo = [np.maximum(x, 1e-6) for x in x_original] # add small value to avoid log(0)
    m_init = [np.log(np.expm1(x)) for x in x_pseudo] # reverse read counts as Gaussian mean z
    ou_params_init = np.ones((n_regimes+2)) # shape: (n_regimes+2)

    # optimize OU for null model
    ou_params_h0 = ou_optimize_scipy(ou_params_init, 1, m_init, diverge_list, share_list, epochs_list, beta_list) # (batch_size, N_sim, n_regimes+2)

    # optimize Lq for null model
    m_init_tensor = [torch.tensor(m, dtype=torch.float32, device=device) for m in m_init] # list of (batch_size, N_sim, n_cells)
    pois_params_init = [torch.cat((m, torch.ones_like(m, device=device)), dim=2) for m in m_init_tensor] # list of (batch_size, N_sim, 2*n_cells)
    init_params = pois_params_init + [ou_params_h0] # list of (batch_size, N_sim, param_dim)
    
    x_original_tensor = [torch.tensor(x, dtype=torch.float32, device=device) for x in x_original] # list of (batch_size, N_sim, n_cells)
    h0_params, h0_loss = Lq_optimize_torch(init_params, 1, x_original_tensor, \
        gene_names, diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch) # (batch_size, N_sim, all_param_dim)

    # optimize OU for alternative model
    ou_params_h1 = ou_optimize_scipy(ou_params_init, 2, m_init, diverge_list, share_list, epochs_list, beta_list) # (batch_size, N_sim, n_regimes+2)

    # optimize Lq for alternative model
    init_params = pois_params_init + [ou_params_h1] # list of (batch_size, N_sim, param_dim)
    h1_params, h1_loss = Lq_optimize_torch(init_params, 2, x_original_tensor, \
        gene_names, diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch) # (batch_size, N_sim, all_param_dim)
    
    return h0_params, h0_loss, h1_params, h1_loss


# main
if __name__ == "__main__":
    results = {}
    results_empirical = {}
    results_empirical_all = {}
    ou_params_all = []
    lr_all = []

    # process data
    tree_list, cells_list, df_list, diverge_list, share_list, epochs_list, beta_list, \
        diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch, regime_list = \
        process_data(tree_files, gene_files, regime_files, rnull)

    regimes = list(dict.fromkeys(x for sub in regime_list for x in sub)) # regimes by order
    n_regimes = len(regimes)

    # loop over gene batches
    for batch_start in range(0, len(df_list[0].columns), batch_size):
        # gene expression in tissues (batch)
        batch_genes = df_list[0].columns[batch_start:batch_start+batch_size]
        # gene names
        if annot_file:
            df_annot = pd.read_csv(annot_file, sep="\t", index_col=0, header=None)
            batch_gene_names = df_annot.loc[batch_genes, 0].tolist()
        else:
            batch_gene_names = batch_genes
        print(f"\ngene batch {batch_start}-{batch_start+len(batch_genes)-1}: {list(batch_gene_names)}", flush=True)
        
        # pseudocount and init params
        x_original = [df[batch_genes].values.T for df in df_list] # list of (batch_size, n_cells)
        x_original = [np.expand_dims(x, axis=1) for x in x_original] # list of (batch_size, 1, n_cells)

        # likelihood ratio test (default)
        h0_params, h0_loss, h1_params, h1_loss = likelihood_ratio_test(x_original, n_regimes, \
            diverge_list, share_list, epochs_list, beta_list, \
            diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch, batch_gene_names) # (batch_size, 1, ...)
        lr = h0_loss - h1_loss # substract -log likelihood
        p_value = 1 - chi2.cdf(lr.flatten(), n_regimes-1)
        
        # for each gene in batch (chi-squared test)
        for i in range(batch_size): 
            h0_theta = np.log1p(np.exp(h0_params[i, 0, -n_regimes]))
            h1_theta = np.log1p(np.exp(h1_params[i, 0, -n_regimes:]))
            result = [batch_start + i, batch_genes[i], h0_theta] + h1_theta.tolist() + \
                [h0_loss[i, 0], h1_loss[i, 0], lr[i, 0], p_value[i]]
            results[batch_start + i] = result
            print("default result: " + '\t'.join(list(map(str, result))))

        # collect all ou params
        ou_params_all.append(h0_params[:, 0, :]) # (batch_size, ...)
        lr_all.append(lr[:, 0]) # (batch_size,)

        # empirical null distribution for each gene TODO: sim for each tree separately
        if N_sim_each:
            x_original = simulate_null_each(tree_list, h0_params, N_sim_each, cells) # (batch_size, N_sim, n_cells)
            _, h0_loss_sim, _, h1_loss_sim = likelihood_ratio_test(x_original, batch_gene_names) # (batch_size, N_sim, ...)
            null_LRs = h0_loss_sim - h1_loss_sim # (batch_size, N_sim)

            # for each gene in batch (using empirical)
            for i in range(batch_size):
                p_empirical = ((sum(null_LRs[i, :] >= lr[i, 0]) + 1) / (len(null_LRs[i, :]) + 1))
                result = results[batch_start + i][:-1] + [p_empirical]
                results_empirical[batch_start + i] = result
                print("sim_gene result: " + '\t'.join(list(map(str, result))))
        
    # FDR by Benjamini-Hochberg procedure
    results = sorted(list(results.values()), key=lambda x: x[-1])
    p_values = [r[-1] for r in results]
    signif, q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[:2]

    with open(output_file + "_chi-squared.tsv", 'w') as f:
        f.write(f"ID\tgene\th0_theta\t" + "\t".join(["h1_theta" + r for r in regimes]) + f"\th0\th1\tLR\tp\tq\tsignif\n")
        for i in range(len(results)):
            output = '\t'.join(list(map(str, results[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()

    # using empirical for each gene TODO: sim for each tree separately
    if N_sim_each: 
        results_empirical = sorted(list(results_empirical.values()), key=lambda x: x[-1])
        p_values = [r[-1] for r in results_empirical]
        signif, q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[:2]
        
        with open(output_file + "_empirical_each.tsv", 'w') as f:
            f.write(f"ID\tgene\th0_theta\t" + "\t".join(["h1_theta" + r for r in regimes]) + f"\th0\th1\tLR\tp\tq\tsignif\n")
            for i in range(len(results_empirical)):
                output = '\t'.join(list(map(str, results_empirical[i])))
                f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
                f.flush()

    # empirical null distribution for all genes TODO: sim for each tree separately TODO: output sims
    if N_sim_all:
        # simulate null for all genes
        ou_params_all = np.concatenate(ou_params_all, axis=0) # (gene_num, ...)
        x_original = simulate_null_all(tree_list, ou_params_all, N_sim_all, cells) # (N_sim_all, n_cells)
        
        # empirical null distribution for all genes
        x_original = np.expand_dims(x_original, axis=1) # shape: (N_sim_all, 1, n_cells)
        gene_names = ['sim_all'] * N_sim_all # (N_sim_all,)
        _, h0_loss_sim, _, h1_loss_sim = likelihood_ratio_test(x_original, gene_names) # (N_sim_all, 1, ...)
        null_LRs = h0_loss_sim[:, 0] - h1_loss_sim[:, 0] # (N_sim_all,)
        
        # LRT using empirical null distribution
        lr_all = np.concatenate(lr_all, axis=0) # (gene_num,)
        for i in range(lr_all.shape[0]):
            p_empirical_all = ((sum(null_LRs >= lr_all[i]) + 1) / (len(null_LRs) + 1))
            result = results[i][:-1] + [p_empirical_all]
            results_empirical_all[i] = result

        results_empirical_all = sorted(list(results_empirical_all.values()), key=lambda x: x[-1])
        p_values = [r[-1] for r in results_empirical_all]
        signif, q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[:2]
        
        with open(output_file + "_empirical_all.tsv", 'w') as f:
            f.write(f"ID\tgene\th0_theta\t" + "\t".join(["h1_theta" + r for r in regimes]) + f"\th0\th1\tLR\tp\tq\tsignif\n")
            for i in range(len(results_empirical_all)):
                output = '\t'.join(list(map(str, results_empirical_all[i])))
                f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
                f.flush()

