#!/usr/bin/env python

from Bio import Phylo
from scipy.stats import chi2, norm
from scipy.optimize import minimize
from scipy.special import logsumexp
from statistics import mean
from statsmodels.stats.multitest import multipletests
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import sys
import argparse
import math
import time
from joblib import Parallel, delayed
import threading
import wandb
import gc

wandb.login()
wandb.init(project="OU-Poisson-sim", name="OUP-BEAM")

# Add to both scripts
torch.manual_seed(42)  # For GPU version
np.random.seed(42)     # For CPU version

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# input parameters
parser = argparse.ArgumentParser(description="Script description")
parser.add_argument("-t", type=str, required=True, help="Clonal tree file")
parser.add_argument("-n", type=int, required=True, help="Number of tree samples")
parser.add_argument("-g", type=str, required=True, help="Gene read count file")
parser.add_argument("-m1", type=str, required=True, help="Regime file (primary)")
parser.add_argument("-m2", type=str, required=True, help="Regime file (metastatic)")
parser.add_argument("-b", type=int, required=False, default=1, help="Number of genes for batch processing") # default is 1
parser.add_argument("-r", type=str, required=False, help="Regime (tissue) for testing") # default is None
parser.add_argument("-lr", type=float, required=False, default=1e-3, help="Learning rate for Adam optimizer") # default is 1e-3
parser.add_argument("-i", type=int, required=False, default=1000, help="Max number of iterations for optimization") # default is 1000
parser.add_argument("-s1", type=int, required=False, default=None, help="Number of simulations for empirical null distribution (one distribution for all genes)") # default is None
parser.add_argument("-s2", type=int, required=False, default=None, help="Number of simulations for empirical null distribution (one distribution for each gene)") # default is None
parser.add_argument("-a", type=str, required=False, default=None, help="Annotation file") # default is None
args = parser.parse_args()

name = args.t
ntrees = args.n
gene_file = args.g
PRIregime_file = args.m1
METregime_file = args.m2
batch_size = args.b
rtest = args.r
learning_rate = args.lr
max_iter = args.i
N_sim_all = args.s1
N_sim = args.s2
annot_file = args.a

tree_list = []
diverge_list = []
share_list = []
epochs_list = []
beta_list = []
epochs_list_torch = []
beta_list_torch = []

# processing trees
for treeid in range(1, ntrees + 1):
    tree_file = f"{name}_{treeid}.nwk"
    
    # Read the phylogenetic tree
    tree = Phylo.read(tree_file, 'newick', rooted=True)
    cells = sorted([n.name for n in tree.get_terminals()])
    tree_list.append(tree)
    
    # Read the gene read count matrix, reorder cells as tree leaves
    df = pd.read_csv(gene_file, sep="\t", index_col=0, header=0)
    df.index = df.index.astype(str)
    df = df.loc[cells] # important to match cell orders!

    # Edit tree
    #total_length = max(tree.depths().values())
    clades = [n for n in tree.find_clades()] # sort by depth-first search
    i = 0
    for clade in clades:
        # Re-scale branch lengths to [0, 1]
        if clade.branch_length is None: 
            clade.branch_length = 0
        #else:
        #    clade.branch_length = clade.branch_length / total_length
        
        # Fix internal node names 
        if clade.name is None: 
            clade.name = "internal_node" + str(i)
            i = i + 1
    
    # Store paths from each node to root
    root = [Phylo.BaseTree.Clade(branch_length=0.0, name='root')]
    paths = {clade.name: root + tree.get_path(clade) for clade in clades}
    paths['root'] = root
    #paths = {clade.name: tree.get_path(clade) for clade in clades}

    # Precompute node index mapping for all nodes
    node_names = [clade.name for clade in clades]
    node_names.append('root')
    node_idx = {name: i for i, name in enumerate(node_names)}

    # Precompute paths for all cells (as sets for fast intersection)
    cell_paths = [set(a.name for a in paths[cell]) for cell in cells]

    # Build MRCA index matrix for all cell pairs (vectorized)
    mrca_idx = np.zeros((len(cells), len(cells)), dtype=int)
    for i in range(len(cells)):
        for j in range(len(cells)):
            # Find the deepest common ancestor (MRCA) by set intersection
            common_ancestors = cell_paths[i] & cell_paths[j]
            # Find the deepest (last in path) ancestor
            # Use the path of cell i, reversed, to find the deepest ancestor in order
            for ancestor in reversed([a.name for a in paths[cells[i]]]):
                if ancestor in common_ancestors:
                    mrca_idx[i, j] = node_idx[ancestor]
                    break

    # Vectorized depth calculation for all nodes
    depth_vec = np.array([sum(x.branch_length for x in paths[name]) for name in node_names], dtype=np.float32)

    # Get the index of each cell in node_names
    cell_indices = np.array([node_idx[cell] for cell in cells])

    # share_mat: depth of MRCA for each cell pair
    share_mat = depth_vec[mrca_idx]

    # diverge_mat: depth[a] + depth[b] - 2 * depth[mrca(a, b)]
    depth_a = depth_vec[cell_indices][:, None]  # shape (n_cells, 1)
    depth_b = depth_vec[cell_indices][None, :]  # shape (1, n_cells)
    diverge_mat = depth_a + depth_b - 2 * share_mat
    
    # time intervals of each cell lineage
    epochs = []
    for n in cells:
        lineage = [a.name for a in paths[n]][::-1] # leaf to root
        epochs.append(np.array([depth_vec[node_idx[x]] for x in lineage]))
    epochs_torch = [torch.tensor(ep, dtype=torch.float32, device=device) for ep in epochs]
    
    # regime for thetas
    with open(METregime_file, 'r') as f:
        csv_file = csv.reader(f)
        next(csv_file)
        node_regime = {}
        regimes = set()
        for row in csv_file:
            if row[1] == '':
                # Use cell indices for mrca_idx
                cell_idx = cells.index(row[0])
                node = mrca_idx[cell_idx, cell_idx]
            else:
                # Use cell indices for mrca_idx
                cell1_idx = cells.index(row[0])
                cell2_idx = cells.index(row[1])
                node = mrca_idx[cell1_idx, cell2_idx]

            # if rtest is provided, use default regime for all other regimes
            if rtest and row[2] != rtest:
                r = 'default'
            else:
                r = row[2]

            node_regime[node] = r
            regimes.add(r)
    
    # move root regime to 1st
    with open(PRIregime_file, 'r') as f:
        root_regime = f.readlines()[1].strip().split(",")[-1]
        if rtest and root_regime != rtest:
            root_regime = 'default'

    # sort regimes
    regimes = sorted(list(regimes))
    regimes.remove(root_regime)
    regimes.insert(0, root_regime)
    regime_idx = {regime: i for i, regime in enumerate(regimes)}

    # beta is binary label for activation of regime along each lineage
    beta = []
    for n in cells:
        lineage = [a.name for a in paths[n]][::-1] # from leaf to root
        regime_indices = [regime_idx[node_regime[node_idx[x]]] for x in lineage] # active regimes
        beta_matrix = np.eye(len(regimes), dtype=int)[regime_indices]
        beta.append(beta_matrix)
    beta_torch = [torch.tensor(beta_matrix, dtype=torch.float32, device=device) for beta_matrix in beta]

    diverge_list.append(diverge_mat)
    share_list.append(share_mat)
    epochs_list.append(epochs)
    beta_list.append(beta)
    epochs_list_torch.append(epochs_torch)
    beta_list_torch.append(beta_torch)

diverge_list_torch = [torch.tensor(diverge_mat, dtype=torch.float32, device=device) for diverge_mat in diverge_list]
share_list_torch = [torch.tensor(share_mat, dtype=torch.float32, device=device) for share_mat in share_list]

def theta_weight_W_numpy(alpha, epochs, beta):
    '''
    Weight matrix of theta adapted from EvoGeneX.
    Calculated from time intervals and regimes (sites) along tree.
    Weights control the expected optimal value theta at leaves.
    t: entire time intervals
    y: adjacent time intervals
    beta: activated regime
    W: theta weight
    '''
    n_cells = len(cells)
    n_regimes = len(regimes)
    W = np.zeros((n_cells, n_regimes))

    for i, ep in enumerate(epochs):
        t = ep[0] - ep          # shape (L,) time intervals to leaf
        y = np.exp(-alpha * t)  # shape (L,) time intervals btw nodes
        # Compute differences: y[i] = y[i] - y[i+1] for i < L-1, keep last element
        y[:-1] = y[:-1] - y[1:]  # differences for all but last
        W[i, :] = np.sum(y[:, None] * beta[i], axis=0) # (n_cells, n_regimes) time * active theta
    
    return W

def ou_neg_log_lik_numpy(params, mode, expr):
    '''
    Two times negative log likelihood of OU along tree adapted from EvoGeneX.
    Averaged from multiple tree samples with log-sum-exp trick.
    V: covariance matrix
    W: theta weight
    '''
    alpha, sigma2, theta0 = params[:3]

    # share the same OU parameters for all BEAM trees
    loss_list = []
    for i in range(ntrees):
        diverge = diverge_list[i]
        share = share_list[i]
        epochs = epochs_list[i]
        beta = beta_list[i]
        
        V = (1 / (2 * alpha)) * np.exp(-alpha * diverge) * (1 - np.exp(-2 * alpha * share))
        
        if mode == 1:
            W = np.ones(len(expr))
            diff = expr - W * theta0
        elif mode == 2:
            W = theta_weight_W_numpy(alpha, epochs, beta)
            diff = expr - W @ params[-len(regimes):]
        
        log_det = np.linalg.slogdet(V)[1]
        exp = (diff @ np.linalg.solve(V, diff)).item()
        loss = log_det + len(cells) * np.log(sigma2) + exp / sigma2
        loss_list.append(loss)

    loss_list = np.array(loss_list)
    average_L = logsumexp(loss_list) - np.log(ntrees)
    return average_L

def ou_optimize_scipy(params_init, mode, expr):
    """
    Optimize OU likelihood with SciPy.
    params_init: (len(regimes)+2)
    mode: 1 for H0, 2 for H1
    expr: (batch_size, N_sim, n_cells) numpy array
    Returns: (batch_size, N_sim, len(regimes)+2) tensor on GPU
    """
    batch_size, N_sim, n_cells = expr.shape
    
    def fit_one(i, j):
        res = minimize(ou_neg_log_lik_numpy, params_init, args=(mode, expr[i, j, :]),
            bounds=[(1e-6, None)]*2 + [(None, None)]*len(regimes), method="L-BFGS-B")
        return i, j, res.x

    # Parallel execution
    results = Parallel(n_jobs=-1)(delayed(fit_one)(i, j)
                for i in range(batch_size) for j in range(N_sim))

    # Convert results to tensor
    result_tensor = torch.empty((batch_size, N_sim, len(regimes)+2), dtype=torch.float32, device=device)
    for i, j, x in results:
        result_tensor[i, j, :] = torch.tensor(x, dtype=torch.float32, device=device)

    return result_tensor

def theta_weight_W_torch(alpha_batch, epochs, beta):
    """
    Same weight with torch.
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

def ou_neg_log_lik_torch(params_batch, mode, expr_batch, diverge, share, epochs, beta):
    """
    Same OU likelihood with torch.
    params_batch: (batch_size, N_sim, ou_param_dim)
    expr_batch: (batch_size, N_sim, n_cells)
    diverge, share: (n_cells, n_cells) [shared for all]
    epochs, beta: as before (shared for all)
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
    # log_det for each batch
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=2, dim2=3)), dim=2)  # (batch_size, N_sim)
    # Solve for exp for each batch
    diff = diff.unsqueeze(-1)  # (batch_size, N_sim, n_cells, 1)
    exp = torch.matmul(diff.transpose(2, 3), torch.cholesky_solve(diff, L)).squeeze(-1).squeeze(-1)  # (batch_size, N_sim)

    loss = log_det + n_cells * torch.log(sigma2).squeeze(-1).squeeze(-1) + exp / sigma2.squeeze(-1).squeeze(-1)
    return loss  # (batch_size, N_sim)

def Lq_neg_log_lik_torch(params_tensor, mode, x_tensor, diverge, share, epochs, beta):
    """
    ELBO for approximating model evidence.
    params_tensor: (batch_size, N_sim, all_param_dim)
    mode: 1 for H0, 2 for H1
    x_tensor: (batch_size, N_sim, n_cells)
    Returns: (batch_size, N_sim) tensor of losses
    """
    batch_size, N_sim, n_cells = x_tensor.shape
    m = params_tensor[:, :, :n_cells]  # (batch_size, N_sim, n_cells)
    s2 = params_tensor[:, :, n_cells:2*n_cells]  # (batch_size, N_sim, n_cells)
    ou_params = params_tensor[:, :, 2*n_cells:] # (batch_size, N_sim, ou_param_dim)
    
    # OU -log lik
    term1 = 0.5 * ou_neg_log_lik_torch(ou_params, mode, m, diverge, share, epochs, beta)  # (batch_size, N_sim)

    # Approximation for E[log(softplus(z))] and E[softplus(z)]
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

def Lq_optimize_torch(params, mode, x_tensor, gene_names, max_iter=max_iter, learning_rate=learning_rate):
    """
    Optimize ELBO with Adam.
    params: (batch_size, N_sim, all_param_dim)
    mode: 1 for H0, 2 for H1
    x_tensor: (batch_size, N_sim, n_cells)
    gene_names: list of gene names
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    params_tensor = params.clone().detach().requires_grad_(True) # (batch_size, N_sim, all_param_dim)
    batch_size = params_tensor.shape[0]
    N_sim = params_tensor.shape[1]
    best_params = params.clone().detach()
    best_loss = torch.full((batch_size, N_sim), float('inf'), device=device)
    optimizer = torch.optim.Adam([params_tensor], lr=learning_rate)
    
    for n in range(max_iter):
        optimizer.zero_grad()

        # get loss for each tree
        loss_matrix = torch.zeros(batch_size, N_sim, ntrees, device=device)
        for i in range(ntrees):
            diverge = diverge_list_torch[i]
            share = share_list_torch[i]
            epochs = epochs_list_torch[i]
            beta = beta_list_torch[i]
            loss = Lq_neg_log_lik_torch(params_tensor, mode, x_tensor, diverge, share, epochs, beta)
            loss_matrix[:, :, i] = loss

        # average loss across trees
        # Use torch.logsumexp for better numerical stability
        average_loss = torch.logsumexp(loss_matrix, dim=2) - torch.log(torch.tensor(ntrees, device=device, dtype=loss_matrix.dtype)) # (batch_size, N_sim)

        # update params
        loss = average_loss.sum()
        loss.backward()

        with torch.no_grad():
            # s2, alpha, sigma2 should be positive
            if (params_tensor[:, :, len(cells):-len(regimes)] < 1e-6).any():
                params_tensor[:, :, len(cells):-len(regimes)].clamp_(min=1e-6)

            # update best params
            mask = average_loss < best_loss 
            best_loss[mask] = average_loss[mask].clone().detach()
            best_params[mask] = params_tensor[mask].clone().detach()

            # plot loss of first gene in batch
            wandb.log({f"{gene_names[0]}_h{mode-1}_loss": best_loss[0, 0], "iter": n})
            
        optimizer.step()
        
    return best_params.clone().detach().cpu().numpy(), best_loss.clone().detach().cpu().numpy()

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
        alpha = np.tile(h0_params[:, 0, -len(regimes)-2][:, np.newaxis], (1, n_sims)) # (batch_size, n_sims)
        sigma2 = np.tile(h0_params[:, 0, -len(regimes)-1][:, np.newaxis], (1, n_sims)) # (batch_size, n_sims)
        theta0 = np.tile(h0_params[:, 0, -len(regimes)][:, np.newaxis], (1, n_sims)) # (batch_size, n_sims)

        # simulate z_values for this tree
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
    alpha_sample = h0_params[idx1, -len(regimes)-2] # (sims_per_tree+1,)
    sigma2_sample = h0_params[idx2, -len(regimes)-1] # (sims_per_tree+1,)
    theta0_sample = h0_params[idx3, -len(regimes)] # (sims_per_tree+1,)

    x_sim_list = []
    for i, tree in enumerate(trees):
        # get number of simulations for this tree   
        n_sims = sims_per_tree + (1 if i < remainder else 0)
        alpha = alpha_sample[:n_sims] # (n_sims,)
        sigma2 = sigma2_sample[:n_sims] # (n_sims,)
        theta0 = theta0_sample[:n_sims] # (n_sims,)

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
    
def likelihood_ratio_test(x_original, gene_names):
    """
    Hypothesis testing for lineage-specific gene expression change.
    x_original: (batch_size, N_sim, n_cells) numpy array
    gene_names: list of gene names
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    x_pseudo = np.maximum(x_original, 1e-6) # add small value to avoid log(0)
    m_init = np.log(np.expm1(x_pseudo)) # reverse read counts as Gaussian mean
    ou_params_init = np.ones((len(regimes)+2)) # shape: (len(regimes)+2)

    # optimize OU for null model
    ou_params_h0 = ou_optimize_scipy(ou_params_init, 1, m_init) # (batch_size, N_sim, len(regimes)+2)

    # optimize Lq for null model
    m_init_tensor = torch.tensor(m_init, dtype=torch.float32, device=device) # (batch_size, N_sim, n_cells)
    pois_params_init = torch.cat((m_init_tensor, torch.ones_like(m_init_tensor, device=device)), dim=2) # (batch_size, N_sim, 2*n_cells)
    x_original_tensor = torch.tensor(x_original, dtype=torch.float32, device=device)  # (batch_size, N_sim, n_cells)
    h0_params, h0_loss = Lq_optimize_torch(torch.cat((pois_params_init, ou_params_h0), dim=2), 1, x_original_tensor, gene_names) # (batch_size, N_sim, all_param_dim)

    # optimize OU for alternative model
    ou_params_h1 = ou_optimize_scipy(ou_params_init, 2, m_init) # (batch_size, N_sim, len(regimes)+2)

    # optimize Lq for alternative model
    h1_params, h1_loss = Lq_optimize_torch(torch.cat((pois_params_init, ou_params_h1), dim=2), 2, x_original_tensor, gene_names) # (batch_size, N_sim, all_param_dim)
    
    return h0_params, h0_loss, h1_params, h1_loss

results = {}
results_empirical = {}
results_empirical_all = {}
ou_params_all = []
lr_all = []
# loop over gene batches
for batch_start in range(0, len(df.columns), batch_size):
    # gene expression in tissues (batch)
    batch_genes = df.columns[batch_start:batch_start+batch_size]
    # gene names
    if annot_file:
        df_annot = pd.read_csv(annot_file, sep="\t", index_col=0, header=None)
        batch_gene_names = df_annot.loc[batch_genes, 0].tolist()
    else:
        batch_gene_names = batch_genes
    print(f"\ntree{name}, gene batch {batch_start}-{batch_start+len(batch_genes)-1}: {list(batch_gene_names)}", flush=True)
    
    # pseudocount and init params
    x_original = df[batch_genes].values.T # shape: (batch_size, n_cells)
    x_original = np.expand_dims(x_original, axis=1) # shape: (batch_size, 1, n_cells)

    # likelihood ratio test (default)
    h0_params, h0_loss, h1_params, h1_loss = likelihood_ratio_test(x_original, batch_gene_names) # (batch_size, 1, ...)
    lr = h0_loss - h1_loss # substract -log likelihood
    p_value = 1 - chi2.cdf(lr.flatten(), len(regimes)-1)
    
    # for each gene in batch (chi-squared test)
    for i in range(batch_size): 
        result = [batch_start + i, batch_genes[i], h0_params[i, 0, -len(regimes)]] + h1_params[i, 0, -len(regimes):].tolist() + \
            [h0_loss[i, 0], h1_loss[i, 0], lr[i, 0], p_value[i]]
        results[batch_start + i] = result
        print("default result: " + '\t'.join(list(map(str, result))))

    # collect all ou params
    ou_params_all.append(h0_params[:, 0, -len(regimes)-2:]) # (batch_size, ...)
    lr_all.append(lr[:, 0]) # (batch_size,)

    # empirical null distribution for each gene
    if N_sim:
        x_original = simulate_null_each(tree_list, h0_params, N_sim, cells) # (batch_size, N_sim, n_cells)
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

with open("OUP_tree" + name + "_chi-squared.tsv", 'w') as f:
    f.write(f"ID\tgene\ttheta0\t" + "\t".join([r for r in regimes]) + f"\th0\th1\tLR\tp\tq\tsignif\n")
    for i in range(len(results)):
        output = '\t'.join(list(map(str, results[i])))
        f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
        f.flush()

# using empirical for each gene
if N_sim: 
    results_empirical = sorted(list(results_empirical.values()), key=lambda x: x[-1])
    p_values = [r[-1] for r in results_empirical]
    signif, q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[:2]
    
    with open("OUP_tree" + name + "_empirical_each.tsv", 'w') as f:
        f.write(f"ID\tgene\ttheta0\t" + "\t".join([r for r in regimes]) + f"\th0\th1\tLR\tp\tq\tsignif\n")
        for i in range(len(results_empirical)):
            output = '\t'.join(list(map(str, results_empirical[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()

# empirical null distribution for all genes
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
    
    with open("OUP_tree" + name + "_empirical_all.tsv", 'w') as f:
        f.write(f"ID\tgene\ttheta0\t" + "\t".join([r for r in regimes]) + f"\th0\th1\tLR\tp\tq\tsignif\n")
        for i in range(len(results_empirical_all)):
            output = '\t'.join(list(map(str, results_empirical_all[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()

