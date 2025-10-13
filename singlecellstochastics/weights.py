import numpy as np
import torch


# calculate theta weight matrix
def theta_weight_W_numpy(alpha, epochs, beta):
    """
    Compute weight matrix of theta adapted from EvoGeneX.
    Calculated from time intervals and regimes along tree.
    Weights affect the expected optimal value theta at leaves.

    alpha: OU parameter for selection
    epochs: entire time intervals (node to root)
    t: entire time intervals (node to leaf)
    y: adjacent decays by time (node to parent node)
    beta: activated regime along lineage (leaf to root)
    W: theta weight (n_cells, n_regimes)
    """
    n_cells = len(epochs)
    n_regimes = beta[0].shape[1]
    W = np.zeros((n_cells, n_regimes))

    # calculate theta weight for each cell
    for i, ep in enumerate(epochs):
        t = ep[0] - ep  # time intervals from leaf to each node
        y = np.exp(-alpha * t)  # exponential decay by time
        y[:-1] = (
            y[:-1] - y[1:]
        )  # adjacent changes y[i] = y[i] - y[i+1], keeping the last (root)
        W[i, :] = np.sum(
            y[:, None] * beta[i], axis=0
        )  # (1, n_regimes): changes * active regimes

    return W


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
        t = (
            t.unsqueeze(0).unsqueeze(0).expand(batch_size, N_sim, -1)
        )  # (batch_size, N_sim, L)
        y = torch.exp(-alpha_batch[:, :, None] * t)  # (batch_size, N_sim, L)

        # Compute differences: y[i] = y[i] - y[i+1] for i < L-1, keep last element
        y_diff = torch.cat(
            [y[:, :, :-1] - y[:, :, 1:], y[:, :, -1:]], dim=2
        )  # (batch_size, N_sim, L)

        # Compute weighted sum using matrix multiplication
        W_i = torch.matmul(y_diff, beta[i].to(y.dtype))  # (batch_size, N_sim, n_regimes)
        W_list.append(W_i)

    W = torch.stack(W_list, dim=2)  # (batch_size, N_sim, n_cells, n_regimes)
    return W
