"""Variational q-variance parameterization helpers.

Internally, the second half of each Lq parameter tensor stores log(s2), not a
raw q standard deviation. This keeps the variance strictly positive on every
forward pass and avoids sqrt/log of invalid variances in the MC likelihood.
Saved q files are still exported as mean + q standard deviation.
"""
import numpy as np
import torch


LOG_S2_MIN = -27.631021115928547  # log(1e-12)
LOG_S2_MAX = 20.0                 # exp(20) ~= 4.85e8
S2_EPS = 1e-12


def log_s2_from_std_array(std):
    std = np.maximum(np.asarray(std), np.sqrt(S2_EPS))
    return np.log(std**2 + S2_EPS)


def log_s2_from_std_tensor(std):
    return torch.log(std.clamp_min(S2_EPS**0.5).square() + S2_EPS)


def s2_from_log_s2(log_s2):
    return torch.exp(torch.clamp(log_s2, LOG_S2_MIN, LOG_S2_MAX)) + S2_EPS


def std_from_log_s2_tensor(log_s2):
    return torch.sqrt(s2_from_log_s2(log_s2))


def export_mean_std_tensor(lq_params, n_cells=None):
    if n_cells is None:
        n_cells = lq_params.shape[-1] // 2
    mean = lq_params[..., :n_cells]
    std = std_from_log_s2_tensor(lq_params[..., n_cells:2 * n_cells])
    return torch.cat((mean, std), dim=-1)
