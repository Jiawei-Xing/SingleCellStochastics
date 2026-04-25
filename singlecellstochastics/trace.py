"""
Lightweight tracer for ELBO / OU likelihood internals.

Off by default. When `TRACE.enabled` is True, the inner likelihood functions
write per-step diagnostics into `TRACE.diag` (a fresh dict at every step).
The optimizer then reads `TRACE.diag` after the forward pass and combines
it with parameter / gradient / Adam-state info before invoking the user's
`step_callback`.

The trace assumes a *single* gene at batch index 0 and a single sim at
index 0 (i.e. shape (1, 1, ...)) so we can flatten safely.
"""
import torch


class _TraceState:
    def __init__(self):
        self.enabled = False
        self.diag = {}

    def reset(self):
        self.diag = {}

    def write(self, key, value):
        if not self.enabled:
            return
        # store scalar tensors as floats; small tensors as numpy arrays
        if isinstance(value, torch.Tensor):
            try:
                if value.numel() == 1:
                    value = float(value.detach().cpu().item())
                else:
                    value = value.detach().cpu().numpy()
            except Exception:
                value = None
        self.diag[key] = value


TRACE = _TraceState()


def safe_float(x):
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float("nan")


def cov_diag(V):
    """V: (..., n, n). Return (cond_number, min_eig, has_neg_diag) for the first batch slot."""
    if V.dim() > 2:
        V0 = V.reshape(-1, V.shape[-2], V.shape[-1])[0]
    else:
        V0 = V
    V0 = 0.5 * (V0 + V0.transpose(-1, -2))
    try:
        eigs = torch.linalg.eigvalsh(V0)
        emin = float(eigs.min().detach().cpu())
        emax = float(eigs.max().detach().cpu())
        cond = emax / emin if emin > 0 else float("inf")
    except Exception:
        emin = float("nan")
        cond = float("nan")
    diag = torch.diagonal(V0)
    n_neg_diag = int((diag < 0).sum().detach().cpu())
    return cond, emin, n_neg_diag


def nan_inf_count(t):
    """Return (#nan, #inf) in tensor t."""
    if not isinstance(t, torch.Tensor):
        return 0, 0
    nan = int(torch.isnan(t).sum().detach().cpu())
    inf = int(torch.isinf(t).sum().detach().cpu())
    return nan, inf
