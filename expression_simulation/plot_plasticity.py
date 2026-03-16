"""
Scenario panels for latent process + softplus + NB/Poisson observation.

Each panel contains:
- Left: simulated latent trajectories (OU or BM)
- Right: horizontal histogram of observed counts after softplus + NB/Poisson
- One consistent color per panel, shared by trajectories and histogram

Scenarios:
    A-J from the provided specification

Usage:
    python plot_scenarios_softplus_nb.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

np.random.seed(42)

# ── Scenario specification ───────────────────────────────────────────
# label, root, sigma, alpha, optim, disp, bg
SCENARIOS = [
    ("A", 1.0, 3.0, 1.0, 1.0, 5.0, "OU"),   # Baseline
    ("B", 3.0, 3.0, 1.0, 3.0, 5.0, "OU"),   # High expression
    ("C", 1.0, 3.0, 1.0, 3.0, 5.0, "OU"),   # Diff theta
    ("D", 1.0, 5.0, 1.0, 1.0, 5.0, "OU"),   # High variance
    ("E", 1.0, 3.0, None, None, 5.0, "BM"), # Pure BM + NB
    ("F", 1.0, 3.0, 0.3, 1.0, 5.0, "OU"),   # Weak reversion
    ("G", 1.0, 3.0, 5.0, 1.0, 5.0, "OU"),   # Strong reversion
    ("H", 1.0, 3.0, 1.0, 1.0, 0.5, "OU"),   # Overdispersed
    ("I", 1.0, 3.0, 1.0, 1.0, 50.0, "OU"),  # Low overdispersion
    ("J", 1.0, 3.0, 1.0, 1.0, 0.0, "OU"),   # Poisson
]

SCENARIO_DESC = {
    "A": "Baseline",
    "B": "High expression",
    "C": r"$\\theta \\neq x_0$",
    "D": "High variance",
    "E": "BM + NB",
    "F": "Weak reversion",
    "G": "Strong reversion",
    "H": "Overdispersed",
    "I": "Low overdisp.",
    "J": "Poisson",
}

# one matching color per panel
CMAP = plt.get_cmap("tab10")
PANEL_COLORS = {label: CMAP(i % 10) for i, (label, *_rest) in enumerate(SCENARIOS)}


# ── Helpers ──────────────────────────────────────────────────────────
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sample_observation(mean, dispersion):
    """
    Observation model:
    - dispersion > 0: Negative binomial with Var = mean + mean^2 / dispersion
    - dispersion == 0: Poisson
    """
    mean = np.maximum(mean, 1e-8)

    if dispersion == 0:
        return np.random.poisson(mean)

    n = dispersion
    p = n / (n + mean)
    return np.random.negative_binomial(n, p)


def simulate_latent_paths(process, root, sigma, alpha, optim, n_paths=70, n_steps=100, T=1.0):
    """
    Simulate either OU or BM paths.
    sigma here is used as diffusion scale.
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = root

    for i in range(n_steps):
        noise = sigma * np.sqrt(dt) * np.random.randn(n_paths)

        if process == "BM":
            drift = 0.0
        elif process == "OU":
            drift = alpha * (optim - paths[:, i]) * dt
        else:
            raise ValueError(f"Unknown process: {process}")

        paths[:, i + 1] = paths[:, i] + drift + noise

    return t, paths


def counts_to_display_y(counts, y_lim, quantile=0.98):
    """
    Map observed counts to the trajectory y-range for the right histogram panel.
    Purely schematic display mapping.
    """
    counts = np.asarray(counts, dtype=float)
    upper = max(np.quantile(counts, quantile), 1.0)
    return (counts / upper) * y_lim


def plot_panel(fig, outer_spec, label, root, sigma, alpha, optim, disp, process,
               count_scale=2.5, n_paths=80, n_steps=100):
    color = PANEL_COLORS[label]

    subgs = outer_spec.subgridspec(1, 2, width_ratios=[5.0, 1.35], wspace=0.04)
    ax_traj = fig.add_subplot(subgs[0, 0])
    ax_hist = fig.add_subplot(subgs[0, 1], sharey=ax_traj)

    # latent simulation
    t, paths = simulate_latent_paths(
        process=process,
        root=root,
        sigma=sigma,
        alpha=alpha,
        optim=optim,
        n_paths=n_paths,
        n_steps=n_steps,
        T=1.0,
    )

    # trajectories
    for i in range(n_paths):
        ax_traj.plot(t, paths[i], color=color, alpha=0.45, lw=0.7)

    # terminal observations
    x_T = paths[:, -1]
    mean_counts = count_scale * softplus(x_T)
    obs_counts = sample_observation(mean_counts, dispersion=disp)

    # y-limits
    y_min, y_max = ax_traj.get_ylim()
    y_abs = max(abs(y_min), abs(y_max), abs(root), abs(optim) if optim is not None else abs(root))
    y_lim = 1.08 * y_abs
    ax_traj.set_ylim(-y_lim, y_lim)

    # right histogram
    y_obs = counts_to_display_y(obs_counts, y_lim=y_lim)
    bins = np.linspace(-y_lim, y_lim, 13)

    ax_hist.hist(
        y_obs,
        bins=bins,
        orientation="horizontal",
        color=color,
        alpha=0.9,
        edgecolor="none",
    )

    # style
    ax_traj.spines["top"].set_visible(False)
    ax_traj.spines["right"].set_visible(False)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    ax_hist.set_xticks([])
    ax_hist.set_yticks([])

    ax_hist.set_ylim(ax_traj.get_ylim())
    ax_hist.grid(False)

    ax_traj.set_title(f"{label}", fontsize=7, fontweight="bold", pad=2)
    ax_traj.text(
        0.02, 0.96, SCENARIO_DESC[label],
        transform=ax_traj.transAxes,
        ha="left", va="top",
        fontsize=5.2, color=color
    )

    return ax_traj, ax_hist


def main():
    os.makedirs("simulation", exist_ok=True)

    fig = plt.figure(figsize=(7.8, 5.6), dpi=300)
    gs = fig.add_gridspec(
        5, 2,
        left=0.05, right=0.985, top=0.97, bottom=0.06,
        hspace=0.42, wspace=0.22
    )

    for spec, scenario in zip(gs, SCENARIOS):
        plot_panel(fig, spec, *scenario)

    for fmt in ["png", "pdf"]:
        out = f"simulation/scenario_panels_softplus_nb.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()