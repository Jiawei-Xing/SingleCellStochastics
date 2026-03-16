"""
2x4 schematic panels for latent process + softplus + NB/Poisson observation.

Layout:
    Row 1: A | HIJ-bars-only | B | C
    Row 2: D | E | F | G

Design:
- First-row second panel contains only count bars for H / I / J
- No third row
- No truncation below 0 for any trajectory
- Only C uses a darker color for the metastatic group
- E (BM) is restored in the second row

Usage:
    python plot_schematic.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

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
    "lines.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

np.random.seed(42)

# ── Labels and colors ────────────────────────────────────────────────
SCENARIO_LABELS = {
    "A": "Baseline",
    "B": "High θ",
    "C": "Different θ",
    "D": "High σ",
    "E": "No α (BM)",
    "F": "Weak α",
    "G": "Strong α",
    "HIJ": "Observation model",
}

COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#7f7f7f",
    "F": "#9467bd",
    "G": "#8c564b",
    "H": "#e377c2",
    "I": "#17becf",
    "J": "#bcbd22",
}

# label, root, sigma, alpha, theta_pri, theta_met, disp, process
SCENARIO_PARAMS = {
    "A": ("A", 1.0, 3.0, 1.0, 1.0, 1.0, 5.0, "OU"),
    "B": ("B", 3.0, 3.0, 1.0, 3.0, 3.0, 5.0, "OU"),
    "C": ("C", 1.0, 3.0, 1.0, 1.0, 3.0, 5.0, "OU"),
    "D": ("D", 1.0, 5.0, 1.0, 1.0, 1.0, 5.0, "OU"),
    "E": ("E", 1.0, 3.0, None, None, None, 5.0, "BM"),
    "F": ("F", 1.0, 3.0, 0.3, 1.0, 1.0, 5.0, "OU"),
    "G": ("G", 1.0, 3.0, 5.0, 1.0, 1.0, 5.0, "OU"),
}

DISPLAY_YMIN = 0.0
DISPLAY_YMAX = 10.0
BIN_WIDTH = 0.5

# A and HIJ share the same latent simulation source
SHARED_LATENT_GROUP = {"A"}


# ── Helpers ──────────────────────────────────────────────────────────
def darken(hex_color, factor=0.58):
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sample_observation(mean, dispersion):
    mean = np.maximum(mean, 1e-8)
    if dispersion == 0:
        return np.random.poisson(mean)
    n = dispersion
    p = n / (n + mean)
    return np.random.negative_binomial(n, p)


def simulate_group_paths(process, root, sigma, alpha, theta, n_paths=35, n_steps=70, T=1.0):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = root

    for i in range(n_steps):
        noise = sigma * np.sqrt(dt) * np.random.randn(n_paths)
        if process == "BM":
            drift = 0.0
        elif process == "OU":
            drift = alpha * (theta - paths[:, i]) * dt
        else:
            raise ValueError(f"Unknown process: {process}")
        paths[:, i + 1] = paths[:, i] + drift + noise

    return t, paths


def counts_to_display_y(counts, y_max=DISPLAY_YMAX, quantile=0.98):
    counts = np.asarray(counts, dtype=float)
    upper = max(np.quantile(counts, quantile), 1.0)
    return (counts / upper) * y_max


def build_latent_cache(n_paths_per_group=35, n_steps=70):
    cache = {}
    t_pri, pri_paths = simulate_group_paths(
        process="OU", root=1.0, sigma=3.0, alpha=1.0, theta=1.0,
        n_paths=n_paths_per_group, n_steps=n_steps
    )
    _, met_paths = simulate_group_paths(
        process="OU", root=1.0, sigma=3.0, alpha=1.0, theta=1.0,
        n_paths=n_paths_per_group, n_steps=n_steps
    )
    cache["shared_A"] = (t_pri, pri_paths, met_paths)
    return cache


def make_panel_axes(fig, outer_spec):
    subgs = outer_spec.subgridspec(1, 2, width_ratios=[1.22, 0.28], wspace=0.01)
    ax_traj = fig.add_subplot(subgs[0, 0])
    ax_hist = fig.add_subplot(subgs[0, 1], sharey=ax_traj)
    return ax_traj, ax_hist


def style_standard_axes(ax_traj, ax_hist, title_text):
    ax_traj.spines["top"].set_visible(False)
    ax_traj.spines["right"].set_visible(False)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    ax_traj.set_xlim(0, 1.0)
    ax_traj.set_ylim(DISPLAY_YMIN, DISPLAY_YMAX)
    ax_hist.set_ylim(DISPLAY_YMIN, DISPLAY_YMAX)

    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    ax_hist.set_xticks([])
    ax_hist.set_yticks([])
    ax_hist.grid(False)

    ax_traj.text(
        0.02, 0.96, title_text,
        transform=ax_traj.transAxes,
        ha="left", va="top",
        fontsize=7.5, color="black"
    )


def plot_paths(ax, t, pri_paths, met_paths, pri_color, met_color, highlight_c=False):
    common_alpha = 0.28
    common_lw = 0.85

    for i in range(pri_paths.shape[0]):
        ax.plot(
            t, pri_paths[i],
            color=pri_color,
            alpha=common_alpha,
            lw=common_lw,
            clip_on=False,
            zorder=2
        )

    for i in range(met_paths.shape[0]):
        ax.plot(
            t, met_paths[i],
            color=met_color,
            alpha=0.42 if highlight_c else common_alpha,
            lw=0.95 if highlight_c else common_lw,
            clip_on=False,
            zorder=3 if highlight_c else 2
        )


def plot_standard_panel(fig, outer_spec, label, root, sigma, alpha, theta_pri, theta_met, disp, process,
                        latent_cache, count_scale=2.5, n_paths_per_group=35, n_steps=70):
    ax_traj, ax_hist = make_panel_axes(fig, outer_spec)
    title_text = SCENARIO_LABELS[label]
    hist_base_color = COLORS[label]
    traj_base_color = COLORS[label]

    if label in SHARED_LATENT_GROUP:
        t, pri_paths, met_paths = latent_cache["shared_A"]
    else:
        if process == "BM":
            t, pri_paths = simulate_group_paths(
                process="BM", root=root, sigma=sigma, alpha=None, theta=None,
                n_paths=n_paths_per_group, n_steps=n_steps
            )
            _, met_paths = simulate_group_paths(
                process="BM", root=root, sigma=sigma, alpha=None, theta=None,
                n_paths=n_paths_per_group, n_steps=n_steps
            )
        else:
            t, pri_paths = simulate_group_paths(
                process="OU", root=root, sigma=sigma, alpha=alpha, theta=theta_pri,
                n_paths=n_paths_per_group, n_steps=n_steps
            )
            _, met_paths = simulate_group_paths(
                process="OU", root=root, sigma=sigma, alpha=alpha, theta=theta_met,
                n_paths=n_paths_per_group, n_steps=n_steps
            )

    pri_traj_color = traj_base_color
    met_traj_color = darken(traj_base_color) if label == "C" else traj_base_color

    plot_paths(
        ax_traj, t, pri_paths, met_paths,
        pri_traj_color, met_traj_color,
        highlight_c=(label == "C")
    )

    xT_pri = pri_paths[:, -1]
    xT_met = met_paths[:, -1]

    mean_pri = count_scale * softplus(xT_pri)
    mean_met = count_scale * softplus(xT_met)

    obs_pri = sample_observation(mean_pri, dispersion=disp)
    obs_met = sample_observation(mean_met, dispersion=disp)

    y_obs_pri = counts_to_display_y(obs_pri)
    y_obs_met = counts_to_display_y(obs_met)

    bins = np.arange(DISPLAY_YMIN, DISPLAY_YMAX + BIN_WIDTH, BIN_WIDTH)

    if label == "C":
        ax_hist.hist(
            y_obs_pri,
            bins=bins,
            orientation="horizontal",
            color=hist_base_color,
            alpha=0.55,
            edgecolor="none",
            zorder=1
        )
        ax_hist.hist(
            y_obs_met,
            bins=bins,
            orientation="horizontal",
            color=darken(hist_base_color),
            alpha=0.95,
            edgecolor="none",
            zorder=2
        )
    else:
        ax_hist.hist(
            np.concatenate([y_obs_pri, y_obs_met]),
            bins=bins,
            orientation="horizontal",
            color=hist_base_color,
            alpha=0.9,
            edgecolor="none",
            zorder=2
        )

    style_standard_axes(ax_traj, ax_hist, title_text)


def plot_hij_bars_only_panel(fig, outer_spec, latent_cache, count_scale=2.5):
    """
    Panel with no trajectory. Only H / I / J count bars, shown separately.
    """
    subgs = outer_spec.subgridspec(1, 3, wspace=0.22)
    axes = [fig.add_subplot(subgs[0, i]) for i in range(3)]

    _, pri_paths, met_paths = latent_cache["shared_A"]
    xT = np.concatenate([pri_paths[:, -1], met_paths[:, -1]])
    mean_counts = count_scale * softplus(xT)

    obs_dict = {
        "H": sample_observation(mean_counts, dispersion=0.5),
        "I": sample_observation(mean_counts, dispersion=50.0),
        "J": sample_observation(mean_counts, dispersion=0.0),
    }

    bins = np.arange(DISPLAY_YMIN, DISPLAY_YMAX + BIN_WIDTH, BIN_WIDTH)

    for ax, key in zip(axes, ["H", "I", "J"]):
        y_obs = counts_to_display_y(obs_dict[key])

        ax.hist(
            y_obs,
            bins=bins,
            orientation="horizontal",
            color=COLORS[key],
            alpha=0.9,
            edgecolor="none"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(DISPLAY_YMIN, DISPLAY_YMAX)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel(
            {"H": "Strong r⁻¹", "I": "Weak r⁻¹", "J": "No r\n(Poisson)"}[key],
            fontsize=7.5, labelpad=3
        )

    # Put title on the first mini-axis
    axes[0].text(
        0.00, 0.96, "",
        transform=axes[0].transAxes,
        ha="left", va="top",
        fontsize=7.5, color="black"
    )


def main():

    latent_cache = build_latent_cache(n_paths_per_group=35, n_steps=70)

    layout = [
        ["A", "HIJ", "B", "C"],
        ["D", "E", "F", "G"],
    ]

    fig = plt.figure(figsize=(8, 4), dpi=300)
    outer = fig.add_gridspec(
        2, 4,
        left=0.035, right=0.995, top=0.992, bottom=0.035,
        hspace=0.5, wspace=0.16
    )

    for i in range(2):
        for j in range(4):
            label = layout[i][j]
            if label == "HIJ":
                plot_hij_bars_only_panel(fig, outer[i, j], latent_cache=latent_cache)
            else:
                plot_standard_panel(
                    fig,
                    outer[i, j],
                    *SCENARIO_PARAMS[label],
                    latent_cache=latent_cache
                )

    for fmt in ["png"]:
        out = f"scenario_panels_2x4.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()