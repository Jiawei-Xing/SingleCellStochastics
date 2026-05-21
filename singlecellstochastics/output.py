"""Result-table helpers for differential-expression likelihood-ratio tests."""

import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


def _legacy_h0(h0_regimes):
    return h0_regimes is None


def _h0_theta_columns(h0_regimes):
    if _legacy_h0(h0_regimes):
        return ["h0_theta"]
    return [f"h0_theta_{regime}" for regime in h0_regimes]


def save_result(
    batch_start,
    batch_size,
    batch_genes,
    h0_params,
    h1_params,
    h0_loss,
    h1_loss,
    results,
    h0_regimes=None,
    h1_regimes=None,
    dof=None,
):
    """Append per-gene LRT results from one batch into ``results``.

    ``h0_params`` and ``h1_params`` are ``(batch_size, 1, param_dim)`` arrays
    laid out as ``[log_r, log_alpha, sigma, theta_0, theta_1, ...]``. In the
    legacy test, H0 has one reported theta and ``h0_regimes`` is ``None``. For
    a multi-theta null, ``h0_regimes`` names the H0 theta columns.
    """
    h1_n_regimes = len(h1_regimes) if h1_regimes is not None else h1_params.shape[2] - 3
    h0_n_regimes = 1 if _legacy_h0(h0_regimes) else len(h0_regimes)
    if dof is None:
        dof = h1_n_regimes - h0_n_regimes
    if dof <= 0:
        raise ValueError(f"LRT degrees of freedom must be positive, got {dof}.")

    lrt_stat = 2 * (h0_loss - h1_loss)
    p_value = 1 - chi2.cdf(lrt_stat.flatten(), dof)

    for i in range(batch_size):
        if i >= h0_params.shape[0]:
            break
        h0_r = np.exp(h0_params[i, 0, 0])
        h0_alpha = np.exp(h0_params[i, 0, 1])
        h0_sigma = np.abs(h0_params[i, 0, 2])
        h0_theta = h0_params[i, 0, 3 : 3 + h0_n_regimes]

        h1_r = np.exp(h1_params[i, 0, 0])
        h1_alpha = np.exp(h1_params[i, 0, 1])
        h1_sigma = np.abs(h1_params[i, 0, 2])
        h1_theta = h1_params[i, 0, 3 : 3 + h1_n_regimes]

        result = (
            [batch_start + i, batch_genes[i], h0_r, h0_alpha, h0_sigma]
            + h0_theta.tolist()
            + [h1_r, h1_alpha, h1_sigma]
            + h1_theta.tolist()
            + [h0_loss[i, 0], h1_loss[i, 0], lrt_stat[i, 0], p_value[i]]
        )
        results[batch_start + i] = result

    return results


def result_columns(h1_regimes, h0_regimes=None):
    """Column names for rows produced by `save_result`."""
    return (
        ["ID", "gene", "h0_r", "h0_alpha", "h0_sigma"]
        + _h0_theta_columns(h0_regimes)
        + ["h1_r", "h1_alpha", "h1_sigma"]
        + [f"h1_theta_{regime}" for regime in h1_regimes]
        + ["h0", "h1", "lrt", "p"]
    )


def output_results(results, output_file, h1_regimes, h0_regimes=None):
    """
    Output saved results of likelihood ratio test for each gene to a tsv file.
    """
    # FDR by Benjamini-Hochberg procedure
    # Sort by p-value ascending, pushing NaN p (failed optimizations) to the end
    # so q-values stay monotone with p and NaN rows are grouped at the bottom.
    results = sorted(
        list(results.values()),
        key=lambda x: (np.isnan(x[-1]), x[-1]),
    )
    p_values = np.array([r[-1] for r in results], dtype=float)
    q_values = np.full(p_values.shape, np.nan)
    signif = np.zeros(p_values.shape, dtype=bool)
    valid = ~np.isnan(p_values)
    if valid.any():
        signif[valid], q_values[valid] = multipletests(
            p_values[valid], alpha=0.05, method="fdr_bh"
        )[:2]

    # output results
    with open(output_file, "w") as f:
        f.write(
            "\t".join(result_columns(h1_regimes, h0_regimes) + ["q", "signif"]) + "\n"
        )
        for i in range(len(results)):
            output = "\t".join(list(map(str, results[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()


def output_model_params(results, output_file, h1_regimes, h0_regimes=None):
    """
    Write fitted model parameters in long form.

    This table is easier to consume for calibration, plotting, and history
    reconstruction than the compact hypothesis-test result table.
    """
    h0_names = ["shared"] if _legacy_h0(h0_regimes) else list(h0_regimes)
    h1_names = list(h1_regimes)
    h0_n_regimes = len(h0_names)
    h1_n_regimes = len(h1_names)
    header = ["ID", "gene", "hypothesis", "regime", "r", "alpha", "sigma", "theta"]

    rows = []
    for result in sorted(results.values(), key=lambda x: int(x[0])):
        gene_id, gene = result[0], result[1]
        h0_r, h0_alpha, h0_sigma = result[2:5]
        h0_theta_start = 5
        h0_theta_end = h0_theta_start + h0_n_regimes
        h0_theta = result[h0_theta_start:h0_theta_end]
        h1_param_start = h0_theta_end
        h1_r, h1_alpha, h1_sigma = result[h1_param_start : h1_param_start + 3]
        h1_theta_start = h1_param_start + 3
        h1_theta = result[h1_theta_start : h1_theta_start + h1_n_regimes]

        for regime, theta in zip(h0_names, h0_theta):
            rows.append([gene_id, gene, "h0", regime, h0_r, h0_alpha, h0_sigma, theta])
        for regime, theta in zip(h1_names, h1_theta):
            rows.append([gene_id, gene, "h1", regime, h1_r, h1_alpha, h1_sigma, theta])

    with open(output_file, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
