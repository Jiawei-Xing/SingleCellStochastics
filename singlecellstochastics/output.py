"""Result-table helpers for differential-expression likelihood-ratio tests."""

import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


def save_result(
    batch_start, batch_size, batch_genes,
    h0_params, h1_params, h0_loss, h1_loss, results,
):
    """Append per-gene LRT results from one batch into ``results``.

    ``h0_params`` and ``h1_params`` are ``(batch_size, 1, param_dim)`` arrays
    laid out as ``[log_r, log_alpha, sigma, theta_0, theta_1, ...]``. ``h0_loss``
    and ``h1_loss`` are ``(batch_size, 1)`` ELBO losses.
    """
    n_regimes = h0_params.shape[2] - 3
    delta_nll = h0_loss - h1_loss
    lrt_stat = 2 * delta_nll
    p_value = 1 - chi2.cdf(lrt_stat.flatten(), n_regimes - 1)

    for i in range(batch_size):
        if i >= h0_params.shape[0]:
            break
        h0_r = np.exp(h0_params[i, 0, 0])
        h0_alpha = np.exp(h0_params[i, 0, 1])
        h0_sigma = np.abs(h0_params[i, 0, 2])
        h0_theta = h0_params[i, 0, 3]

        h1_r = np.exp(h1_params[i, 0, 0])
        h1_alpha = np.exp(h1_params[i, 0, 1])
        h1_sigma = np.abs(h1_params[i, 0, 2])
        h1_theta = h1_params[i, 0, 3:]

        result = (
            [batch_start + i, batch_genes[i], h0_r, h0_alpha, h0_sigma, h0_theta]
            + [h1_r, h1_alpha, h1_sigma] + h1_theta.tolist()
            + [h0_loss[i, 0], h1_loss[i, 0], delta_nll[i, 0], lrt_stat[i, 0], p_value[i]]
        )
        results[batch_start + i] = result
    
    return results


def result_columns(regimes):
    """Column names for rows produced by `save_result`."""
    return (
        ["ID", "gene", "h0_r", "h0_alpha", "h0_sigma", "h0_theta",
         "h1_r", "h1_alpha", "h1_sigma"]
        + [f"h1_theta_{regime}" for regime in regimes]
        + ["h0", "h1", "delta_nll", "lrt", "p"]
    )


def output_results(results, output_file, regimes):
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
        f.write("\t".join(result_columns(regimes) + ["q", "signif"]) + "\n")
        for i in range(len(results)):
            output = "\t".join(list(map(str, results[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()


def output_model_params(results, output_file, regimes):
    """
    Write fitted model parameters in long form.

    This table is easier to consume for calibration, plotting, and history
    reconstruction than the compact hypothesis-test result table.
    """
    n_regimes = len(regimes)
    header = ["ID", "gene", "hypothesis", "regime", "r", "alpha", "sigma", "theta"]

    rows = []
    for result in sorted(results.values(), key=lambda x: int(x[0])):
        gene_id, gene = result[0], result[1]
        h0_r, h0_alpha, h0_sigma, h0_theta = result[2:6]
        h1_r, h1_alpha, h1_sigma = result[6:9]
        h1_theta = result[9:9 + n_regimes]

        rows.append([gene_id, gene, "h0", "shared", h0_r, h0_alpha, h0_sigma, h0_theta])
        for regime, theta in zip(regimes, h1_theta):
            rows.append([gene_id, gene, "h1", regime, h1_r, h1_alpha, h1_sigma, theta])

    with open(output_file, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
