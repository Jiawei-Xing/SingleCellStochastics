import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


def save_result(batch_start, batch_size, batch_genes, \
    h0_params, h1_params, h0_loss, h1_loss, results, approx):
    """
    Save result of likelihood ratio test for each gene in batch.
    params: (batch_size, 1, param_dim) numpy array, (r, alpha, sigma2, theta, ...) for h0 and h1
    loss: (batch_size, 1) numpy array
    """
    n_regimes = h0_params.shape[2] - 3
    lr = h0_loss - h1_loss
    p_value = 1 - chi2.cdf(2 * lr.flatten(), n_regimes - 1)

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
            + [h0_loss[i, 0], h1_loss[i, 0], lr[i, 0], p_value[i]]
        )
        results[batch_start + i] = result
    
    return results


def output_results(results, output_file, regimes):
    """
    Output saved results of likelihood ratio test for each gene to a tsv file.
    """
    # FDR by Benjamini-Hochberg procedure
    results = sorted(list(results.values()), key=lambda x: x[-1])
    p_values = [r[-1] for r in results]
    signif, q_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[:2]

    # output results
    with open(output_file, "w") as f:
        f.write(
            f"ID\tgene\th0_r\th0_alpha\th0_sigma\th0_theta\th1_r\th1_alpha\th1_sigma\t"
            + "\t".join(["h1_theta" + r for r in regimes])
            + f"\th0\th1\tLR\tp\tq\tsignif\n"
        )
        for i in range(len(results)):
            output = "\t".join(list(map(str, results[i])))
            f.write(f"{output}\t{q_values[i]}\t{signif[i]}\n")
            f.flush()
