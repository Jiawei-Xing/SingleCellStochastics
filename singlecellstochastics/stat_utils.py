
import scipy.stats


def calculate_lrt_and_pvalue(
    null_log_lik: float,
    alt_log_lik: float,
    dof: int = 1
) -> (float, float):
    """
    Calculate the likelihood ratio test statistic and p-value.
    
    Args:
        null_neg_log_lik: Negative log-likelihood of the null model (float).
        alt_neg_log_lik: Negative log-likelihood of the alternative model (float).
        dof: Degrees of freedom for the chi-squared test (int).
    
    Returns:
        lrt_statistic: The LRT statistic (float).
        p_value: The p-value from the chi-squared distribution (float).
    """
    lrt_statistic = 2 * (alt_log_lik - null_log_lik)
    p_value = scipy.stats.chi2.sf(lrt_statistic, df=dof)
    return lrt_statistic, p_value