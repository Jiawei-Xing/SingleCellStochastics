import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import argparse
import sys
import os


def compare_results():
    """
    Compare results of stochastic simulation and generate ROC and Precision-Recall curves.
    
    Parameters:
    -----------
    oup_neg : str
        Path to OUP result for negative simulation
    oup_pos : str
        Path to OUP result for positive simulation
    egx_neg : str
        Path to EGX result for negative simulation
    egx_pos : str
        Path to EGX result for positive simulation
    dea_neg : str
        Path to DEA result for negative simulation
    dea_pos : str
        Path to DEA result for positive simulation
    output : str
        Output file prefix
    """

    parser = argparse.ArgumentParser(description="Compare results of stochastic simulation")
    parser.add_argument("--OUP_neg", type=str, required=True, help="OUP result for negative simulation")
    parser.add_argument("--OUP_pos", type=str, required=True, help="OUP result for positive simulation")
    parser.add_argument("--EGX_neg", type=str, required=True, help="EGX result for negative simulation")
    parser.add_argument("--EGX_pos", type=str, required=True, help="EGX result for positive simulation")
    parser.add_argument("--DEA_neg", type=str, required=True, help="DEA result for negative simulation")
    parser.add_argument("--DEA_pos", type=str, required=True, help="DEA result for positive simulation")
    parser.add_argument("--output", type=str, required=True, help="Output dir")
    parser.add_argument("--category", type=str, required=True, help="Simulation category")
    parser.add_argument("--label", type=str, required=True, help="Simulation label")
    args = parser.parse_args()

    # load results from files
    df_oup_neg = pd.read_csv(args.OUP_neg, sep="\t")
    df_oup_pos = pd.read_csv(args.OUP_pos, sep="\t")
    df_egx_neg = pd.read_csv(args.EGX_neg, sep=",")
    df_egx_pos = pd.read_csv(args.EGX_pos, sep=",")
    df_dea_neg = pd.read_csv(args.DEA_neg, sep="\t")
    df_dea_pos = pd.read_csv(args.DEA_pos, sep="\t")

    output = args.output
    category = args.category
    label = args.label

    # ground truth
    n_neg = len(df_oup_neg)
    n_pos = len(df_oup_pos)
    truth = np.array([False]*n_neg + [True]*n_pos)

    # ROC and AUC
    result_oup_neg = np.sign(df_oup_neg["h1_theta1"].values - df_oup_neg["h1_theta0"].values) * (-np.log10(df_oup_neg["p"].values))
    result_oup_pos = np.sign(df_oup_pos["h1_theta1"].values - df_oup_pos["h1_theta0"].values) * (-np.log10(df_oup_pos["p"].values))
    p_oup = np.concatenate((result_oup_neg, result_oup_pos))
    fpr1, tpr1, _ = roc_curve(truth, p_oup)
    roc_auc1 = auc(fpr1, tpr1)
    
    result_dea_neg = np.sign(df_dea_neg["log2FC"].values) * (-np.log10(df_dea_neg["p_value"].values))
    result_dea_pos = np.sign(df_dea_pos["log2FC"].values) * (-np.log10(df_dea_pos["p_value"].values))
    p_dea = np.concatenate((result_dea_neg, result_dea_pos))
    fpr2, tpr2, _ = roc_curve(truth, p_dea)
    roc_auc2 = auc(fpr2, tpr2)

    result_egx_neg = np.sign(df_egx_neg["ou2_theta"].iloc[1::2].values - df_egx_neg["ou2_theta"].iloc[::2].values) * (-np.log10(df_egx_neg["ou2_vs_ou1_pvalue"].iloc[::2].values))
    result_egx_pos = np.sign(df_egx_pos["ou2_theta"].iloc[1::2].values - df_egx_pos["ou2_theta"].iloc[::2].values) * (-np.log10(df_egx_pos["ou2_vs_ou1_pvalue"].iloc[::2].values))
    p_egx = np.concatenate((result_egx_neg, result_egx_pos))
    fpr3, tpr3, _ = roc_curve(truth, p_egx)
    roc_auc3 = auc(fpr3, tpr3)

    # plot ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr2, tpr2, color='gray', lw=2, label=f'DEA (AUC = {roc_auc2:.3f})')
    plt.plot(fpr3, tpr3, color='blue', lw=2, label=f'EGX (AUC = {roc_auc3:.3f})')
    plt.plot(fpr1, tpr1, color='red', lw=2, label=f'OUP (AUC = {roc_auc1:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(output, f'{category}_{label}_ROC.png'))
    plt.close()

    # OUP: Compute Precision-Recall curve and AUC
    precision1, recall1, _ = precision_recall_curve(truth, p_oup)
    prc_auc1 = auc(recall1, precision1)

    # DEA: Compute Precision-Recall curve and AUC
    precision2, recall2, _ = precision_recall_curve(truth, p_dea)
    prc_auc2 = auc(recall2, precision2)

    # EGX: Compute Precision-Recall curve and AUC
    precision3, recall3, _ = precision_recall_curve(truth, p_egx)
    prc_auc3 = auc(recall3, precision3)

    # Plot Precision-Recall curve
    plt.figure(figsize=(6, 6))
    plt.plot(recall2, precision2, color='gray', lw=2, label=f'DEA (AUPRC = {prc_auc2:.3f})')
    plt.plot(recall3, precision3, color='blue', lw=2, label=f'EGX (AUPRC = {prc_auc3:.3f})')
    plt.plot(recall1, precision1, color='red', lw=2, label=f'OUP (AUPRC = {prc_auc1:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(output, f'{category}_{label}_PRC.png'))
    plt.close()

    # f1 scores
    results_oup = np.concatenate((df_oup_neg["signif"].values, df_oup_pos["signif"].values))
    precision_oup = precision_score(truth, results_oup, zero_division=0)
    recall_oup = recall_score(truth, results_oup)
    f1_oup = f1_score(truth, results_oup)

    results_dea = np.concatenate((df_dea_neg["signif"].values, df_dea_pos["signif"].values))
    precision_dea = precision_score(truth, results_dea, zero_division=0)
    recall_dea = recall_score(truth, results_dea)
    f1_dea = f1_score(truth, results_dea)

    results_egx = np.concatenate((df_egx_neg["adaptive"].iloc[::2].values, df_egx_pos["adaptive"].iloc[::2].values))
    precision_egx = precision_score(truth, results_egx, zero_division=0)
    recall_egx = recall_score(truth, results_egx)
    f1_egx = f1_score(truth, results_egx)

    # check if file is newly created
    outfile = 'all_results.tsv'
    flag = False
    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
         flag = True

    with open(outfile, 'a') as f:
        if flag: # first line in file
            f.write("category\tlabel\tmethod\troc\tprc\tprecision\trecall\tf1\n")
        f.write(f"{category}\t{label}\tOUP\t{roc_auc1}\t{prc_auc1}\t{precision_oup}\t{recall_oup}\t{f1_oup}\n")
        f.write(f"{category}\t{label}\tDEA\t{roc_auc2}\t{prc_auc2}\t{precision_dea}\t{recall_dea}\t{f1_dea}\n")
        f.write(f"{category}\t{label}EGX\t{roc_auc3}\t{prc_auc3}\t{precision_egx}\t{recall_egx}\t{f1_egx}\n")


if __name__ == "__main__":
    compare_results()
