#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_e15_fp_vine_pro_diff
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-e15-rep1-fp-vine-pro-diff-%j.out
#SBATCH --error=logs/lavous-e15-rep1-fp-vine-pro-diff-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

# Heritability list comes from the 920-leaf VINE no-migration tree
# (outputs/heritability/E15_rep1_clone_heritability_vine.tsv). Diff and
# calibration run on the 162-leaf VINE fp tree labeled with Sankoff parsimony.
# This variant tests H0: single-theta pro vs H1: pro+diff (two thetas),
# i.e. whether diff cells diverge from the pro reference at all.
FP_PREFIX="${FP_PREFIX:-E15_rep1_fp_lineage_homogeneous_clone}"
FDR="${FDR:-0.05}"
SIM_ALL="${SIM_ALL:-2000}"
SIM_EACH="${SIM_EACH:-2000}"
DIFF_BATCH="${DIFF_BATCH:-1000}"
CAL_BATCH="${CAL_BATCH:-1000}"
EACH_BATCH="${EACH_BATCH:-1}"

FP_EXPR_DIR="${FP_EXPR_DIR:-data/expression/e15_rep1_fp_lineage}"
FP_VINE_DIR="${FP_VINE_DIR:-data/vine_no_migration/e15_rep1_fp_lineage}"
HERITABILITY="${HERITABILITY:-outputs/heritability/E15_rep1_clone_heritability_vine.tsv}"
FP_OUT_BASE="${FP_OUT_BASE:-outputs/e15_rep1_fp_lineage_vine}"
DIFF_DIR="${FP_OUT_BASE}/diff"
CAL_DIR="${FP_OUT_BASE}/calibration"
HERIT_DIR="${FP_OUT_BASE}/heritability"

FP_TREE="${FP_VINE_DIR}/${FP_PREFIX}_vine_parsimony_named_tree.nwk"
ALT_REGIME="${FP_VINE_DIR}/${FP_PREFIX}_vine_lavous_null_pro_diff.csv"
FP_EXPR="${FP_EXPR_DIR}/${FP_PREFIX}_readcounts.tsv"
FP_LIBRARY="${FP_EXPR_DIR}/${FP_PREFIX}_library.tsv"

HERIT_EXPR="${HERIT_DIR}/${FP_PREFIX}_readcounts_heritability_q${FDR}.tsv"
DIFF_PREFIX="${FP_PREFIX}_vine_diff_pro_diff"
CHI_TSV="${DIFF_DIR}/${DIFF_PREFIX}_chi-squared.tsv"
CACHE="${CAL_DIR}/${DIFF_PREFIX}_null_LRs.pkl"
CACHE_EACH="${CAL_DIR}/${DIFF_PREFIX}_sim_each${SIM_EACH}_null_LRs.pkl"

mkdir -p "${HERIT_DIR}" "${DIFF_DIR}" "${CAL_DIR}" logs

for f in "${FP_TREE}" "${ALT_REGIME}" "${FP_EXPR}" "${FP_LIBRARY}"; do
    if [[ ! -s "${f}" ]]; then
        echo "Missing required input: ${f}" >&2
        exit 1
    fi
done
# HERITABILITY is only needed if regenerating HERIT_EXPR (block below).

# Reuse the heritability-filtered fp expr matrix produced by the pro_da_glu run
# (identical filter: q < FDR on the same heritability TSV intersected with the
# same fp expr columns). Regenerate by uncommenting if HERIT_EXPR is missing.
if [[ ! -s "${HERIT_EXPR}" ]]; then
    echo "Missing ${HERIT_EXPR}; regenerate via the pro_da_glu slurm or uncomment the block below." >&2
    exit 1
fi
# export FP_EXPR HERITABILITY HERIT_EXPR FDR
# python - <<'PY'
# import os
# import pandas as pd
#
# fdr = float(os.environ["FDR"])
# heritability = pd.read_csv(os.environ["HERITABILITY"], sep="\t")
# selected = heritability.loc[heritability["q"] < fdr, "gene"].astype(str).tolist()
# if not selected:
#     raise SystemExit(f"No heritable genes passed q < {fdr}; skipping lavous-diff and calibration")
#
# expr = pd.read_csv(os.environ["FP_EXPR"], sep="\t", index_col=0)
# in_fp = [gene for gene in selected if gene in expr.columns]
# dropped = len(selected) - len(in_fp)
# if dropped:
#     print(f"Dropping {dropped} heritable genes absent from fp expr (sparsity filter)")
# if not in_fp:
#     raise SystemExit("No heritable genes remain after intersection with fp expr matrix")
# expr[in_fp].to_csv(os.environ["HERIT_EXPR"], sep="\t")
# print(f"Selected {len(in_fp)} heritable genes at q < {fdr} (kept in fp expr); wrote {os.environ['HERIT_EXPR']}")
# PY

# H0: single-theta (all pro). Omit --null_regime so lavous-diff uses one theta.
# H1: pro vs diff regime split given by ALT_REGIME.
lavous-diff \
    --tree "${FP_TREE}" \
    --expression "${HERIT_EXPR}" \
    --regime "${ALT_REGIME}" \
    --library "${FP_LIBRARY}" \
    --null pro \
    --outdir "${DIFF_DIR}" \
    --prefix "${DIFF_PREFIX}" \
    --batch "${DIFF_BATCH}" \
    --resume

lavous-calibrate \
    --chi "${CHI_TSV}" \
    --outdir "${CAL_DIR}" \
    --prefix "${DIFF_PREFIX}" \
    --sim_all "${SIM_ALL}" \
    --batch "${CAL_BATCH}" \
    --gpd_tail \
    --cache "${CACHE}"

lavous-calibrate \
    --chi "${CHI_TSV}" \
    --outdir "${CAL_DIR}" \
    --prefix "${DIFF_PREFIX}_sim_each${SIM_EACH}" \
    --sim_each "${SIM_EACH}" \
    --batch "${EACH_BATCH}" \
    --cache "${CACHE_EACH}"
