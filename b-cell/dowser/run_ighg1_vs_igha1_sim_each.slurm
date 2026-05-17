#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_bcell_ighg1_igha1_each
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-bcell-ighg1-igha1-sim-each-%j.out
#SBATCH --error=logs/lavous-bcell-ighg1-igha1-sim-each-%j.err

set -euo pipefail

DEFAULT_WORK_ROOT="/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/b-cell/dowser"
WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_WORK_ROOT}}}"
if [[ ! -d "${WORK_ROOT}/ighg1_vs_igha1" && -d "${DEFAULT_WORK_ROOT}/ighg1_vs_igha1" ]]; then
    WORK_ROOT="${DEFAULT_WORK_ROOT}"
fi
cd "${WORK_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

WORK_DIR="${WORK_DIR:-ighg1_vs_igha1}"
GENE_SET="${GENE_SET:-full}"
SIMS="${SIMS:-2000}"
BATCH="${BATCH:-1}"
CALIBRATE_EXTRA_ARGS="${CALIBRATE_EXTRA_ARGS:-}"

DIFF_DIR="${WORK_DIR}/joint_outputs/diff/${GENE_SET}"
PREFIX="${PREFIX:-ighg1_vs_igha1_joint.${GENE_SET}}"
CHI="${DIFF_DIR}/${PREFIX}_chi-squared.tsv"
CACHE="${DIFF_DIR}/${PREFIX}.sim_each${SIMS}.null_lrs.pkl"

if [[ ! -s "${CHI}" ]]; then
    echo "Missing chi-squared input: ${CHI}" >&2
    exit 2
fi

echo "[$(date)] per-gene empirical calibration ${CHI} with sim_each=${SIMS}, batch=${BATCH}"
lavous-calibrate \
    --chi "${CHI}" \
    --sim_each "${SIMS}" \
    --batch "${BATCH}" \
    --cache "${CACHE}" \
    ${CALIBRATE_EXTRA_ARGS}

echo "[$(date)] sim_each calibration done"
