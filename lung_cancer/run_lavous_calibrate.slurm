#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_cal
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-calibrate-%j.out
#SBATCH --error=logs/lavous-calibrate-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=xing@cshl.edu

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

PREFIX="${PREFIX:-3724_NT_All}"
SIM_ALL="${SIM_ALL:-20000}"
BATCH="${BATCH:-1000}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_BASE="${OUT_BASE:-outputs}"
DIFF_DIR="${OUT_BASE}/diff"
CHI_TSV="${DIFF_DIR}/${PREFIX}_diff_Pri-Met_chi-squared.tsv"
CACHE="${DIFF_DIR}/${PREFIX}_diff_Pri-Met_null_LRs.pkl"

lavous-calibrate \
    --chi "${CHI_TSV}" \
    --sim_all "${SIM_ALL}" \
    --batch "${BATCH}" \
    --gpd_tail \
    --cache "${CACHE}"
