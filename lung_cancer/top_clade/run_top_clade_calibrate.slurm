#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_top_cal
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-top-calibrate-%j.out
#SBATCH --error=logs/lavous-top-calibrate-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

PREFIX="${PREFIX:-3724_NT_TopClade}"
SIM_ALL="${SIM_ALL:-10000}"
BATCH="${BATCH:-1000}"
DIFF_DIR="${DIFF_DIR:-top_clade/outputs/diff}"

CHI_TSV="${DIFF_DIR}/${PREFIX}_diff_Pri-Met_chi-squared.tsv"
CACHE="${DIFF_DIR}/${PREFIX}_diff_Pri-Met_null_LRs.pkl"

lavous-calibrate \
    --chi "${CHI_TSV}" \
    --sim_all "${SIM_ALL}" \
    --batch "${BATCH}" \
    --gpd_tail \
    --cache "${CACHE}"
