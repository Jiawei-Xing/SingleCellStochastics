#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --job-name=lavous_reconst
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lavous-reconstruct-%j.out
#SBATCH --error=logs/lavous-reconstruct-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

PREFIX="${PREFIX:-3724_NT_All}"
DATA_DIR="${DATA_DIR:-data}"
OUT_BASE="${OUT_BASE:-outputs}"
DIFF_DIR="${DIFF_DIR:-${OUT_BASE}/diff}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_BASE}/reconstruct/highlighted_genes}"

Q_THRESHOLD="${Q_THRESHOLD:-0.1}"
EFFECT_THRESHOLD="${EFFECT_THRESHOLD:-1.0}"
LABELS_PER_SIDE="${LABELS_PER_SIDE:-6}"
TREE_LINEWIDTH="${TREE_LINEWIDTH:-0.6}"
BAR_LINEWIDTH="${BAR_LINEWIDTH:-0.5}"
OVERWRITE="${OVERWRITE:-1}"
GENES="${GENES:-}"

mkdir -p logs "${OUTPUT_DIR}"

cmd=(
    python scripts/reconstruct_highlighted_genes.py
    --diff-results "${DIFF_DIR}/${PREFIX}_diff_Pri-Met_empirical-all_10000.tsv"
    --tree "${DATA_DIR}/${PREFIX}_vine_tree.nwk"
    --regime "${DATA_DIR}/${PREFIX}_vine_regime.csv"
    --q-params "${DIFF_DIR}/${PREFIX}_diff_Pri-Met_h1_q-mean-std_0.tsv"
    --ou-params "${DIFF_DIR}/${PREFIX}_diff_Pri-Met_model-params.tsv"
    --read-counts "${DATA_DIR}/${PREFIX}_readcounts.tsv"
    --library "${DATA_DIR}/${PREFIX}_library.tsv"
    --output-dir "${OUTPUT_DIR}"
    --q-threshold "${Q_THRESHOLD}"
    --effect-threshold "${EFFECT_THRESHOLD}"
    --labels-per-side "${LABELS_PER_SIDE}"
    --tree-linewidth "${TREE_LINEWIDTH}"
    --bar-linewidth "${BAR_LINEWIDTH}"
)

if [[ -n "${GENES}" ]]; then
    # GENES may be space-separated or comma-separated; the Python script accepts both.
    cmd+=(--genes ${GENES})
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
fi

if [[ "${NO_OUTER:-0}" == "1" ]]; then
    cmd+=(--no-outer)
fi

"${cmd[@]}"
