#!/bin/bash
#SBATCH -p cpuq
#SBATCH --job-name=vine_e15_nomig
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/vine-e15-nomig-%j.out
#SBATCH --error=logs/vine-e15-nomig-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

VINE_ENV="${VINE_ENV:-vine}"
NSAMPLES="${NSAMPLES:-100}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

PREFIX="${PREFIX:-E15_rep1_clone}"
INPUT_MATRIX="${INPUT_MATRIX:-data/laml/e15_rep1/${PREFIX}_character_matrix.tsv}"
OUT_DIR="${OUT_DIR:-data/vine_no_migration/e15_rep1}"

mkdir -p "${OUT_DIR}" logs

SAMPLES="${OUT_DIR}/${PREFIX}_vine_samples.nwk"
MEAN="${OUT_DIR}/${PREFIX}_vine_mean.nwk"
LOG="${OUT_DIR}/${PREFIX}_vine.log"

if [[ ! -s "${INPUT_MATRIX}" ]]; then
    echo "Missing input character matrix: ${INPUT_MATRIX}" >&2
    exit 1
fi

conda activate "${VINE_ENV}"

# Tree-only inference: no --migration, no --labeled-trees, no --primary.
# Variational posterior over topologies + ultrametric branch lengths from
# the CRISPR mutation matrix alone.
vine "${INPUT_MATRIX}" \
    --format CRISPR \
    --mean "${MEAN}" \
    --nsamples "${NSAMPLES}" \
    -j "${THREADS}" \
    --logfile "${LOG}" \
    > "${SAMPLES}"

echo "VINE finished. mean=${MEAN}, samples=${SAMPLES}"
