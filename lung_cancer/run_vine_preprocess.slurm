#!/bin/bash
#SBATCH -p cpuq
#SBATCH --job-name=vine_prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vine-preprocess-%j.out
#SBATCH --error=logs/vine-preprocess-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

PREFIX="${PREFIX:-3724_NT_All}"
DEPOSITED_DATA_DIR="${DEPOSITED_DATA_DIR:-/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/real_data/Yang_KP-Tracer}"
DATA_DIR="${DATA_DIR:-data}"
VINE_DIR="${VINE_DIR:-${DATA_DIR}/vine}"
PRIMARY="${PRIMARY:-T1}"
MET_STATES="${MET_STATES:-S1,L1,L2,L3}"
NSAMPLES="${NSAMPLES:-100}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
PREPROCESS_ENV="${PREPROCESS_ENV:-singlecellstochastics}"
VINE_ENV="${VINE_ENV:-vine}"

mkdir -p "${DATA_DIR}" "${VINE_DIR}" logs

conda activate "${PREPROCESS_ENV}"
python scripts/preprocess_kptracer_deposited.py \
    --datadir "${DEPOSITED_DATA_DIR}" \
    --outdir "${VINE_DIR}" \
    --lavous-outdir "${DATA_DIR}" \
    --prefix "${PREFIX}"

conda activate "${VINE_ENV}"
vine "${VINE_DIR}/${PREFIX}_character_matrix_representative.tsv" \
    --format CRISPR \
    --migration "${VINE_DIR}/${PREFIX}_statelabels.csv" \
    --primary "${PRIMARY}" \
    --mean "${VINE_DIR}/${PREFIX}_vine_mean.nwk" \
    --nsamples "${NSAMPLES}" \
    -j "${THREADS}" \
    --logfile "${VINE_DIR}/${PREFIX}_vine.log" \
    --labeled-trees "${VINE_DIR}/${PREFIX}_vine_states.nex" \
    > "${VINE_DIR}/${PREFIX}_vine_samples.nwk"

conda activate "${PREPROCESS_ENV}"
python scripts/prepare_vine_tree_regime.py \
    --nexus "${VINE_DIR}/${PREFIX}_vine_states.nex" \
    --tree-out "${DATA_DIR}/${PREFIX}_vine_tree.nwk" \
    --regime-out "${DATA_DIR}/${PREFIX}_vine_regime.csv" \
    --primary "${PRIMARY}" \
    --met-states "${MET_STATES}"
