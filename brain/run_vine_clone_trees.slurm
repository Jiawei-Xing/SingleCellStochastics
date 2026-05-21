#!/bin/bash
#SBATCH -p cpuq
#SBATCH --job-name=crest_vine
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/crest-vine-%j.out
#SBATCH --error=logs/crest-vine-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

DATASETS="${DATASETS:-aggregate}"
UNIT="${UNIT:-clone}"
MIN_CLONE_SIZE="${MIN_CLONE_SIZE:-1}"
MIN_DOMINANT_FRACTION="${MIN_DOMINANT_FRACTION:-0}"
NSAMPLES="${NSAMPLES:-100}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
PREPROCESS_ENV="${PREPROCESS_ENV:-singlecellstochastics}"
VINE_ENV="${VINE_ENV:-vine}"
RDS="${RDS:-data/raw/GSE210139_single_cell_CREST_barcode_qc.rds}"
VINE_DIR="${VINE_DIR:-data/vine}"
OUT_DIR="${OUT_DIR:-data}"
NJ_ONLY="${NJ_ONLY:-0}"

mkdir -p "${VINE_DIR}" "${OUT_DIR}" logs

conda activate "${PREPROCESS_ENV}"
Rscript scripts/prepare_crest_vine_inputs.R \
    --rds "${RDS}" \
    --outdir "${VINE_DIR}" \
    --datasets "${DATASETS}" \
    --unit "${UNIT}" \
    --min-clone-size "${MIN_CLONE_SIZE}" \
    --min-dominant-fraction "${MIN_DOMINANT_FRACTION}"

MANIFEST="${VINE_DIR}/crest_${UNIT}_manifest.tsv"

conda activate "${VINE_ENV}"
tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r dataset unit matrix statelabels metadata state_map n_leaves n_sites; do
    prefix="${dataset}_${unit}"
    samples="${VINE_DIR}/${prefix}_vine_samples.nwk"
    mean="${VINE_DIR}/${prefix}_vine_mean.nwk"
    states="${VINE_DIR}/${prefix}_vine_states.nex"
    log="${VINE_DIR}/${prefix}_vine.log"

    if [[ "${NJ_ONLY}" == "1" ]]; then
        vine "${matrix}" \
            --format CRISPR \
            --migration "${statelabels}" \
            --nj-only \
            --labeled-trees "${states}" \
            --logfile "${log}" \
            > "${samples}"
    else
        vine "${matrix}" \
            --format CRISPR \
            --migration "${statelabels}" \
            --mean "${mean}" \
            --nsamples "${NSAMPLES}" \
            -j "${THREADS}" \
            --logfile "${log}" \
            --labeled-trees "${states}" \
            > "${samples}"
    fi
done

conda activate "${PREPROCESS_ENV}"
if [[ "${NJ_ONLY}" == "1" ]]; then
    tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r dataset unit matrix statelabels metadata state_map n_leaves n_sites; do
        prefix="${dataset}_${unit}"
        python scripts/prepare_vine_newick_tree.py \
            --newick "${VINE_DIR}/${prefix}_vine_samples.nwk" \
            --tree-out "${OUT_DIR}/${prefix}_vine_tree.nwk"
    done
else
    tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r dataset unit matrix statelabels metadata state_map n_leaves n_sites; do
        prefix="${dataset}_${unit}"
        python scripts/prepare_vine_tree_regime.py \
            --nexus "${VINE_DIR}/${prefix}_vine_states.nex" \
            --tree-out "${OUT_DIR}/${prefix}_vine_tree.nwk" \
            --regime-out "${OUT_DIR}/${prefix}_vine_regime.csv"
    done
fi
