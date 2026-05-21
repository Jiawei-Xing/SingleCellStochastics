#!/bin/bash
# Reconstruct latent OU history for a single gene under the pro_da_glu H1 fit
# from the VINE fp lineage tree. CPU-only — no slurm needed.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

GENE="${GENE:-Ddc}"
HYP="${HYP:-h1}"

FP_PREFIX="${FP_PREFIX:-E15_rep1_fp_lineage_homogeneous_clone}"
FP_EXPR_DIR="${FP_EXPR_DIR:-data/expression/e15_rep1_fp_lineage}"
FP_VINE_DIR="${FP_VINE_DIR:-data/vine_no_migration/e15_rep1_fp_lineage}"
FP_OUT_BASE="${FP_OUT_BASE:-outputs/e15_rep1_fp_lineage_vine}"
DIFF_DIR="${FP_OUT_BASE}/diff"
RECON_DIR="${FP_OUT_BASE}/reconstruct"

DIFF_PREFIX="${DIFF_PREFIX:-${FP_PREFIX}_vine_diff_pro_da_glu}"

FP_TREE="${FP_VINE_DIR}/${FP_PREFIX}_vine_parsimony_named_tree.nwk"
ALT_REGIME="${FP_VINE_DIR}/${FP_PREFIX}_vine_lavous_alt_pro_da_glu.csv"
FP_EXPR="${FP_EXPR_DIR}/${FP_PREFIX}_readcounts.tsv"
FP_LIBRARY="${FP_EXPR_DIR}/${FP_PREFIX}_library.tsv"

Q_PARAMS="${DIFF_DIR}/${DIFF_PREFIX}_${HYP}_q-mean-std_0.tsv"
OU_PARAMS="${DIFF_DIR}/${DIFF_PREFIX}_model-params.tsv"

mkdir -p "${RECON_DIR}"

for f in "${FP_TREE}" "${ALT_REGIME}" "${FP_EXPR}" "${FP_LIBRARY}" \
         "${Q_PARAMS}" "${OU_PARAMS}"; do
    if [[ ! -s "${f}" ]]; then
        echo "Missing required input: ${f}" >&2
        exit 1
    fi
done

OUT_FIG="${RECON_DIR}/${GENE}_pro_da_glu_${HYP}.png"
OUT_TSV="${RECON_DIR}/${GENE}_pro_da_glu_${HYP}.tsv"

lavous-reconstruct \
    --tree "${FP_TREE}" \
    --q_params "${Q_PARAMS}" \
    --gene "${GENE}" \
    --read_counts "${FP_EXPR}" \
    --library "${FP_LIBRARY}" \
    --regime "${ALT_REGIME}" \
    --ou "${OU_PARAMS}" \
    --hypothesis "${HYP}" \
    --out_fig "${OUT_FIG}" \
    --out_tsv "${OUT_TSV}"
