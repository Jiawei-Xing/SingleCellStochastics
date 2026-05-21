#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_e15_fp
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-e15-rep1-fp-%j.out
#SBATCH --error=logs/lavous-e15-rep1-fp-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

# Heritability runs on the full 920-leaf E15_rep1 tree (all clones, mixed
# allowed) for Pagel-lambda power; mixed-clone representative cells are
# picked by idxmax(total_counts) within the clone's dominant cell type. Diff
# and calibration stay on the 162-leaf fp lineage.
ALL_PREFIX="${ALL_PREFIX:-E15_rep1_clone}"
FP_PREFIX="${FP_PREFIX:-E15_rep1_fp_lineage_homogeneous_clone}"
FDR="${FDR:-0.05}"
SIM_ALL="${SIM_ALL:-2000}"
BATCH="${BATCH:-1000}"
DIFF_BATCH="${DIFF_BATCH:-1000}"
CAL_BATCH="${CAL_BATCH:-1000}"

ALL_DATA_DIR="${ALL_DATA_DIR:-data/lavous/e15_rep1}"
ALL_TREE_DIR="${ALL_TREE_DIR:-data/laml/e15_rep1}"
FP_DATA_DIR="${FP_DATA_DIR:-data/lavous/e15_rep1_fp_lineage}"
FP_TREE_DIR="${FP_TREE_DIR:-data/laml/e15_rep1_fp_lineage}"
ALL_OUT_BASE="${ALL_OUT_BASE:-outputs/e15_rep1}"
FP_OUT_BASE="${FP_OUT_BASE:-outputs/e15_rep1_fp_lineage}"
HERIT_DIR="${ALL_OUT_BASE}/heritability"
DIFF_DIR="${FP_OUT_BASE}/diff"
CAL_DIR="${FP_OUT_BASE}/calibration"

ALL_TREE="${ALL_TREE_DIR}/${ALL_PREFIX}_laml_n1_trees.nwk"
ALL_EXPR="${ALL_DATA_DIR}/${ALL_PREFIX}_readcounts.tsv"
ALL_LIBRARY="${ALL_DATA_DIR}/${ALL_PREFIX}_library.tsv"

FP_TREE="${FP_TREE_DIR}/${FP_PREFIX}_laml_full_topology_parsimony_named_tree.nwk"
ALT_REGIME="${FP_TREE_DIR}/${FP_PREFIX}_laml_full_topology_lavous_alt_pro_da_glu.csv"
NULL_REGIME="${FP_TREE_DIR}/${FP_PREFIX}_laml_full_topology_lavous_null_pro_diff.csv"
FP_EXPR="${FP_DATA_DIR}/${FP_PREFIX}_readcounts.tsv"
FP_LIBRARY="${FP_DATA_DIR}/${FP_PREFIX}_library.tsv"

HERITABILITY="${HERIT_DIR}/${ALL_PREFIX}_heritability.tsv"
HERIT_EXPR="${HERIT_DIR}/${FP_PREFIX}_readcounts_heritability_q${FDR}.tsv"
DIFF_PREFIX="${FP_PREFIX}_diff_pro_da_glu"
CHI_TSV="${DIFF_DIR}/${DIFF_PREFIX}_chi-squared.tsv"
CACHE="${CAL_DIR}/${DIFF_PREFIX}_null_LRs.pkl"

mkdir -p "${HERIT_DIR}" "${DIFF_DIR}" "${CAL_DIR}" logs

for f in "${ALL_TREE}" "${ALL_EXPR}" "${ALL_LIBRARY}" \
         "${FP_TREE}" "${ALT_REGIME}" "${NULL_REGIME}" "${FP_EXPR}" "${FP_LIBRARY}"; do
    if [[ ! -s "${f}" ]]; then
        echo "Missing required input: ${f}" >&2
        exit 1
    fi
done

lavous-heritability \
    --tree "${ALL_TREE}" \
    --expression "${ALL_EXPR}" \
    --library "${ALL_LIBRARY}" \
    --outfile "${HERITABILITY}" \
    --batch "${BATCH}" \
    --resume

export FP_EXPR HERITABILITY HERIT_EXPR FDR
python - <<'PY'
import os
import pandas as pd

fdr = float(os.environ["FDR"])
heritability = pd.read_csv(os.environ["HERITABILITY"], sep="\t")
selected = heritability.loc[heritability["q"] < fdr, "gene"].astype(str).tolist()
if not selected:
    raise SystemExit(f"No heritable genes passed q < {fdr}; skipping lavous-diff and calibration")

# Heritable list is derived from the full 920-leaf expr matrix; the diff test
# consumes the 162-leaf fp matrix. Intersect rather than error on genes
# filtered out of the smaller matrix.
expr = pd.read_csv(os.environ["FP_EXPR"], sep="\t", index_col=0)
in_fp = [gene for gene in selected if gene in expr.columns]
dropped = len(selected) - len(in_fp)
if dropped:
    print(f"Dropping {dropped} heritable genes absent from fp expr (sparsity filter)")
if not in_fp:
    raise SystemExit("No heritable genes remain after intersection with fp expr matrix")
expr[in_fp].to_csv(os.environ["HERIT_EXPR"], sep="\t")
print(f"Selected {len(in_fp)} heritable genes at q < {fdr} (kept in fp expr); wrote {os.environ['HERIT_EXPR']}")
PY

lavous-diff \
    --tree "${FP_TREE}" \
    --expression "${HERIT_EXPR}" \
    --regime "${ALT_REGIME}" \
    --null_regime "${NULL_REGIME}" \
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
