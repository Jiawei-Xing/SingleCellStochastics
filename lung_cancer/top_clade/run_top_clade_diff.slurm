#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_top
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-top-clade-%j.out
#SBATCH --error=logs/lavous-top-clade-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

SOURCE_PREFIX="${SOURCE_PREFIX:-3724_NT_All}"
PREFIX="${PREFIX:-3724_NT_TopClade}"
TARGET_NODE="${TARGET_NODE:-auto}"
FDR="${FDR:-0.05}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-data}"
WORK_DIR="${WORK_DIR:-top_clade}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
OUT_BASE="${OUT_BASE:-${WORK_DIR}/outputs}"
DIFF_DIR="${OUT_BASE}/diff"
HERIT_DIR="${OUT_BASE}/heritability"
LOG_DIR="${LOG_DIR:-logs}"

GENE_MODE="${GENE_MODE:-source_diff}"
SOURCE_DIFF="${SOURCE_DIFF:-outputs/diff/${SOURCE_PREFIX}_diff_Pri-Met_empirical-all_10000.tsv}"
GENE_FILE="${GENE_FILE:-}"
RUN_CALIBRATE="${RUN_CALIBRATE:-0}"
SIM_ALL="${SIM_ALL:-10000}"
CAL_BATCH="${CAL_BATCH:-1000}"

mkdir -p "${DATA_DIR}" "${DIFF_DIR}" "${HERIT_DIR}" "${LOG_DIR}"

python "${WORK_DIR}/prepare_top_clade_inputs.py" \
    --tree "${SOURCE_DATA_DIR}/${SOURCE_PREFIX}_vine_tree.nwk" \
    --regime "${SOURCE_DATA_DIR}/${SOURCE_PREFIX}_vine_regime.csv" \
    --read-counts "${SOURCE_DATA_DIR}/${SOURCE_PREFIX}_readcounts.tsv" \
    --library "${SOURCE_DATA_DIR}/${SOURCE_PREFIX}_library.tsv" \
    --outdir "${DATA_DIR}" \
    --prefix "${PREFIX}" \
    --target-node "${TARGET_NODE}"

TREE="${DATA_DIR}/${PREFIX}_vine_tree.nwk"
REGIME="${DATA_DIR}/${PREFIX}_vine_regime.csv"
FULL_EXPR="${DATA_DIR}/${PREFIX}_readcounts.tsv"
LIBRARY="${DATA_DIR}/${PREFIX}_library.tsv"
DIFF_EXPR="${DIFF_DIR}/${PREFIX}_readcounts_for_diff.tsv"
HERITABILITY="${HERIT_DIR}/${PREFIX}_heritability.tsv"

if [[ "${GENE_MODE}" == "top_heritability" ]]; then
    lavous-heritability \
        --tree "${TREE}" \
        --expression "${FULL_EXPR}" \
        --library "${LIBRARY}" \
        --outfile "${HERITABILITY}" \
        --resume
fi

export FULL_EXPR DIFF_EXPR GENE_MODE SOURCE_DIFF GENE_FILE FDR HERITABILITY
python - <<'PY'
import os
from pathlib import Path

import pandas as pd

expr = pd.read_csv(os.environ["FULL_EXPR"], sep="\t", index_col=0)
mode = os.environ["GENE_MODE"]

if mode == "source_diff":
    source = Path(os.environ["SOURCE_DIFF"])
    if not source.exists():
        raise SystemExit(f"GENE_MODE=source_diff but {source} does not exist")
    genes = pd.read_csv(source, sep="\t")["gene"].astype(str).tolist()
elif mode == "gene_file":
    gene_file = Path(os.environ["GENE_FILE"])
    if not gene_file.exists():
        raise SystemExit("GENE_MODE=gene_file requires GENE_FILE to exist")
    genes = [line.strip() for line in gene_file.read_text().splitlines() if line.strip()]
elif mode == "top_heritability":
    heritability = Path(os.environ["HERITABILITY"])
    if not heritability.exists():
        raise SystemExit(f"Top-clade heritability file is missing: {heritability}")
    h = pd.read_csv(heritability, sep="\t")
    genes = h.loc[h["q"] < float(os.environ["FDR"]), "gene"].astype(str).tolist()
elif mode == "all":
    genes = list(expr.columns)
else:
    raise SystemExit(
        f"Unknown GENE_MODE={mode!r}; use source_diff, gene_file, top_heritability, or all."
    )

genes = [gene for gene in dict.fromkeys(genes) if gene in expr.columns]
if not genes:
    raise SystemExit("No requested genes were present in the top-clade expression matrix")

expr[genes].to_csv(os.environ["DIFF_EXPR"], sep="\t")
print(f"Wrote {len(genes)} genes x {expr.shape[0]} cells to {os.environ['DIFF_EXPR']}")
PY

lavous-diff \
    --tree "${TREE}" \
    --expression "${DIFF_EXPR}" \
    --regime "${REGIME}" \
    --library "${LIBRARY}" \
    --null T1 \
    --outdir "${DIFF_DIR}" \
    --prefix "${PREFIX}_diff_Pri-Met" \
    --resume

if [[ "${RUN_CALIBRATE}" == "1" ]]; then
    lavous-calibrate \
        --chi "${DIFF_DIR}/${PREFIX}_diff_Pri-Met_chi-squared.tsv" \
        --sim_all "${SIM_ALL}" \
        --batch "${CAL_BATCH}" \
        --gpd_tail \
        --cache "${DIFF_DIR}/${PREFIX}_diff_Pri-Met_null_LRs.pkl"
fi
