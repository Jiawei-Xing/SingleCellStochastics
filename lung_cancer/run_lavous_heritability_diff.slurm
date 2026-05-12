#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_hd
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-herit-diff-%j.out
#SBATCH --error=logs/lavous-herit-diff-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

PREFIX="${PREFIX:-3724_NT_All}"
FDR="${FDR:-0.05}"

DATA_DIR="${DATA_DIR:-data}"
OUT_BASE="${OUT_BASE:-outputs}"
HERIT_DIR="${OUT_BASE}/heritability"
DIFF_DIR="${OUT_BASE}/diff"

TREE="${DATA_DIR}/${PREFIX}_vine_tree.nwk"
REGIME="${DATA_DIR}/${PREFIX}_vine_regime.csv"
FULL_EXPR="${DATA_DIR}/${PREFIX}_readcounts.tsv"
LIBRARY="${DATA_DIR}/${PREFIX}_library.tsv"
HERITABILITY="${HERIT_DIR}/${PREFIX}_heritability.tsv"
HERIT_EXPR="${HERIT_DIR}/${PREFIX}_readcounts_heritability_q${FDR}.tsv"

mkdir -p "${HERIT_DIR}" "${DIFF_DIR}" logs

lavous-heritability \
    --tree "${TREE}" \
    --expression "${FULL_EXPR}" \
    --library "${LIBRARY}" \
    --outfile "${HERITABILITY}" \
    --resume

export FULL_EXPR HERITABILITY HERIT_EXPR FDR
python - <<'PY'
import os
import pandas as pd

fdr = float(os.environ["FDR"])
heritability = pd.read_csv(os.environ["HERITABILITY"], sep="\t")
selected = heritability.loc[heritability["q"] < fdr, "gene"].astype(str).tolist()
if not selected:
    raise SystemExit(f"No heritable genes passed q < {fdr}; skipping lavous-diff")

expr = pd.read_csv(os.environ["FULL_EXPR"], sep="\t", index_col=0)
missing = [gene for gene in selected if gene not in expr.columns]
if missing:
    preview = ", ".join(missing[:10])
    raise SystemExit(f"{len(missing)} selected genes were missing from expression matrix: {preview}")

expr[selected].to_csv(os.environ["HERIT_EXPR"], sep="\t")
print(f"Selected {len(selected)} genes at q < {fdr}; wrote {os.environ['HERIT_EXPR']}")
PY

lavous-diff \
    --tree "${TREE}" \
    --expression "${HERIT_EXPR}" \
    --regime "${REGIME}" \
    --library "${LIBRARY}" \
    --null T1 \
    --outdir "${DIFF_DIR}" \
    --prefix "${PREFIX}_diff_Pri-Met" \
    --resume
