#!/usr/bin/env bash
set -euo pipefail

DEFAULT_WORK_ROOT="/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/b-cell/dowser"
WORK_ROOT="${WORK_ROOT:-${DEFAULT_WORK_ROOT}}"
cd "${WORK_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

WORK_DIR="${WORK_DIR:-ighg1_vs_igha1}"
DIFF_DIR="${WORK_DIR}/joint_outputs/diff/full"
PREFIX="${PREFIX:-ighg1_vs_igha1_joint.full}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/reconstruct/joint_h1_xbp1_jchain}"

mkdir -p "${OUT_DIR}/inputs" "${OUT_DIR}/tables" "${OUT_DIR}/plots"

export WORK_DIR DIFF_DIR PREFIX OUT_DIR
python - <<'PY'
import os
from pathlib import Path

import pandas as pd

work_dir = Path(os.environ["WORK_DIR"])
diff_dir = Path(os.environ["DIFF_DIR"])
prefix = os.environ["PREFIX"]
out_dir = Path(os.environ["OUT_DIR"])

genes = {
    "JCHAIN": "ENSG00000132465",
    "XBP1": "ENSG00000100219",
}
clones = [
    ("Clonotype_1261", 0),
    ("Clonotype_133", 1),
]

for clone, idx in clones:
    q_path = diff_dir / f"{prefix}_h1_q-mean-std_{idx}.tsv"
    rc_path = work_dir / "readcounts" / f"{clone}.readcounts.tsv"
    if not q_path.exists():
        raise FileNotFoundError(q_path)
    if not rc_path.exists():
        raise FileNotFoundError(rc_path)

    q = pd.read_csv(q_path, sep="\t", index_col=0)
    missing_q = [ensg for ensg in genes.values() if ensg not in q.index]
    if missing_q:
        raise ValueError(f"{q_path} missing q rows: {missing_q}")
    q_subset = q.loc[list(genes.values())].copy()
    q_subset.index = list(genes.keys())
    q_subset.to_csv(out_dir / "inputs" / f"{clone}.h1_q_xbp1_jchain.tsv", sep="\t")

    rc = pd.read_csv(rc_path, sep="\t", index_col=0)
    missing_rc = [ensg for ensg in genes.values() if ensg not in rc.columns]
    if missing_rc:
        raise ValueError(f"{rc_path} missing readcount columns: {missing_rc}")
    rc_subset = rc.loc[:, list(genes.values())].copy()
    rc_subset.columns = list(genes.keys())
    rc_subset.to_csv(out_dir / "inputs" / f"{clone}.readcounts_xbp1_jchain.tsv", sep="\t")

print(f"Wrote reconstruction input slices to {out_dir / 'inputs'}")
PY

for CLONE in Clonotype_1261 Clonotype_133; do
    for GENE in JCHAIN XBP1; do
        echo "[$(date)] reconstructing ${GENE} on ${CLONE}"
        lavous-reconstruct \
            --tree "${WORK_DIR}/newick/${CLONE}.nwk" \
            --q_params "${OUT_DIR}/inputs/${CLONE}.h1_q_xbp1_jchain.tsv" \
            --gene "${GENE}" \
            --read_counts "${OUT_DIR}/inputs/${CLONE}.readcounts_xbp1_jchain.tsv" \
            --library "${WORK_DIR}/library/${CLONE}.library.tsv" \
            --model ou \
            --regime "${WORK_DIR}/regimes/${CLONE}.ighg1_vs_igha1.regime.csv" \
            --ou "${DIFF_DIR}/${PREFIX}_model-params.tsv" \
            --hypothesis h1 \
            --out_tsv "${OUT_DIR}/tables/${CLONE}.${GENE}.h1_reconstruction.tsv" \
            --out_fig "${OUT_DIR}/plots/${CLONE}.${GENE}.h1_reconstruction.png"
    done
done

echo "[$(date)] reconstruction done: ${OUT_DIR}"
