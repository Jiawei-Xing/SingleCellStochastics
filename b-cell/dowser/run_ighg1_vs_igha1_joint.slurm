#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=lavous_bcell_ighg1_igha1_joint
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lavous-bcell-ighg1-igha1-joint-%j.out
#SBATCH --error=logs/lavous-bcell-ighg1-igha1-joint-%j.err

set -euo pipefail

DEFAULT_WORK_ROOT="/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/b-cell/dowser"
WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_WORK_ROOT}}}"
if [[ ! -d "${WORK_ROOT}/ighg1_vs_igha1" && -d "${DEFAULT_WORK_ROOT}/ighg1_vs_igha1" ]]; then
    WORK_ROOT="${DEFAULT_WORK_ROOT}"
fi
cd "${WORK_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-singlecellstochastics}"

WORK_DIR="${WORK_DIR:-ighg1_vs_igha1}"
OUT_BASE="${OUT_BASE:-${WORK_DIR}/joint_outputs}"
GENE_SET="${GENE_SET:-full}"
FDR="${FDR:-0.05}"
MIN_CELLS="${MIN_CELLS:-5}"
MIN_TOTAL_COUNTS="${MIN_TOTAL_COUNTS:-10}"
SIMS="${SIMS:-2000}"

BATCH="${BATCH:-1000}"
ITER="${ITER:-10000}"
LR="${LR:-1e-1}"
WINDOW="${WINDOW:-200}"
TOL="${TOL:-1e-4}"
MODEL="${MODEL:-nb}"
HERIT_EXTRA_ARGS="${HERIT_EXTRA_ARGS:-}"
DIFF_EXTRA_ARGS="${DIFF_EXTRA_ARGS:-}"
CALIBRATE_EXTRA_ARGS="${CALIBRATE_EXTRA_ARGS:-}"

CLONES="${CLONES:-Clonotype_1261,Clonotype_133}"
IFS=',' read -r -a CLONE_LIST <<< "${CLONES}"
if [[ "${#CLONE_LIST[@]}" -lt 2 ]]; then
    echo "CLONES must contain at least two comma-separated clonotypes" >&2
    exit 2
fi

PREFIX="${PREFIX:-ighg1_vs_igha1_joint.${GENE_SET}}"
NULL_REGIME="${NULL_REGIME:-IGHG1}"

mkdir -p "${OUT_BASE}/filtered" "${OUT_BASE}/heritability" "${OUT_BASE}/diff/${GENE_SET}" logs

if [[ "${GENE_SET}" == "no_ig" ]]; then
    EXPR_DIR="${WORK_DIR}/readcounts_no_ig"
    EXPR_SUFFIX=".readcounts.no_ig.tsv"
    ANNOT="${WORK_DIR}/metadata/gene_annotation.no_ig.tsv"
elif [[ "${GENE_SET}" == "full" ]]; then
    EXPR_DIR="${WORK_DIR}/readcounts"
    EXPR_SUFFIX=".readcounts.tsv"
    ANNOT="${WORK_DIR}/metadata/gene_annotation.tsv"
else
    echo "Unknown GENE_SET=${GENE_SET}; use no_ig or full" >&2
    exit 2
fi

HERITABILITY="${OUT_BASE}/heritability/${PREFIX}.heritability.tsv"
DIFF_DIR="${OUT_BASE}/diff/${GENE_SET}"

TREE_CSV=""
EXPR_CSV=""
REGIME_CSV=""
LIB_CSV=""
FILTERED_CSV=""
HERIT_EXPR_CSV=""

for CLONE in "${CLONE_LIST[@]}"; do
    CLONE="$(echo "${CLONE}" | xargs)"
    [[ -n "${CLONE}" ]] || continue

    TREE="${WORK_DIR}/newick/${CLONE}.nwk"
    EXPR="${EXPR_DIR}/${CLONE}${EXPR_SUFFIX}"
    REGIME="${WORK_DIR}/regimes/${CLONE}.ighg1_vs_igha1.regime.csv"
    LIB="${WORK_DIR}/library/${CLONE}.library.tsv"
    FILTERED="${OUT_BASE}/filtered/${CLONE}.${GENE_SET}.joint.mincells${MIN_CELLS}.mincounts${MIN_TOTAL_COUNTS}.tsv"
    HERIT_EXPR="${OUT_BASE}/heritability/${CLONE}.${PREFIX}.heritability_q${FDR}.tsv"

    for REQUIRED in "${TREE}" "${EXPR}" "${REGIME}" "${LIB}"; do
        if [[ ! -s "${REQUIRED}" ]]; then
            echo "Missing required input: ${REQUIRED}" >&2
            exit 2
        fi
    done

    TREE_CSV="${TREE_CSV:+${TREE_CSV},}${TREE}"
    EXPR_CSV="${EXPR_CSV:+${EXPR_CSV},}${EXPR}"
    REGIME_CSV="${REGIME_CSV:+${REGIME_CSV},}${REGIME}"
    LIB_CSV="${LIB_CSV:+${LIB_CSV},}${LIB}"
    FILTERED_CSV="${FILTERED_CSV:+${FILTERED_CSV},}${FILTERED}"
    HERIT_EXPR_CSV="${HERIT_EXPR_CSV:+${HERIT_EXPR_CSV},}${HERIT_EXPR}"
done

echo "[$(date)] filtering shared ${GENE_SET} genes for joint IGHG1-vs-IGHA1 analysis"
export EXPR_CSV FILTERED_CSV MIN_CELLS MIN_TOTAL_COUNTS
python - <<'PY'
import os
import pandas as pd

expr_paths = os.environ["EXPR_CSV"].split(",")
filtered_paths = os.environ["FILTERED_CSV"].split(",")
exprs = [pd.read_csv(path, sep="\t", index_col=0) for path in expr_paths]

min_cells = int(os.environ["MIN_CELLS"])
min_total = int(os.environ["MIN_TOTAL_COUNTS"])
common = list(exprs[0].columns)
for expr in exprs[1:]:
    expr_genes = set(expr.columns)
    common = [gene for gene in common if gene in expr_genes]

exprs = [expr.loc[:, common] for expr in exprs]
combined = pd.concat(exprs, axis=0)
keep = ((combined > 0).sum(axis=0) >= min_cells) & (combined.sum(axis=0) >= min_total)
genes = keep.index[keep].tolist()
if not genes:
    raise SystemExit(
        f"No genes remain after joint sparse filter: min_cells={min_cells}, "
        f"min_total_counts={min_total}"
    )

for expr, path in zip(exprs, filtered_paths):
    expr.loc[:, genes].to_csv(path, sep="\t")

print(
    f"Kept {len(genes)} / {combined.shape[1]} shared genes across "
    f"{' + '.join(str(expr.shape[0]) for expr in exprs)} cells"
)
for path in filtered_paths:
    print(f"Wrote {path}")
PY

if [[ -s "${HERITABILITY}" ]]; then
    echo "[$(date)] using existing joint heritability: ${HERITABILITY}"
else
    echo "[$(date)] running joint lavous-heritability"
    lavous-heritability \
        --tree "${TREE_CSV}" \
        --expression "${FILTERED_CSV}" \
        --library "${LIB_CSV}" \
        --outfile "${HERITABILITY}" \
        --model "${MODEL}" \
        --batch "${BATCH}" \
        --iter "${ITER}" \
        --lr "${LR}" \
        --window "${WINDOW}" \
        --tol "${TOL}" \
        --resume \
        ${HERIT_EXTRA_ARGS}
fi

echo "[$(date)] selecting jointly heritable genes at q < ${FDR}"
export FILTERED_CSV HERITABILITY HERIT_EXPR_CSV FDR
SELECT_STATUS=0
python - <<'PY' || SELECT_STATUS=$?
import os
from pathlib import Path
import pandas as pd

fdr = float(os.environ["FDR"])
heritability = pd.read_csv(os.environ["HERITABILITY"], sep="\t")
selected = heritability.loc[heritability["q"] < fdr, "gene"].astype(str).tolist()
selected = list(dict.fromkeys(selected))

filtered_paths = os.environ["FILTERED_CSV"].split(",")
herit_expr_paths = os.environ["HERIT_EXPR_CSV"].split(",")
exprs = [pd.read_csv(path, sep="\t", index_col=0) for path in filtered_paths]
selected = [
    gene for gene in selected
    if all(gene in expr.columns for expr in exprs)
]

if not selected:
    for path in herit_expr_paths:
        Path(path).write_text("")
    print(f"No jointly heritable genes passed q < {fdr}; skipping diff/calibration")
    raise SystemExit(3)

for expr, path in zip(exprs, herit_expr_paths):
    expr.loc[:, selected].to_csv(path, sep="\t")
print(f"Selected {len(selected)} jointly heritable genes at q < {fdr}")
for path in herit_expr_paths:
    print(f"Wrote {path}")
PY
if [[ "${SELECT_STATUS}" == "3" ]]; then
    exit 0
elif [[ "${SELECT_STATUS}" != "0" ]]; then
    exit "${SELECT_STATUS}"
fi

echo "[$(date)] running joint lavous-diff for IGHA1 vs ${NULL_REGIME}"
lavous-diff \
    --tree "${TREE_CSV}" \
    --expression "${HERIT_EXPR_CSV}" \
    --regime "${REGIME_CSV}" \
    --library "${LIB_CSV}" \
    --annot "${ANNOT}" \
    --null "${NULL_REGIME}" \
    --outdir "${DIFF_DIR}" \
    --prefix "${PREFIX}" \
    --batch "${BATCH}" \
    --iter "${ITER}" \
    --lr "${LR}" \
    --window "${WINDOW}" \
    --tol "${TOL}" \
    --resume \
    ${DIFF_EXTRA_ARGS}

CHI="${DIFF_DIR}/${PREFIX}_chi-squared.tsv"
CACHE="${DIFF_DIR}/${PREFIX}.sim_all${SIMS}.null_lrs.pkl"
if [[ ! -s "${CHI}" ]]; then
    echo "Missing chi-squared input after diff: ${CHI}" >&2
    exit 2
fi

echo "[$(date)] calibrating joint diff ${CHI} with sim_all=${SIMS}"
lavous-calibrate \
    --chi "${CHI}" \
    --sim_all "${SIMS}" \
    --batch "${BATCH}" \
    --cache "${CACHE}" \
    --gpd_tail \
    ${CALIBRATE_EXTRA_ARGS}

echo "[$(date)] joint heritability, diff, and calibration done"
