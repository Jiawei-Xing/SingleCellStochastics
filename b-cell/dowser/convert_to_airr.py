"""
Convert TRIBAL covid19 inputs to AIRR/Change-O TSV for Dowser/IgPhyML.

Input files (from tribal_sarscov2_score_example/_covid_data/covid19_data/):
  - human_data.filtered.csv    one row per cell, heavy + light chain seqs + metadata
  - human_data_root_seq.csv    per-clonotype germline-ish root sequences
  - human_encoding.txt         isotype index -> isotype name (1-based file, 0-based in data)

Output: AIRR TSV with the columns Dowser's formatClones() expects:
  sequence_id, clone_id, sequence_alignment, germline_alignment_d_mask,
  v_call, j_call, junction_length, plus passthrough metadata columns.

Notes:
  - Within a clone, all heavy-chain seqs share the same length (TRIBAL pre-aligns).
    For 80/207 clones the stored root sequence is *longer* than the cell seqs by a
    multiple-of-3 tail; we trim the root tail to the cell seq length.
  - Sequences are trimmed at the 3' end to the nearest multiple of 3 to keep the
    codon frame (heavy chain starts in frame from position 1).
  - junction_length is a placeholder (default 30). IgPhyML's HLP19 model will still
    run; CDR3-region hot-spot weighting just won't be regionally targeted. Re-run
    IgBLAST on the source FASTQs if you need accurate CDR3 boundaries.
  - 4 clonotypes have within-clone length variation; we drop them.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path(
    "/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/"
    "b-cell/tribal/tribal_sarscov2_score_example/_covid_data/covid19_data"
)


def load_inputs(data_dir: Path):
    df = pd.read_csv(data_dir / "human_data.filtered.csv")
    root = pd.read_csv(data_dir / "human_data_root_seq.csv", index_col=0)
    return df, root


def build_airr(df: pd.DataFrame, root: pd.DataFrame, junction_length: int = 30) -> pd.DataFrame:
    rows = []
    dropped_lenvar = 0
    dropped_missing_root = 0

    for clone, sub in df.groupby("Clonotype"):
        if clone not in root.index:
            dropped_missing_root += len(sub)
            continue

        seq_lens = sub["Heavy Chain Variable Seq"].str.len().unique()
        if len(seq_lens) > 1:
            dropped_lenvar += len(sub)
            continue

        seq_len = int(seq_lens[0])
        codon_len = (seq_len // 3) * 3  # truncate trailing 1-2 bp to keep frame

        root_seq = root.loc[clone, "Heavy Chain Root"]
        # Root sometimes has a trailing extension; trim from the end to match.
        if len(root_seq) >= seq_len:
            root_trim = root_seq[:codon_len]
        else:
            # Rare: root is shorter than seqs. Pad with N's so lengths match.
            root_trim = root_seq + "N" * (codon_len - len(root_seq))
            root_trim = root_trim[:codon_len]

        for _, r in sub.iterrows():
            seq = r["Heavy Chain Variable Seq"][:codon_len]
            rows.append({
                "sequence_id": f"{clone}|{r['cellid']}|{r['seq']}",
                "clone_id": clone,
                "sequence_alignment": seq,
                "germline_alignment_d_mask": root_trim,
                "v_call": f"{r['Heavy Chain V Allele']}*01",
                "j_call": f"{r['Heavy Chain J Allele']}*01",
                "junction_length": junction_length,
                # Passthrough metadata
                "cell_id": r["cellid"],
                "c_call": r["Heavy Chain Isotype"],
                "isotype": r["Heavy Chain Isotype"],
                "isotype_idx": r["isotype"],
                "timepoint": r["Timepoint"],
                "dataset": r["Dataset"],
                "light_v_call": r["Light Chain V Allele"],
                "light_j_call": r["Light Chain J Allele"],
                "light_c_call": r["Light Chain Isotype"],
            })

    out = pd.DataFrame(rows)
    print(f"[convert] kept {len(out)} sequences across {out['clone_id'].nunique()} clones",
          file=sys.stderr)
    print(f"[convert] dropped {dropped_lenvar} seqs (within-clone length variation)",
          file=sys.stderr)
    print(f"[convert] dropped {dropped_missing_root} seqs (missing root)",
          file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, required=True,
                    help="output AIRR TSV path")
    ap.add_argument("--junction-length", type=int, default=30,
                    help="placeholder CDR3 length for IgPhyML (default 30)")
    ap.add_argument("--min-clone-size", type=int, default=3,
                    help="drop clones with fewer than N sequences after filtering")
    args = ap.parse_args()

    df, root = load_inputs(args.data_dir)
    airr = build_airr(df, root, junction_length=args.junction_length)

    if args.min_clone_size > 1:
        sizes = airr.groupby("clone_id").size()
        keep = sizes[sizes >= args.min_clone_size].index
        before = len(airr)
        airr = airr[airr["clone_id"].isin(keep)]
        print(f"[convert] min_clone_size={args.min_clone_size}: "
              f"{before}->{len(airr)} seqs, "
              f"{len(sizes)}->{airr['clone_id'].nunique()} clones",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    airr.to_csv(args.out, sep="\t", index=False)
    print(f"[convert] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
