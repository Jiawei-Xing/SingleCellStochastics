# Dowser/IgPhyML trees from TRIBAL covid19 data

Pipeline to build ML B-cell lineage trees from the TRIBAL SARS-CoV-2 example dataset (`../tribal/tribal_sarscov2_score_example/_covid_data/covid19_data/`).

## Output of conversion step
- 2456 heavy-chain sequences across 203 clones (out of 2508/207 in the source)
- 52 sequences dropped: 4 clonotypes had within-clone HC length variation
- Sequences trimmed at 3' end to keep codon frame (heavy chain in-frame from base 1)

## Caveats
- **`junction_length` is a placeholder (30 bp).** The source TRIBAL CSV doesn't carry CDR3 boundaries. IgPhyML's HLP19 model still runs, but the CDR/FWR region weighting won't be correctly partitioned. If you care about that, run IgBLAST on the original cellranger sequences to recover AIRR fields.
- **Allele suffix added as `*01`.** TRIBAL stores V/J at the gene level (e.g., `IGHV3-23`); Dowser/IgPhyML want `IGHV3-23*01`. The actual allele isn't used by the substitution model — only for sanity checks — so this is fine.
- **Root sequence** in `human_data_root_seq.csv` is used as `germline_alignment_d_mask`. For 80 clones the root has a trailing extension (multiple-of-3) past the cell sequences; we trim it to match. Otherwise it's treated as the inferred germline ancestor for rooting.

## Run

### 1. Convert
```bash
cd /grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/b-cell/dowser
conda activate tribal       # any env with pandas
python convert_to_airr.py --out covid19.airr.tsv
```

### 2. Build trees
You need a conda env with R + Dowser + Alakazam. Create one:
```bash
conda create -n dowser -c conda-forge -c bioconda \
    r-base r-dowser r-alakazam r-airr r-optparse r-ape
conda activate dowser
```

#### Backend choice
- `--build igphyml`  HLP19 codon model, BCR-aware. Slowest but the right model for SHM.
- `--build raxml`    GTR+gamma nucleotide. Fast, no BCR-specific terms.
- `--build pml`      `phangorn::pml`, R-only. No external binary needed; slow for large clones.

#### IgPhyML install
IgPhyML isn't in our conda envs yet. Either:
- `conda install -n dowser -c bioconda igphyml` (try this first)
- Or build from source: https://bitbucket.org/kbhoehn/igphyml

Find the binary then:
```bash
Rscript build_trees.R \
    --airr covid19.airr.tsv \
    --out-dir trees_igphyml \
    --build igphyml \
    --igphyml-exec $(which igphyml) \
    --ncores 8
```

#### RAxML fallback (recommended for first pass)
RAxML-NG is easier to install:
```bash
conda install -n dowser -c bioconda raxml-ng
Rscript build_trees.R \
    --airr covid19.airr.tsv \
    --out-dir trees_raxml \
    --build raxml \
    --raxml-exec $(which raxml-ng) \
    --ncores 8
```

#### Pure-R fallback (no external binary)
```bash
Rscript build_trees.R --airr covid19.airr.tsv --out-dir trees_pml --build pml --ncores 8
```

## Outputs
- `trees/trees.rds`         Dowser tibble (one row per clone, `trees` column holds `phylo` objects + tip data)
- `trees/newick/<clone>.nwk` per-clone Newick trees
- `trees/tree_summary.tsv`  n_tips and total branch length per clone

## Step 3: Generate regime CSVs for LaVOUS

`generate_regimes.R` builds one `node_name,regime` CSV per clone (every tip and
every internal node) suitable for `run-ou-poisson --regime`.

```bash
Rscript generate_regimes.R \
    --trees trees_igphyml/trees.rds \
    --airr  covid19.airr.tsv \
    --trait isotype \
    --out-dir regimes
```

What it does, per clone:
1. Reroots the IgPhyML tree on the `Germline` tip, then drops it.
2. Assigns unique internal node labels (`n1, n2, ...`).
3. Runs Fitch parsimony (`phangorn::ancestral.pars`, type `ACCTRAN`) on the
   tip isotype trait; ties broken by argmax.
4. Writes `regimes/<clone>.regime.csv` and the rerooted/pruned Newick to
   `regimes/newick/<clone>.nwk` so node names line up.

`--trait` can also be `timepoint` or `dataset` if you want a different regime
axis. Isotype is the default because CSR is the trait the B-cell tree model
is built to capture.

## Next step: LaVOUS / OU-Poisson
Feed each `regimes/newick/<clone>.nwk` plus its `regimes/<clone>.regime.csv`
to `run-ou-poisson` along with that clone's cell × gene expression matrix.
