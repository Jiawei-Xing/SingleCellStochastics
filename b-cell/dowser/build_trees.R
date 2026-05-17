#!/usr/bin/env Rscript
# Build B-cell lineage trees from the AIRR TSV produced by convert_to_airr.py.
#
# Usage:
#   Rscript build_trees.R \
#       --airr covid19.airr.tsv \
#       --out-dir trees \
#       --build igphyml \
#       --igphyml-exec /path/to/igphyml \
#       --ncores 8
#
# Tree-building backends:
#   igphyml  HLP19 codon model (BCR-aware; recommended). Needs IgPhyML binary.
#   raxml    GTR+gamma on nucleotides. Faster, no BCR-specific model.
#   pml      phangorn's pml (R-only). Slowest, no extra dependencies.

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(alakazam)
  library(dowser)
})

opt_list <- list(
  make_option("--airr",          type = "character", help = "Input AIRR TSV"),
  make_option("--out-dir",       type = "character", default = "trees"),
  make_option("--build",         type = "character", default = "igphyml",
              help = "igphyml | raxml | pml [default %default]"),
  make_option("--igphyml-exec",  type = "character", default = Sys.which("igphyml"),
              help = "Path to igphyml binary (only for --build igphyml)"),
  make_option("--raxml-exec",    type = "character", default = Sys.which("raxml-ng"),
              help = "Path to raxml-ng binary (only for --build raxml)"),
  make_option("--ncores",        type = "integer",   default = 4),
  make_option("--min-seq",       type = "integer",   default = 5,
              help = "Min clone size after dedup [default %default]"),
  make_option("--traits",        type = "character", default = "isotype,timepoint,dataset",
              help = "Comma-separated metadata columns to attach to tip labels")
)
opt <- parse_args(OptionParser(option_list = opt_list))

stopifnot(!is.null(opt$airr))
dir.create(opt$`out-dir`, recursive = TRUE, showWarnings = FALSE)

cat("[build_trees] reading", opt$airr, "\n")
db <- airr::read_rearrangement(opt$airr)
cat("[build_trees] loaded", nrow(db), "rows,",
    length(unique(db$clone_id)), "clones\n")

traits <- strsplit(opt$traits, ",", fixed = TRUE)[[1]]

cat("[build_trees] formatClones (collapsing identical seqs, min seq =", opt$`min-seq`, ")\n")
clones <- formatClones(
  db,
  traits      = traits,
  minseq      = opt$`min-seq`,
  num_fields  = NULL,
  text_fields = traits,
  columns     = c("cell_id", "c_call"),
  collapse    = TRUE
)
cat("[build_trees]", nrow(clones), "clones passed formatClones\n")

build <- opt$build
cat("[build_trees] building trees with backend:", build, "\n")

trees <- switch(build,
  igphyml = {
    if (opt$`igphyml-exec` == "") stop("igphyml not found; pass --igphyml-exec")
    getTrees(clones, build = "igphyml",
             exec = opt$`igphyml-exec`, nproc = opt$ncores,
             rseed = 42)
  },
  raxml = {
    if (opt$`raxml-exec` == "") stop("raxml-ng not found; pass --raxml-exec")
    getTrees(clones, build = "raxml",
             exec = opt$`raxml-exec`, nproc = opt$ncores,
             rseed = 42)
  },
  pml = {
    getTrees(clones, build = "pml", nproc = opt$ncores)
  },
  stop("unknown build backend: ", build)
)

rds_path <- file.path(opt$`out-dir`, "trees.rds")
saveRDS(trees, rds_path)
cat("[build_trees] wrote", rds_path, "\n")

newick_dir <- file.path(opt$`out-dir`, "newick")
dir.create(newick_dir, showWarnings = FALSE)
for (i in seq_len(nrow(trees))) {
  cid  <- trees$clone_id[i]
  tree <- trees$trees[[i]]
  if (is.null(tree)) next
  ape::write.tree(tree, file = file.path(newick_dir, paste0(cid, ".nwk")))
}
cat("[build_trees] wrote", length(list.files(newick_dir)), "Newick files to",
    newick_dir, "\n")

n_tips   <- vapply(trees$trees, function(t) if (is.null(t)) NA_integer_ else length(t$tip.label),
                   integer(1))
tree_len <- vapply(trees$trees, function(t) if (is.null(t)) NA_real_ else sum(t$edge.length),
                   numeric(1))
meta <- data.frame(clone_id = trees$clone_id, n_tips = n_tips, tree_len = tree_len)
write.table(meta,
            file = file.path(opt$`out-dir`, "tree_summary.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("[build_trees] done\n")
