#!/usr/bin/env Rscript
# Generate per-clone regime CSVs for LaVOUS / run-ou-poisson from the Dowser
# tree tibble produced by build_trees.R.
#
# For each clone:
#   1. Reroot the IgPhyML tree on the "Germline" tip, then drop it.
#   2. Ensure every internal node has a unique label.
#   3. Run parsimony reconstruction (Sankoff with a CSR cost matrix by default;
#      Fitch ACCTRAN with --method acctran) on the tip isotype trait to assign
#      an isotype to every internal node.
#   4. Write `regimes/<clone>.regime.csv` with one row per node (tip + internal).
#   5. Write the rerooted/pruned tree to `regimes/newick/<clone>.nwk` so node
#      names in the CSV match the tree LaVOUS will read.
#
# CSR-aware Sankoff cost matrix:
#   Class switch recombination is irreversible. We forbid back-switches by
#   assigning a large cost to any transition that moves upstream in the human
#   IGH locus (IGHM, IGHD, IGHG3, IGHG1, IGHA1, IGHG2, IGHG4, IGHE, IGHA2).
#   Forward (or same-state) transitions cost 0/1.
#
# Usage:
#   Rscript generate_regimes.R \
#       --trees trees_igphyml/trees.rds \
#       --airr  covid19.airr.tsv \
#       --trait isotype \
#       --method csr_sankoff \
#       --out-dir regimes_csr

suppressPackageStartupMessages({
  library(optparse)
  library(ape)
  library(phangorn)
})

opt_list <- list(
  make_option("--trees",   type = "character",
              help = "Dowser trees.rds from build_trees.R"),
  make_option("--airr",    type = "character",
              help = "AIRR TSV used to build the trees (for tip traits)"),
  make_option("--trait",   type = "character", default = "isotype",
              help = "Tip metadata column to reconstruct [default %default]"),
  make_option("--method",  type = "character", default = "csr_sankoff",
              help = "csr_sankoff (CSR-aware Sankoff) | acctran (unweighted Fitch) [default %default]"),
  make_option("--out-dir", type = "character", default = "regimes"),
  make_option("--root-tip", type = "character", default = "Germline",
              help = "Outgroup tip to root on and drop [default %default]"),
  make_option("--back-switch-cost", type = "numeric", default = 1e4,
              help = "Sankoff cost for biologically forbidden back-switches [default %default]")
)
opt <- parse_args(OptionParser(option_list = opt_list))
stopifnot(!is.null(opt$trees), !is.null(opt$airr))

dir.create(opt$`out-dir`, recursive = TRUE, showWarnings = FALSE)
nwk_dir <- file.path(opt$`out-dir`, "newick")
dir.create(nwk_dir, showWarnings = FALSE)

cat("[generate_regimes] loading", opt$trees, "\n")
trees <- readRDS(opt$trees)

cat("[generate_regimes] loading", opt$airr, "\n")
airr <- airr::read_rearrangement(opt$airr)
stopifnot(opt$trait %in% colnames(airr))
tip_trait <- setNames(as.character(airr[[opt$trait]]), airr$sequence_id)

# Use a single global level set so regime labels are consistent across clones.
trait_levels <- sort(unique(tip_trait))
cat("[generate_regimes] trait =", opt$trait,
    "(", length(trait_levels), "levels):",
    paste(trait_levels, collapse = ", "), "\n")
cat("[generate_regimes] method =", opt$method, "\n")

# 5'-to-3' order of the human IGH locus — order in which CSR is allowed.
# A B cell can switch from any isotype to any *downstream* one, but never back.
IGH_ORDER <- c("IGHM", "IGHD", "IGHG3", "IGHG1", "IGHA1",
               "IGHG2", "IGHG4", "IGHE", "IGHA2")

build_csr_cost <- function(levels, back_cost = 1e4) {
  # phangorn convention: cost[i, j] = cost of CHILD being state-i given PARENT
  # is state-j. So a back-switch (parent downstream, child upstream) is
  # cost[upstream_row, downstream_col] and gets back_cost.
  pos <- match(levels, IGH_ORDER)  # NA for unknown
  n <- length(levels)
  m <- matrix(1, nrow = n, ncol = n,
              dimnames = list(levels, levels))
  for (i in seq_len(n)) {       # i = child (row)
    for (j in seq_len(n)) {     # j = parent (col)
      if (i == j) { m[i, j] <- 0; next }
      if (is.na(pos[i]) || is.na(pos[j])) next
      if (pos[i] < pos[j]) m[i, j] <- back_cost  # child upstream of parent
    }
  }
  m
}

reroot_and_prune <- function(tree, outgroup) {
  if (!(outgroup %in% tree$tip.label)) return(NULL)
  tr <- ape::root(tree, outgroup = outgroup, resolve.root = TRUE)
  tr <- ape::drop.tip(tr, outgroup)
  # ape::drop.tip preserves remaining node.label; collapse any NA/dup labels.
  if (is.null(tr$node.label) ||
      length(tr$node.label) != tr$Nnode ||
      any(is.na(tr$node.label) | tr$node.label == "")) {
    tr <- ape::makeNodeLabel(tr, method = "number", prefix = "n")
  } else {
    tr$node.label <- make.unique(tr$node.label, sep = "_")
  }
  tr
}

reconstruct_states <- function(tree, tip_states, levels,
                               method = "csr_sankoff",
                               back_cost = 1e4) {
  # tip_states: named character vector keyed by tip label
  missing <- setdiff(tree$tip.label, names(tip_states))
  if (length(missing) > 0) {
    warning("missing trait for ", length(missing), " tips; assigning NA")
  }
  states <- tip_states[tree$tip.label]
  if (any(is.na(states))) {
    levels <- c(levels, "NA")
    states[is.na(states)] <- "NA"
  }
  mat <- matrix(states, ncol = 1)
  rownames(mat) <- tree$tip.label
  dat <- phyDat(mat, type = "USER", levels = levels)

  if (method == "csr_sankoff") {
    cost <- build_csr_cost(levels, back_cost = back_cost)
    # MPR with a cost matrix performs Sankoff parsimony.
    anc <- ancestral.pars(tree, dat, type = "MPR", cost = cost)
    # phangorn occasionally yields all-NaN rows at some internal nodes due to
    # probability normalization with high-cost rows. Fill those from ACCTRAN.
    anc_acc <- ancestral.pars(tree, dat, type = "ACCTRAN")
  } else if (method == "acctran") {
    anc <- ancestral.pars(tree, dat, type = "ACCTRAN")
    anc_acc <- anc
  } else {
    stop("unknown method: ", method)
  }

  # anc is a phyDat indexed 1..(Ntip+Nnode); each entry is a 1 x nlevels matrix
  # of state weights/probabilities. Pick argmax at every node.
  out <- character(length(anc))
  for (i in seq_along(anc)) {
    row <- anc[[i]][1, ]
    if (all(is.na(row)) || all(row == 0)) {
      row <- anc_acc[[i]][1, ]
    }
    out[i] <- levels[which.max(row)]
  }
  names(out) <- c(tree$tip.label, tree$node.label)
  out
}

n_total <- nrow(trees)
n_ok    <- 0
skip_log <- character()

for (i in seq_len(n_total)) {
  cid  <- trees$clone_id[i]
  tree <- trees$trees[[i]]
  if (is.null(tree)) {
    skip_log <- c(skip_log, paste(cid, "no tree")); next
  }
  tr <- tryCatch(reroot_and_prune(tree, opt$`root-tip`),
                 error = function(e) NULL)
  if (is.null(tr) || ape::Ntip(tr) < 2) {
    skip_log <- c(skip_log, paste(cid, "no outgroup or too small")); next
  }
  states <- tryCatch(reconstruct_states(tr, tip_trait, trait_levels,
                                        method = opt$method,
                                        back_cost = opt$`back-switch-cost`),
                     error = function(e) { message(cid, ": ", e$message); NULL })
  if (is.null(states)) {
    skip_log <- c(skip_log, paste(cid, "reconstruction failed")); next
  }
  df <- data.frame(node_name = names(states),
                   regime    = unname(states),
                   stringsAsFactors = FALSE)
  write.csv(df,
            file = file.path(opt$`out-dir`, paste0(cid, ".regime.csv")),
            row.names = FALSE, quote = FALSE)
  ape::write.tree(tr, file = file.path(nwk_dir, paste0(cid, ".nwk")))
  n_ok <- n_ok + 1
}

cat("[generate_regimes] wrote", n_ok, "/", n_total, "regime CSVs to",
    opt$`out-dir`, "\n")
if (length(skip_log) > 0) {
  log_path <- file.path(opt$`out-dir`, "skipped.log")
  writeLines(skip_log, log_path)
  cat("[generate_regimes] skipped", length(skip_log),
      "clones; see", log_path, "\n")
}
