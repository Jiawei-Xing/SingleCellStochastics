#!/usr/bin/env Rscript
# Run SCOUT on all 12 simulations used in the ROC comparison.
#
# Usage:  conda activate scout
#         Rscript run_scout.R [cores]
#
# Output: scout/scout_<label>.csv  for each simulation label

library(SCOUT)
library(ape)
library(dplyr)
library(future)
library(future.apply)
library(progressr)

args <- commandArgs(trailingOnly = TRUE)
cores <- if (length(args) >= 1) as.integer(args[1]) else 4

# ---------- paths ----------------------------------------------------------
base      <- "/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/examples"
sim_dir   <- file.path(base, "simulation")
input_dir <- file.path(base, "input_data")
out_dir   <- file.path(base, "scout")
dir.create(out_dir, showWarnings = FALSE)

tree_path   <- file.path(input_dir, "tree_egx.nwk")
regime_path <- file.path(input_dir, "regime.csv")

# Read regime mapping (node_name -> 0/1)
regime_df <- read.csv(regime_path, stringsAsFactors = FALSE)

# Set up parallel backend
plan(multisession, workers = cores)

# The 12 simulations used in plot_diff_roc.py
labels <- c("A", "C", "C_diff",
            "I", "I_diff", "J", "J_diff",
            "t5_r0", "t5_r50",
            "t7_r0", "t7_r50", "t7_r5")

for (label in labels) {
  outfile <- file.path(out_dir, paste0("scout_", label, ".csv"))
  if (file.exists(outfile)) {
    message(sprintf("Skipping %s (already exists)", label))
    next
  }

  rc_file <- file.path(sim_dir, paste0("readcounts_", label, ".tsv"))
  if (!file.exists(rc_file)) {
    warning(sprintf("Missing: %s — skipping", rc_file))
    next
  }

  message(sprintf("=== Running SCOUT on %s ===", label))

  # Read expression: rows = cells (tip labels), cols = genes
  expr <- read.csv(rc_file, sep = "\t", row.names = 1, check.names = FALSE)

  # Build metadata: species + regime + gene columns
  tips <- rownames(expr)
  regime_map <- setNames(regime_df$regime, regime_df$node_name)
  reg_vals <- regime_map[tips]
  # Tips not in regime file default to 0
  reg_vals[is.na(reg_vals)] <- 0

  meta <- data.frame(species = tips, regime = as.character(reg_vals),
                     stringsAsFactors = FALSE)
  # Rename gene columns to be safe R names
  gene_names <- paste0("g", colnames(expr))
  colnames(expr) <- gene_names
  meta <- cbind(meta, expr)

  # Write temp metadata
  tmp_meta <- tempfile(fileext = ".csv")
  write.csv(meta, tmp_meta, row.names = TRUE)

  # Run SCOUT pipeline
  tryCatch({
    inputs <- prepare_data(
      tree_path     = tree_path,
      metadata_path = tmp_meta,
      outpath       = out_dir,
      species_key   = "species",
      regimes       = c("BM1", "OU1", "regime"),
      normalize     = TRUE,
      smoothing     = NULL,
      name          = label
    )

    results <- fitModel(inputs, cores = cores, write = FALSE)
    metrics <- get_fit_metrics(results)

    # Extract per-gene summary: for each gene, get the AICc weight of OUM
    # (regime-specific model) as the "differential expression" score
    if (is.list(metrics)) {
      res_df <- metrics[["results"]]
    } else {
      res_df <- metrics
    }

    # Filter to the regime-specific model results
    oum_rows <- res_df[res_df$regime == "regime", ]

    # Per gene: extract OUM AICc weight and delta AICc vs best model
    gene_scores <- oum_rows %>%
      select(quant_trait, loglik, AICc, delta_aicc, aicc_weight, alpha, sigma.sq) %>%
      rename(gene = quant_trait, OUM_loglik = loglik, OUM_AICc = AICc,
             OUM_delta_aicc = delta_aicc, OUM_aicc_weight = aicc_weight,
             OUM_alpha = alpha, OUM_sigma_sq = sigma.sq)

    # Strip the "g" prefix from gene names
    gene_scores$gene <- sub("^g", "", gene_scores$gene)

    write.csv(gene_scores, outfile, row.names = FALSE)
    message(sprintf("  -> Wrote %s (%d genes)", outfile, nrow(gene_scores)))

  }, error = function(e) {
    message(sprintf("  ERROR on %s: %s", label, conditionMessage(e)))
  })

  unlink(tmp_meta)
}

message("Done.")
