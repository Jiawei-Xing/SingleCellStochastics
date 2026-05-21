#!/usr/bin/env Rscript

# Extract per-clone read counts and library-size factors from a Seurat object,
# using one representative cell per CREST barcode pattern (idxmax of raw total
# UMI). Works for any clone metadata that has columns leaf_id, pattern, and
# clone_size (e.g. data/fp_lineage/... or data/vine_reps_homogeneous/...).

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(
    seurat = "data/raw/GSE210139_E15_seurat_final_20230406.rds",
    barcode = "data/raw/GSE210139_single_cell_CREST_barcode_qc.rds",
    metadata = "data/fp_lineage/E15_rep1_fp_lineage_homogeneous_clone_metadata.tsv",
    outdir = "data/lavous/e15_rep1_fp_lineage",
    prefix = "E15_rep1_fp_lineage_homogeneous_clone",
    sample = "E15_rep1",
    min_total = "1",
    min_leaf_fraction = "0.10",
    match_dominant = "TRUE"
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    name <- gsub("-", "_", substring(key, 3), fixed = TRUE)
    if (!name %in% names(opts)) stop("Unknown option: ", key)
    if (i == length(args)) stop("Missing value for ", key)
    opts[[name]] <- args[[i + 1]]
    i <- i + 2
  }
  opts$min_total <- as.numeric(opts$min_total)
  opts$min_leaf_fraction <- as.numeric(opts$min_leaf_fraction)
  opts$match_dominant <- isTRUE(as.logical(opts$match_dominant))
  opts
}

get_assay_counts <- function(seurat_obj, assay_name = "RNA") {
  assays <- attributes(seurat_obj)[["assays"]]
  if (is.null(assays) || !assay_name %in% names(assays)) {
    stop("Assay not found in Seurat object: ", assay_name)
  }
  counts <- attributes(assays[[assay_name]])[["counts"]]
  if (is.null(counts)) stop("Counts slot is missing for assay: ", assay_name)
  counts
}

main <- function() {
  opts <- parse_args()
  suppressPackageStartupMessages(library(Matrix))
  dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)

  message("Reading clone metadata: ", opts$metadata)
  clone_meta <- read.delim(opts$metadata, stringsAsFactors = FALSE, check.names = FALSE)
  required_clone_cols <- c("leaf_id", "pattern", "clone_size")
  if (opts$match_dominant) {
    required_clone_cols <- c(required_clone_cols, "dominant_cell_type")
  }
  missing_clone_cols <- setdiff(required_clone_cols, colnames(clone_meta))
  if (length(missing_clone_cols) > 0) {
    stop("Clone metadata is missing columns: ", paste(missing_clone_cols, collapse = ", "))
  }
  if (anyDuplicated(clone_meta$leaf_id)) stop("Duplicate leaf_id values in clone metadata")
  if (anyDuplicated(clone_meta$pattern)) stop("Duplicate pattern values in clone metadata")
  leaf_ids <- clone_meta$leaf_id

  message("Reading barcode metadata: ", opts$barcode)
  crest <- readRDS(opts$barcode)
  if (!opts$sample %in% names(crest)) stop(opts$sample, " is not present in barcode RDS")
  barcode <- crest[[opts$sample]]
  barcode <- barcode[, c("Cell.BC", "pattern", "Cell.type", "sample"), drop = FALSE]
  barcode <- barcode[barcode$pattern %in% clone_meta$pattern, , drop = FALSE]
  barcode <- barcode[!is.na(barcode$Cell.BC), , drop = FALSE]

  message("Reading Seurat object: ", opts$seurat)
  seu <- readRDS(opts$seurat)
  counts <- get_assay_counts(seu, "RNA")
  seurat_meta <- attributes(seu)[["meta.data"]]

  in_counts <- barcode$Cell.BC %in% colnames(counts)
  if (!all(in_counts)) {
    warning(sum(!in_counts), " clone-member cells were not present in RNA counts and will be dropped")
    barcode <- barcode[in_counts, , drop = FALSE]
  }
  if (nrow(barcode) == 0) stop("No clone-member cells overlap the RNA count matrix")

  # Cell-level total UMI from the raw count matrix (only for cells we care about).
  cell_totals <- Matrix::colSums(counts[, barcode$Cell.BC, drop = FALSE])
  barcode$total_counts <- as.numeric(cell_totals[barcode$Cell.BC])

  # Optionally restrict candidate cells to the clone's dominant cell type
  # before idxmax. Avoids picking off-type cells in mixed clones; no-op for
  # homogeneous clones where every cell already matches.
  if (opts$match_dominant) {
    barcode$dominant_cell_type <- clone_meta$dominant_cell_type[
      match(barcode$pattern, clone_meta$pattern)
    ]
    keep <- !is.na(barcode$Cell.type) & barcode$Cell.type == barcode$dominant_cell_type
    n_drop <- sum(!keep)
    if (n_drop > 0) message("Dropping ", n_drop, " off-dominant-type cells before idxmax")
    barcode <- barcode[keep, , drop = FALSE]
    patterns_remaining <- unique(barcode$pattern)
    missing_patterns <- setdiff(clone_meta$pattern, patterns_remaining)
    if (length(missing_patterns) > 0) {
      stop(
        length(missing_patterns), " clones have no cell matching dominant_cell_type ",
        "(first 5 patterns: ", paste(head(missing_patterns, 5), collapse = ", "), ")"
      )
    }
  }

  # For each pattern (= clone), pick the cell with the highest total_counts.
  # Ties resolved by sorting Cell.BC (stable; deterministic).
  ord <- order(barcode$pattern, -barcode$total_counts, barcode$Cell.BC)
  barcode <- barcode[ord, , drop = FALSE]
  rep_idx <- !duplicated(barcode$pattern)
  rep_cells <- barcode[rep_idx, , drop = FALSE]

  # Attach leaf_id to each representative cell via pattern.
  rep_cells$leaf_id <- clone_meta$leaf_id[match(rep_cells$pattern, clone_meta$pattern)]
  rep_cells <- rep_cells[order(match(rep_cells$leaf_id, leaf_ids)), , drop = FALSE]

  missing_leaves <- setdiff(leaf_ids, rep_cells$leaf_id)
  if (length(missing_leaves) > 0) {
    stop(
      length(missing_leaves), " leaf_ids have no representative cell after Seurat intersection: ",
      paste(head(missing_leaves, 5), collapse = ", ")
    )
  }

  # Build the leaf × gene count matrix from the representative cells only.
  rep_counts <- t(counts[, rep_cells$Cell.BC, drop = FALSE])
  rownames(rep_counts) <- rep_cells$leaf_id

  gene_names <- rownames(counts)
  if (anyDuplicated(gene_names)) {
    message("Summing ", length(gene_names) - length(unique(gene_names)), " duplicate gene-name columns")
    unique_genes <- unique(gene_names)
    gene_incidence <- sparseMatrix(
      i = seq_along(gene_names),
      j = match(gene_names, unique_genes),
      x = 1,
      dims = c(length(gene_names), length(unique_genes)),
      dimnames = list(gene_names, unique_genes)
    )
    rep_counts <- rep_counts %*% gene_incidence
    colnames(rep_counts) <- unique_genes
  } else {
    colnames(rep_counts) <- gene_names
  }

  raw_library_size <- Matrix::rowSums(rep_counts)

  gene_totals <- Matrix::colSums(rep_counts)
  gene_leaf_counts <- Matrix::colSums(rep_counts > 0)
  min_leaves <- ceiling(opts$min_leaf_fraction * nrow(rep_counts))
  keep_genes <- (gene_totals >= opts$min_total) & (gene_leaf_counts >= min_leaves)
  rep_counts <- rep_counts[, keep_genes, drop = FALSE]

  median_total <- stats::median(raw_library_size)
  if (median_total <= 0) stop("Median raw library size is non-positive")
  library_factor <- as.numeric(raw_library_size) / median_total
  names(library_factor) <- names(raw_library_size)

  # Per-leaf summary table.
  summary_df <- merge(
    clone_meta,
    data.frame(
      leaf_id = rep_cells$leaf_id,
      rep_cell = rep_cells$Cell.BC,
      rep_cell_total = rep_cells$total_counts,
      stringsAsFactors = FALSE
    ),
    by = "leaf_id",
    all.x = TRUE,
    sort = FALSE
  )
  summary_df$library_size <- as.numeric(raw_library_size[summary_df$leaf_id])
  summary_df$library_factor <- as.numeric(library_factor[summary_df$leaf_id])

  counts_path <- file.path(opts$outdir, paste0(opts$prefix, "_readcounts.tsv"))
  library_path <- file.path(opts$outdir, paste0(opts$prefix, "_library.tsv"))
  rep_cells_path <- file.path(opts$outdir, paste0(opts$prefix, "_rep_cells.tsv"))
  summary_path <- file.path(opts$outdir, paste0(opts$prefix, "_readcount_summary.tsv"))

  message("Writing read counts: ", counts_path)
  write.table(
    as.matrix(rep_counts),
    file = counts_path,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = NA
  )
  library_df <- data.frame(library_factor = library_factor)
  rownames(library_df) <- names(library_factor)
  write.table(
    library_df,
    file = library_path,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = FALSE
  )
  write.table(rep_cells, file = rep_cells_path, sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(summary_df, file = summary_path, sep = "\t", quote = FALSE, row.names = FALSE)

  message("Leaves: ", nrow(rep_counts))
  message("Representative cells selected: ", nrow(rep_cells))
  message(
    "Genes kept (total >= ", opts$min_total,
    " AND nonzero in >= ", min_leaves, "/", nrow(rep_counts),
    " leaves [", opts$min_leaf_fraction * 100, "%]): ",
    ncol(rep_counts)
  )
  message("Total UMI in kept matrix: ", sum(Matrix::rowSums(rep_counts)))
  message(
    "Library factor: median = ", format(stats::median(library_factor), digits = 4),
    ", range = ", format(min(library_factor), digits = 3), " - ",
    format(max(library_factor), digits = 3),
    " (median raw lib = ", format(median_total, big.mark = ","), ")"
  )
  if (!is.null(seurat_meta)) {
    message("Seurat metadata cells: ", nrow(seurat_meta))
  }
}

main()
