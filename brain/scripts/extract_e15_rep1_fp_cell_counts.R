#!/usr/bin/env Rscript

# Export all E15 rep1 FP-lineage cells from the Seurat RNA count matrix.

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(
    seurat = "data/raw/GSE210139_E15_seurat_final_20230406.rds",
    barcode = "data/raw/GSE210139_single_cell_CREST_barcode_qc.rds",
    fp_map = "scripts/fp_lineage_groups.tsv",
    sample = "E15_rep1",
    outdir = "outputs/e15_rep1_fp_trajectory_velocity",
    prefix = "e15_rep1_fp_cells"
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

  message("Reading FP cell type map: ", opts$fp_map)
  fp_map <- read.delim(opts$fp_map, stringsAsFactors = FALSE, check.names = FALSE)
  missing_map_cols <- setdiff(c("cell_type", "fp_regime"), colnames(fp_map))
  if (length(missing_map_cols) > 0) {
    stop("FP map is missing columns: ", paste(missing_map_cols, collapse = ", "))
  }

  message("Reading CREST cell annotations: ", opts$barcode)
  crest <- readRDS(opts$barcode)
  if (!opts$sample %in% names(crest)) stop(opts$sample, " is not present in barcode RDS")
  cells <- crest[[opts$sample]]
  required <- c("Cell.BC", "Cell.type", "sample")
  missing_cols <- setdiff(required, colnames(cells))
  if (length(missing_cols) > 0) {
    stop("Cell annotation table is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  cells <- cells[cells$Cell.type %in% fp_map$cell_type, , drop = FALSE]
  cells$fp_regime <- fp_map$fp_regime[match(cells$Cell.type, fp_map$cell_type)]
  cells$selection <- "all_fp_cells_by_cell_type"
  if (anyDuplicated(cells$Cell.BC)) {
    warning("Dropping duplicated Cell.BC rows from cell annotation table")
    cells <- cells[!duplicated(cells$Cell.BC), , drop = FALSE]
  }
  if (nrow(cells) == 0) stop("No FP-lineage cells found for sample: ", opts$sample)

  message("Reading Seurat object: ", opts$seurat)
  seu <- readRDS(opts$seurat)
  counts <- get_assay_counts(seu, "RNA")

  present <- cells$Cell.BC %in% colnames(counts)
  if (!all(present)) {
    warning(sum(!present), " FP cells were absent from the RNA count matrix and will be dropped")
    cells <- cells[present, , drop = FALSE]
  }
  if (nrow(cells) == 0) stop("No FP cells overlap the RNA count matrix")

  counts <- counts[, cells$Cell.BC, drop = FALSE]
  cell_gene_counts <- Matrix::t(counts)

  obs <- cells
  obs$n_counts <- as.numeric(Matrix::rowSums(cell_gene_counts))
  obs$n_genes <- as.numeric(Matrix::rowSums(cell_gene_counts > 0))

  mtx_path <- file.path(opts$outdir, paste0(opts$prefix, "_counts.mtx"))
  obs_path <- file.path(opts$outdir, paste0(opts$prefix, "_obs.tsv"))
  genes_path <- file.path(opts$outdir, paste0(opts$prefix, "_genes.tsv"))

  message("Writing sparse count matrix: ", mtx_path)
  Matrix::writeMM(cell_gene_counts, mtx_path)

  message("Writing cell metadata: ", obs_path)
  write.table(obs, file = obs_path, sep = "\t", quote = FALSE, row.names = FALSE)

  genes <- data.frame(gene_name = rownames(counts), stringsAsFactors = FALSE)
  message("Writing gene metadata: ", genes_path)
  write.table(genes, file = genes_path, sep = "\t", quote = FALSE, row.names = FALSE)

  message("Cells: ", nrow(cell_gene_counts))
  message("Genes: ", ncol(cell_gene_counts))
  message("Nonzero entries: ", length(cell_gene_counts@x))
}

main()
