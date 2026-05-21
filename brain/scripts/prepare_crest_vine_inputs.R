#!/usr/bin/env Rscript

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(
    rds = "data/raw/GSE210139_single_cell_CREST_barcode_qc.rds",
    outdir = "data/vine",
    datasets = "aggregate",
    unit = "clone",
    min_clone_size = "1",
    min_dominant_fraction = "0",
    missing_cassette = "missing"
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key)
    }
    name <- gsub("-", "_", substring(key, 3), fixed = TRUE)
    if (!name %in% names(opts)) {
      stop("Unknown option: ", key)
    }
    if (i == length(args)) {
      stop("Missing value for ", key)
    }
    opts[[name]] <- args[[i + 1]]
    i <- i + 2
  }
  opts$min_clone_size <- as.integer(opts$min_clone_size)
  opts$min_dominant_fraction <- as.numeric(opts$min_dominant_fraction)
  if (!opts$unit %in% c("clone", "cell")) {
    stop("--unit must be one of: clone, cell")
  }
  if (is.na(opts$min_dominant_fraction) || opts$min_dominant_fraction < 0 || opts$min_dominant_fraction > 1) {
    stop("--min-dominant-fraction must be a number in [0, 1]")
  }
  if (!opts$missing_cassette %in% c("missing", "zero")) {
    stop("--missing-cassette must be one of: missing, zero")
  }
  opts
}

safe_id <- function(x) {
  x <- gsub("[^A-Za-z0-9_.-]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  x[x == ""] <- "unnamed"
  x
}

select_datasets <- function(all_names, spec) {
  aggregate <- c("E11", "E15", "pandaE11", "OGN")
  if (spec == "aggregate") {
    return(intersect(aggregate, all_names))
  }
  if (spec == "replicates") {
    return(setdiff(all_names, aggregate))
  }
  if (spec == "all") {
    return(all_names)
  }
  requested <- strsplit(spec, ",", fixed = TRUE)[[1]]
  requested <- trimws(requested)
  missing <- setdiff(requested, all_names)
  if (length(missing) > 0) {
    stop("Datasets not present in RDS: ", paste(missing, collapse = ", "))
  }
  requested
}

split_pattern <- function(pattern, n_sites, missing_value) {
  if (is.na(pattern) || pattern == "NA") {
    return(rep(missing_value, n_sites))
  }
  parts <- strsplit(pattern, "_", fixed = TRUE)[[1]]
  if (length(parts) != n_sites) {
    stop("Expected ", n_sites, " sites but found ", length(parts), " in pattern: ", pattern)
  }
  parts
}

build_token_matrix <- function(df, missing_cassette) {
  missing_value <- if (missing_cassette == "missing") "-1" else "NONE"
  v1 <- t(vapply(df$pattern.v1, split_pattern, character(9), n_sites = 9, missing_value = missing_value))
  v2 <- t(vapply(df$pattern.v2, split_pattern, character(8), n_sites = 8, missing_value = missing_value))
  tokens <- cbind(v1, v2)
  colnames(tokens) <- c(paste0("v1_r", seq_len(9)), paste0("v2_r", seq_len(8)))

  # Long deletions are repeated across adjacent sites in CREST strings. Keep the
  # first edited site and mark repeated adjacent copies as missing, matching the
  # companion paper's Cassiopeia conversion.
  for (i in seq_len(nrow(tokens))) {
    for (idx in list(seq_len(9), 9 + seq_len(8))) {
      for (k in 2:length(idx)) {
        prev <- tokens[i, idx[k - 1]]
        cur <- tokens[i, idx[k]]
        if (!prev %in% c("NONE", "-1") && identical(cur, prev)) {
          tokens[i, idx[k]] <- "-1"
        }
      }
    }
  }

  tokens
}

encode_crispr_matrix <- function(tokens) {
  encoded <- matrix(-1L, nrow = nrow(tokens), ncol = ncol(tokens))
  colnames(encoded) <- paste0("r", seq_len(ncol(tokens)))
  state_maps <- list()

  for (j in seq_len(ncol(tokens))) {
    column <- tokens[, j]
    encoded[column == "NONE", j] <- 0L
    encoded[column == "-1", j] <- -1L
    edited <- sort(unique(column[!column %in% c("NONE", "-1")]))
    if (length(edited) > 0) {
      map <- setNames(seq_along(edited), edited)
      for (state in names(map)) {
        encoded[column == state, j] <- map[[state]]
      }
      state_maps[[colnames(tokens)[j]]] <- data.frame(
        site = colnames(tokens)[j],
        state = as.integer(map),
        scar = names(map),
        stringsAsFactors = FALSE
      )
    }
  }

  list(matrix = as.data.frame(encoded), state_maps = do.call(rbind, state_maps))
}

collapse_to_clones <- function(df, dataset, min_clone_size, min_dominant_fraction) {
  clone_sizes <- sort(table(df$pattern), decreasing = TRUE)
  clone_sizes <- clone_sizes[clone_sizes >= min_clone_size]
  clone_patterns <- names(clone_sizes)
  rows <- lapply(seq_along(clone_patterns), function(i) {
    pattern <- clone_patterns[[i]]
    cells <- df[df$pattern == pattern, , drop = FALSE]
    cell_type_counts <- sort(table(cells$Cell.type), decreasing = TRUE)
    dominant <- names(cell_type_counts)[[1]]
    rep_row <- cells[1, , drop = FALSE]
    rep_row$leaf_id <- sprintf("%s_clone%05d", safe_id(dataset), i)
    rep_row$clone_id <- rep_row$leaf_id
    rep_row$clone_size <- as.integer(clone_sizes[[i]])
    rep_row$dominant_cell_type <- dominant
    rep_row$dominant_fraction <- as.numeric(cell_type_counts[[1]]) / sum(cell_type_counts)
    rep_row$cell_type_counts <- paste(
      paste(names(cell_type_counts), as.integer(cell_type_counts), sep = ":"),
      collapse = ";"
    )
    rep_row
  })
  rows <- do.call(rbind, rows)
  rows[rows$dominant_fraction >= min_dominant_fraction, , drop = FALSE]
}

make_cell_rows <- function(df, dataset) {
  df$leaf_id <- make.unique(safe_id(paste(dataset, df$Cell.BC, sep = ".")))
  df$clone_id <- safe_id(paste(dataset, match(df$pattern, unique(df$pattern)), sep = "_clone"))
  df$clone_size <- as.integer(table(df$pattern)[df$pattern])
  df$dominant_cell_type <- df$Cell.type
  df$dominant_fraction <- 1
  df$cell_type_counts <- df$Cell.type
  df
}

write_group <- function(rows, dataset, unit, outdir, missing_cassette) {
  prefix <- paste(safe_id(dataset), unit, sep = "_")
  tokens <- build_token_matrix(rows, missing_cassette)
  encoded <- encode_crispr_matrix(tokens)
  matrix_df <- encoded$matrix
  rownames(matrix_df) <- rows$leaf_id

  matrix_path <- file.path(outdir, paste0(prefix, "_character_matrix.tsv"))
  labels_path <- file.path(outdir, paste0(prefix, "_statelabels.csv"))
  metadata_path <- file.path(outdir, paste0(prefix, "_metadata.tsv"))
  states_path <- file.path(outdir, paste0(prefix, "_scar_state_map.tsv"))

  write.table(
    matrix_df,
    file = matrix_path,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = NA
  )

  labels <- data.frame(
    leaf_id = rows$leaf_id,
    state = ifelse(is.na(rows$dominant_cell_type) | rows$dominant_cell_type == "", "unknown", rows$dominant_cell_type),
    stringsAsFactors = FALSE
  )
  write.table(labels, file = labels_path, sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)

  metadata <- rows[, c(
    "leaf_id", "clone_id", "Cell.BC", "pattern", "pattern.v1", "pattern.v2",
    "Cell.type", "sample", "clone_size", "dominant_cell_type", "dominant_fraction",
    "cell_type_counts"
  )]
  write.table(metadata, file = metadata_path, sep = "\t", quote = FALSE, row.names = FALSE)

  if (!is.null(encoded$state_maps) && nrow(encoded$state_maps) > 0) {
    write.table(encoded$state_maps, file = states_path, sep = "\t", quote = FALSE, row.names = FALSE)
  } else {
    write.table(data.frame(site = character(), state = integer(), scar = character()), file = states_path, sep = "\t", quote = FALSE, row.names = FALSE)
  }

  data.frame(
    dataset = dataset,
    unit = unit,
    matrix = matrix_path,
    statelabels = labels_path,
    metadata = metadata_path,
    state_map = states_path,
    n_leaves = nrow(matrix_df),
    n_sites = ncol(matrix_df),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  opts <- parse_args()
  dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)
  crest <- readRDS(opts$rds)
  datasets <- select_datasets(names(crest), opts$datasets)

  manifest <- list()
  for (dataset in datasets) {
    df <- crest[[dataset]]
    df$Cell.type[is.na(df$Cell.type) | df$Cell.type == ""] <- "unknown"
        if (opts$unit == "clone") {
            rows <- collapse_to_clones(
                df,
                dataset,
                opts$min_clone_size,
                opts$min_dominant_fraction
            )
        } else {
      rows <- make_cell_rows(df, dataset)
    }
    if (nrow(rows) < 3) {
      warning("Skipping ", dataset, ": fewer than 3 leaves after filtering")
      next
    }
    manifest[[length(manifest) + 1]] <- write_group(
      rows = rows,
      dataset = dataset,
      unit = opts$unit,
      outdir = opts$outdir,
      missing_cassette = opts$missing_cassette
    )
    message(dataset, ": wrote ", nrow(rows), " ", opts$unit, " leaves")
  }

  manifest_df <- do.call(rbind, manifest)
  manifest_path <- file.path(opts$outdir, paste0("crest_", opts$unit, "_manifest.tsv"))
  write.table(manifest_df, file = manifest_path, sep = "\t", quote = FALSE, row.names = FALSE)
  message("Manifest: ", manifest_path)
}

main()
