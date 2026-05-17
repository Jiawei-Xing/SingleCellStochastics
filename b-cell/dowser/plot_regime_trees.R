#!/usr/bin/env Rscript
# Circular tree plots for each clonotype, colored by regime (isotype).
#
# Reads each rerooted/pruned Newick from `regimes/newick/<clone>.nwk` plus the
# matching `regimes/<clone>.regime.csv` and produces:
#   - regimes/plots/<clone>.pdf       one circular tree per clone
#   - regimes/plots_all.pdf           multi-page PDF (all clones, biggest first)
#
# Usage:
#   Rscript plot_regime_trees.R --regime-dir regimes --out-dir regimes/plots

suppressPackageStartupMessages({
  library(optparse)
  library(ape)
  library(ggplot2)
  library(ggtree)
})

opt_list <- list(
  make_option("--regime-dir", type = "character", default = "regimes"),
  make_option("--tree-dir",   type = "character", default = NULL,
              help = "Directory containing Newick files [default: <regime-dir>/newick]"),
  make_option("--regime-suffix", type = "character", default = ".regime.csv",
              help = "Suffix for regime CSV files [default %default]"),
  make_option("--newick-suffix", type = "character", default = ".nwk",
              help = "Suffix for Newick files [default %default]"),
  make_option("--out-dir",    type = "character", default = "regimes/plots"),
  make_option("--combined",   type = "character", default = "regimes/plots_all.pdf",
              help = "Path for multi-page combined PDF")
)
opt <- parse_args(OptionParser(option_list = opt_list))
dir.create(opt$`out-dir`, recursive = TRUE, showWarnings = FALSE)

escape_regex <- function(x) {
  gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", x)
}

tree_dir <- opt$`tree-dir`
if (is.null(tree_dir) || tree_dir == "") {
  tree_dir <- file.path(opt$`regime-dir`, "newick")
}

regime_suffix_re <- paste0(escape_regex(opt$`regime-suffix`), "$")
csv_files <- list.files(opt$`regime-dir`, pattern = regime_suffix_re, full.names = TRUE)
clones <- sub(regime_suffix_re, "", basename(csv_files))
cat("[plot] found", length(clones), "clones\n")

# Pre-load to compute size + global palette
load_one <- function(clone) {
  nwk <- file.path(tree_dir, paste0(clone, opt$`newick-suffix`))
  csv <- file.path(opt$`regime-dir`, paste0(clone, opt$`regime-suffix`))
  if (!file.exists(nwk) || !file.exists(csv)) return(NULL)
  tr  <- ape::read.tree(nwk)
  if (is.null(tr) || ape::Ntip(tr) < 2) return(NULL)
  rg  <- read.csv(csv, stringsAsFactors = FALSE)
  list(clone = clone, tree = tr, regime = rg, n_tips = ape::Ntip(tr))
}

loaded <- Filter(Negate(is.null), lapply(clones, load_one))
cat("[plot] loaded", length(loaded), "clones\n")
loaded <- loaded[order(-vapply(loaded, function(x) x$n_tips, integer(1)))]

all_levels <- sort(unique(unlist(lapply(loaded, function(x) x$regime$regime))))
# Stable color palette for isotypes
isotype_palette <- c(
  IGHM  = "#1f77b4",
  IGHG3 = "#9467bd",
  IGHG1 = "#2ca02c",
  IGHG2 = "#17becf",
  IGHG4 = "#bcbd22",
  IGHA1 = "#d62728",
  IGHA2 = "#ff7f0e",
  IGHE  = "#e377c2"
)
pal <- isotype_palette[all_levels]
pal[is.na(pal)] <- "grey60"
names(pal)[is.na(names(pal))] <- all_levels[is.na(names(isotype_palette[all_levels]))]
pal <- pal[!is.na(pal) & names(pal) != ""]
cat("[plot] regime levels:", paste(all_levels, collapse = ", "), "\n")

plot_one <- function(item) {
  tr  <- item$tree
  rg  <- item$regime
  # Build node-indexed data frame: tips first (in tip.label order), then internals
  # in node.label order. This matches ggtree's node ids (1..Ntip = tips,
  # Ntip+1..Ntip+Nnode = internals).
  node_names <- c(tr$tip.label, tr$node.label)
  m <- match(node_names, rg$node_name)
  dat <- data.frame(node   = seq_along(node_names),
                    label  = node_names,
                    regime = rg$regime[m],
                    stringsAsFactors = FALSE)
  p <- ggtree(tr, layout = "circular", size = 0.4) %<+% dat +
    aes(color = regime) +
    geom_tippoint(aes(color = regime), size = 1.2) +
    scale_color_manual(values = pal, na.value = "grey70",
                       breaks = names(pal), drop = FALSE) +
    ggtitle(sprintf("%s  (n=%d tips)", item$clone, item$n_tips)) +
    theme(legend.position = "right",
          plot.title = element_text(size = 10))
  p
}

# Per-clone PDFs
for (item in loaded) {
  p <- plot_one(item)
  ggsave(file.path(opt$`out-dir`, paste0(item$clone, ".pdf")),
         p, width = 6, height = 5, useDingbats = FALSE)
}
cat("[plot] wrote", length(loaded), "per-clone PDFs to", opt$`out-dir`, "\n")

# Combined multi-page PDF (biggest first)
pdf(opt$combined, width = 6, height = 5, useDingbats = FALSE)
for (item in loaded) print(plot_one(item))
invisible(dev.off())
cat("[plot] wrote combined PDF:", opt$combined, "\n")
