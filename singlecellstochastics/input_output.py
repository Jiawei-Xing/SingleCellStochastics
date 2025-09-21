import csv


def write_read_counts(
    read_counts: dict[int, dict[str, int]],
    cells: list[str],
    n_genes: int,
    output_dir: str,
    label: str,
) -> None:
    """
    Writes read count data for multiple cells and genes to a TSV file.

    The output file will have a header row with gene indices and each subsequent row will contain
    the cell identifier followed by the read counts for each gene.

    Args:
        read_counts (dict): A dictionary where keys are gene indices (int) and values are dictionaries
            mapping cell identifiers to read counts (int).
        cells (iterable): An iterable of cell identifiers to include in the output.
        n_genes (int): The number of genes (columns) to write for each cell.
        output_dir (str): The directory where the output file will be saved.
        label (str): A label to include in the output filename.

    Output:
        Creates a TSV file named 'readcounts_{label}.tsv' in the specified output directory.
    """
    with open(f"{output_dir}/readcounts_{label}.tsv", "w") as f:
        # Write header for all gene names
        f.write("\t" + "\t".join(str(i) for i in range(1, n_genes + 1)) + "\n")
        # Write read counts for each cell for all genes
        for cell in cells:
            counts = [str(read_counts[gene][cell]) for gene in range(n_genes)]
            f.write(f"{cell}\t" + "\t".join(counts) + "\n")


def load_read_count_tsv(file_path: str) -> dict[str, dict[str, int]]:
    """
    Load read count data from a TSV file into a nested dictionary.

    Args:
        file_path (str): Path to the TSV file containing read count data.

    Returns:
        dict: A dictionary where keys are gene names (str) and values are dictionaries
              mapping cell names (str) to read counts (int).
    """
    read_count_data = {}
    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames
        for col in columns:
            if col:
                read_count_data[str(col)] = {}
        for row in reader:
            row_name = row[columns[0]]
            for col in columns[1:]:
                read_count_data[col][row_name] = int(row[col])

    return read_count_data
