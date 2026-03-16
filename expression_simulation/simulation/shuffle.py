import pandas as pd

simulations = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Read the file. 
# index_col=0 tells Pandas to use the very first column as the row names.
for n in simulations:
    df = pd.read_csv(f"readcounts_{n}.tsv", sep="\t", index_col=0)

    # .values ensures the data moves, but the row names (index) stay locked in place
    shuffled_columns_df = df.apply(lambda col: col.sample(frac=1).values)

    shuffled_columns_df.to_csv(f"./readcounts_{n}_shuffled.tsv", sep="\t", header=True, index=True)
