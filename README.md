# SingleCellStochastics
* Stochastic modeling of single-cell gene expression dynamics along cell lineage using Ornstein-Uhlenbeck process and Poisson observation.
* Detecting lineage-specific differential gene expressions by hypothesis testing on stochastic expression models.
* Reconstructing possible ancestral cell states with deep generative models along cell lineage.

## Run example
Create and activate required conda environment:
```
conda env create -n OUP -f env/OUP.yml
conda activate OUP
```
Run examples:
```
# run positive simulation
python src/OUP.py --tree examples/input_data/tree.nwk --expr examples/input_data/readcounts_OU_tip5-10_downsampled.tsv --regime examples/input_data/regime_tip.csv --null 0 --output examples/output_results/test_ou

# run negative control
python src/OUP.py --tree examples/input_data/tree.nwk --expr examples/input_data/readcounts_BM_tip5-10_downsampled.tsv --regime examples/input_data/regime_tip.csv --null 0 --output examples/output_results/test_bm
```
Using GPU and wandb for optimization is recommended.

## Lineage simulation
Example cell lineages can be simulated with a modified version of the agent-based cancer cell simulator from [MACHINA](https://www.nature.com/articles/s41588-018-0106-z). 
### Required environment
Create and activate the required conda environment with `Lineage_simulation/env/lineage_sim.yml`:
```
conda env create -n simulate -f Lineage_simulation/env/lineage_sim.yml
conda activate simulate
```
### Cell simulation
To compile the C++ code:
```
cd Lineage_simulation
rm -r build
mkdir build
cd build
cmake ..
make
```
The sim folder contains an example `color.txt` file for `simulate`. Run an example simulation:
```
build/simulate -C 100 -c sim/color.txt -m 1 -mig 1e-2 -K 1000 -f 1 > sim/m1_mig1e-2_K1e3_f1.log
```
This will output a log file, `cellDivisionHistory.txt` (generation, parent cell, child cell), `cloneCells.txt` (clone site, {clone mutations}, clone cells), and `migrationCells.txt` (cells migrating to other sites). 

Alternatively, clone the original [MACHINA repo](https://github.com/raphael-group/machina) and modify code from `machina/src/simulation` based on `Lineage_simulation/src`. Compile and run the simulation by following the instructions from MACHINA.
### Lineage reconstruction
Sample cells from simulation and build a cell lineage tree using `cell_lineage.py`. It takes clonal cells, migration cells, and cell division history from simulation, samples cells from each clone with a proportion (default p=1.0, using all simulated cells), and coalesces them into a cell lineage tree, along with a regime file indicating migration sites along lineage. An example of cell lineage tree by sampling 30% of simulated cells:
```
python cell_lineage.py --cells sim/cloneCells.txt --division sim/cellDivisionHistory.txt --migration sim/migrationCells.txt -p 0.3
```
## Stochastic simulation
`Stochastic_simulation/data` contains input files for simulations, including a Newick tree file from lineage simulation and regime files indicating tissue labels on the tree. Regime files are previously defined by [EvoGeneX](https://liebertpub.com/doi/10.1089/cmb.2022.0121), containing cell pairs and labels of their most recent common ancestors on the tree. `prep_regime.py` generates regime files from migrating cells, including those in a lineage close to the root (as from lineage simulation), those in a lineage close to the tip, and those in both lineages (visualized in PNG files). Example regimes were produced from the following command:
```
python prep_regime.py -t tree.nwk -c cells_root.txt -r regime_root.csv
python prep_regime.py -t tree.nwk -c cells_tip.txt -r regime_tip.csv
python prep_regime.py -t tree.nwk -c cells_both.txt -r regime_both.csv
```

`Stochastic_simulation/src` contains a Python script for gene expression simulations. `stochas_sim.py` takes the tree and regime as input, and simulates gene expression read counts with Brownian motion (negative control) and Ornstein-Uhlenbeck process (positive expression changes) along the tree. Simulated read counts and illustrations of stochastic processes are generated in `Stochastic_simulation/sim`. Example simulations were produced from the following command:
```
python src/stochas_sim.py --tree data/tree.nwk --regime data/regime_root.csv --test 1 --root 5 --optim 10 --out sim --label root5-10
python src/stochas_sim.py --tree data/tree.nwk --regime data/regime_tip.csv --test 1 --root 5 --optim 10 --out sim --label tip5-10
python src/stochas_sim.py --tree data/tree.nwk --regime data/regime_both.csv --test 1 --root 5 --optim 10 --out sim --label both5-10
```

