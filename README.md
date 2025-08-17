# SingleCellStochastics
* Stochastic modeling of single-cell gene expression dynamics along cell lineage using Ornstein-Uhlenback process and Poisson observation.
* Detecting lineage-specific differential gene expressions by hypothesis testing on stochastic expression models.
* Reconstructing possible ancestral cell states with deep generative models along cell lineage.
## Lineage simulation
Example cell lineages can be simulated with a modified version of the agent-based cancer cell simulator from [MACHINA](https://www.nature.com/articles/s41588-018-0106-z). 
### Required environment
Create and activate the required conda environment with `Lineage_simulation/env/lineage_sim.yml`:
```
conda env create -n simulate -f Lineage_simulation/env/lineage_sim.yml
conda activate simulate
```
### Cell simulation
An executable file is provided in `Lineage_simulation/build/`. Or, to recompile the c++ code:
```
cd Lineage_simulation
rm -r build
mkdir build
cd build
cmake ..
make
```
Go to the sim folder containing an example `color.txt` file. Run an example simulation:
```
../build/simulate -C 100 -c color.txt -m 1 -mig 1e-2 -K 1000 -f 1 > m1_mig1e-2_K1e3_f1.log
```
This will output a log file, `cellDivisionHistory.txt` (generation, parent cell, child cell), `cloneCells.txt` (clone site, {clone mutations}, clone cells), and `migrationCells.txt` (cells migrating to other sites). 

Alternatively, clone the original MACHINA repo (https://github.com/raphael-group/machina) and modify codes from `src/simulation` based on `Lineage_simulation/src`. Compile and run the simulation by following instructions from MACHINA.
### Lineage reconstruction
Sample cells from simulation and build a cell lineage tree using `cell_lineage.py`. It takes clonal cells, migration cells, and cell division history from simulation, samples cells from each clone with a proportion (default p=1.0, using all simulated cells), and coalesces them into a cell lineage tree, together with a regime file indicating migration sites along lineage. An example of cell lineage tree sampling 30% of simulated cells:
```
python cell_lineage.py --cell_file sim/cloneCells.txt --division_file sim/cellDivisionHistory.txt --migration_file migrationCells.txt -p 0.3
```
## Stochastic simulation
`Stochastic_simulation/prep_regime.ipynb` takes an example tree and cells for testing, and generates two regime files. Regime files are previously defined by [EvoGeneX](https://liebertpub.com/doi/10.1089/cmb.2022.0121), containing cell pairs and the label of their most recent common ancestor on the tree. The PRIregime file contains the same label for all nodes, representing the null hypothesis where all cells are the same. The METregime file contains different labels (e.g., different migration sites), corresponding to the alternative hypothesis where gene expressions shift in certain lineages. 

`Stochastic_simulation/sim.ipynb` takes an example tree, lineages of cells for testing, and regime files, and simulates gene expression read counts with Brownian motion (negative control) and Ornstein-Uhlenbeck process (positive expression changes) along the tree. Simulated read counts and illustrations of stochastic processes are generated from the code. Parameters for both stochastic processes can be modified.
