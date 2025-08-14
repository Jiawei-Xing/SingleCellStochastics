# SingleCellStochastics
* Stochastic modeling of single-cell gene expression dynamics along cell lineage using Ornstein-Uhlenback process and Poisson observation.
* Detecting lineage-specific differential gene expressions by hypothesis testing on stochastic expression models.
* Reconstructing possible ancestral cell states with deep generative models along cell lineage.
## Lineage simulation
Example cell lineages can be simulated with a modified version of the agent-based cancer cell simulator from [MACHINA](https://www.nature.com/articles/s41588-018-0106-z). 
### Required environment
Create the required conda environment with `Lineage_simulation/env/lineage_sim.yml`:
```
conda env create -n simulate -f Lineage_simulation/env/lineage_sim.yml
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
Run an example simulation:
```
./simulate -C 100 -c ../sim/color.txt -m 1 -mig 1e-2 -K 1000 -f 1 > ../sim/m1_mig1e-2_K1e3_f1.log
```
This will output a log file, which can be further split into `cellDivisionHistory.txt` (generation, parent cell, child cell), `cloneCells.txt` (clone site, {clone mutations}, clone cells), and `migrationCells.txt` (cells migrating to other sites). 

Alternatively, clone the original MACHINA repo (https://github.com/raphael-group/machina) and modify codes from `src/simulation` using `Lineage_simulation/src`. Compile and run simulation by following instructions from MACHINA.
### Lineage reconstruction
Sample cells from simulation and build a cell lineage tree using `cell_lineage.py`. It takes clonal cells and cell division history from simulation, samples cells from each clone with a proportion (default p=1.0, using all simulated cells), and coalesces them into a cell lineage tree. An example of cell lineage tree sampling 30% of simulated cells:
```
cd ..
python cell_lineage.py --cell_file sim/cloneCells.txt --history_file sim/cellDivisionHistory.txt --output_file m1_mig1e-2_K1e3_f1_p0.3.nwk -p 0.3
```
