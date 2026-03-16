#!/bin/bash
#SBATCH --job-name=scout
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/examples/scout/scout_%j.out
#SBATCH --error=/grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/examples/scout/scout_%j.err

source ~/.bashrc
conda activate scout

cd /grid/siepel/home/xing/gene_expression_evolution/SingleCellStochastics/examples

Rscript scout/run_scout.R 8
