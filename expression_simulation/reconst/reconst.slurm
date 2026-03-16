#!/bin/bash
#SBATCH --job-name=reconst
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=reconst_%j.out
#SBATCH --error=reconst_%j.err

source ~/.bashrc
conda activate singlecellstochastics
bash ./run_reconst.sh
