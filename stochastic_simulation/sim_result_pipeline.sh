#!/bin/bash
#SBATCH --job-name=sim_result_pipeline
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=sim_result_pipeline_%j.out
#SBATCH --error=sim_result_pipeline_%j.err
#SBATCH --export=NONE

lineage=3tip
root=10
optim=13
alpha=1.0
sigma2=1.0

module load EBModules
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate singlecellstochastics

# simulation
run-stochas-sim \
    --tree data/tree.nwk \
    --regime data/regime_$lineage.csv \
    --test 1 \
    --root $root \
    --optim $optim \
    --alpha $alpha \
    --sigma2 $sigma2 \
    --out sim/ \
    --label $lineage$root-$optim

# OUP
run-ou-poisson \
    --tree data/tree.nwk \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_$lineage$root-$optim.tsv \
    --null 0 \
    --outdir results/ \
    --prefix h1_$lineage$root-$optim \
    --wandb h1_$lineage$root-$optim

run-ou-poisson \
    --tree data/tree.nwk \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_all$root-$root.tsv \
    --null 0 \
    --outdir results/ \
    --prefix h0_$lineage$root \
    --wandb h0_$lineage$root

conda activate EGX
cd evogenex

# EGX
python prep_EGX_long.py ../sim/readcounts_$lineage$root-$optim.tsv
python prep_EGX_long.py ../sim/readcounts_all$root-$root.tsv

bash adaptive_runner.sh \
    -t tree.nwk \
    -r regime_${lineage}_PRI.csv \
    -u regime_${lineage}_MET.csv \
    -d ./ \
    -o ../results/ \
    -p EGX_

cd ..
conda activate singlecellstochastics

# DEA
run-dea \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_$lineage$root-$optim.tsv \
    --outdir results/

run-dea \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_all$root-$root.tsv \
    --outdir results/

cd results
mv EGX_readcounts_$lineage$root-${optim}_long.csv EGX_h1_readcounts_$lineage$root-$optim.csv
mv EGX_readcounts_all$root-${root}_long.csv EGX_h0_readcounts_$lineage$root.csv
mv DEA_readcounts_$lineage$root-$optim.tsv DEA_h1_readcounts_$lineage$root-$optim.tsv
mv DEA_readcounts_all$root-$root.tsv DEA_h0_readcounts_$lineage$root.tsv

# Compare results
compare-results \
    --OUP_neg h0_$lineage${root}_chi-squared.tsv \
    --OUP_pos h1_$lineage$root-${optim}_chi-squared.tsv \
    --OU_neg h0_$lineage${root}_ou-init.tsv \
    --OU_pos h1_$lineage$root-${optim}_ou-init.tsv \
    --EGX_neg EGX_h0_readcounts_$lineage$root.csv \
    --EGX_pos EGX_h1_readcounts_$lineage$root-$optim.csv \
    --DEA_neg DEA_h0_readcounts_$lineage$root.tsv \
    --DEA_pos DEA_h1_readcounts_$lineage$root-$optim.tsv \
    --output $lineage$root-$optim
