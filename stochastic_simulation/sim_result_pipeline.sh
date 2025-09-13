#!/bin/bash
#SBATCH --job-name=sim_result_pipeline
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=sim_result_pipeline_%j.out
#SBATCH --error=sim_result_pipeline_%j.err

lineage=root
root=5
optim=10

conda activate singlecellstochastics

# simulation
run-stochas-sim \
    --tree data/tree.nwk \
    --regime data/regime_$lineage.csv \
    --test 1 \
    --root $root \
    --optim $optim \
    --out sim/ \
    --label $lineage$root-$optim

# OUP
run-ou-poisson \
    --tree data/tree.nwk \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_BM_$lineage$root-$optim.tsv \
    --null 0 \
    --outdir results/ \
    --prefix BM_$lineage$root-$optim \
    --wandb BM_$lineage$root-$optim

run-ou-poisson \
    --tree data/tree.nwk \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_OU_$lineage$root-$optim.tsv \
    --null 0 \
    --outdir results/ \
    --prefix OU_$lineage$root-$optim \
    --wandb OU_$lineage$root-$optim

conda activate EGX
cd evogenex

# EGX
python prep_EGX_long.py ../sim/readcounts_BM_$lineage$root-$optim.tsv
python prep_EGX_long.py ../sim/readcounts_OU_$lineage$root-$optim.tsv

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
    --expr sim/readcounts_BM_$lineage$root-$optim.tsv \
    --outdir results/

run-dea \
    --regime data/regime_$lineage.csv \
    --expr sim/readcounts_OU_$lineage$root-$optim.tsv \
    --outdir results/

cd results

# Compare results
compare-results \
    --OUP_neg BM_$lineage$root-${optim}_chi-squared.tsv \
    --OUP_pos OU_$lineage$root-${optim}_chi-squared.tsv \
    --OU_neg BM_$lineage$root-${optim}_ou-init.tsv \
    --OU_pos OU_$lineage$root-${optim}_ou-init.tsv \
    --EGX_neg EGX_readcounts_BM_$lineage$root-${optim}_long.csv \
    --EGX_pos EGX_readcounts_OU_$lineage$root-${optim}_long.csv \
    --DEA_neg DEA_readcounts_BM_$lineage$root-$optim.tsv \
    --DEA_pos DEA_readcounts_OU_$lineage$root-$optim.tsv \
    --output $lineage$root-$optim
