#!/bin/bash
#PBS -o ankh_tps.txt
#PBS -e ankh_tps.txt
#PBS -N ankh_tps

# Example run from the parent directory of the script
# qsub -A $PROJECT_ID -q qnvidia -l walltime=48:00:00 train.sh

# Change to working directory
cd "${PBS_O_WORKDIR}/ankh_tps" || exit 1

# Prepare project environment
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ankh_tps

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py
