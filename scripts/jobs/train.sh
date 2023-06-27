#!/bin/bash
#PBS -o ankh_tps.txt
#PBS -e ankh_tps.txt
#PBS -N ank_tps

# Example run from the parent directory of the script
# qsub -A $PROJECTID_R -q qnvidia -l walltime=48:00:00 train.sh

# Change to working directory
cd "${PBS_O_WORKDIR}" || exit 1

# Prepare project environment
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ankh_tps

python ../train.py
