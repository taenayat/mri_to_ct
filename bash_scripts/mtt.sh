#!/bin/bash

# Job name
#SBATCH --job-name MTT
#SBATCH --output /mnt/homeGPU/tenayat/slurm_archive/slurm-%j.out
#SBATCH --partition dgx

#SBACTH --mem 64G

# Use GPU
#SBATCH --gres=gpu:1

# Default configs for NGPU
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
export TFHUB_CACHE_DIR=.

set -euo pipefail

# Activating conda enviroment
conda activate /mnt/homeGPU/tenayat/cuda11

cd /mnt/homeGPU/tenayat

# python MTT-Net/train.py
python MTT-Net/test.py

printf "\n\nFINISHED\n\n"
