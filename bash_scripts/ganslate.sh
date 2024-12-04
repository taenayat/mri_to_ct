#!/bin/bash

# Job name
#SBATCH --job-name cyclegan
#SBATCH --output /mnt/homeGPU/tenayat/slurm_archive/slurm-%j.out
#SBATCH --partition dgx2

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

CONFIG_YAML=$1
METRIC="mae_clean_mask"

echo "$PWD"
echo "Using ${1} experiment"

ganslate train config="$CONFIG_YAML"

python python_scripts/get_best_model.py --config $CONFIG_YAML --selection-metric "$METRIC"

ganslate test config="$CONFIG_YAML"

printf "\n\nFINISHED\n\n"
