#!/bin/bash


#SBATCH --job-name test         # Process Name
#SBATCH --partition dios
#SBACTH --nodelist hera
#SBATCH --gres=gpu:1              # Number of GPUs to use
#SBATCH --output /mnt/homeGPU/tenayat/slurm_archive/slurm-%j.out


export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

# conda activate /mnt/homeGPU/tenayat/condaenv38
# conda activate /mnt/homeGPU/tenayat/conda4ganslate
# conda activate /mnt/homeGPU/ggomez/envs/torch2cu11
conda activate /mnt/homeGPU/tenayat/cuda11

export TFHUB_CACHE_DIR=.

# which python
# python neckmask_temporary.py
# python python_scripts/cuda_version_get.py

ganslate train config="mri_to_ct/experiments/pix2pix_temporary.yaml"

