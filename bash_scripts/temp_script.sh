#!/bin/bash


#SBATCH --job-name grace_test         # Process Name
#SBATCH --partition dgx   # Queue to run
#SBATCH --gres=gpu:1              # Number of GPUs to use


export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/tenayat/condaenv38

export TFHUB_CACHE_DIR=.


python neckmask_temporary.py

