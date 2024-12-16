#!/bin/bash

# Job name
#SBATCH --job-name cyclegan
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

METRIC="mae_clean_mask"

echo "$PWD"

ganslate train config="mri_to_ct/24_12_13_patch_n_batch/b2_p32/b2_p32.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_13_patch_n_batch/b2_p32/b2_p32.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_13_patch_n_batch/b2_p32/b2_p32.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_13_patch_n_batch/b2_p32/b2_p32.yaml

echo "EXPERIMENT 1 FINISHED"

ganslate train config="mri_to_ct/24_12_13_patch_n_batch/b1_p64/b1_p64.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_13_patch_n_batch/b1_p64/b1_p64.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_13_patch_n_batch/b1_p64/b1_p64.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_13_patch_n_batch/b1_p64/b1_p64.yaml

echo "EXPERIMENT 2 FINISHED"

ganslate train config="mri_to_ct/24_12_13_patch_n_batch/b2_p64/b2_p64.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_13_patch_n_batch/b2_p64/b2_p64.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_13_patch_n_batch/b2_p64/b2_p64.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_13_patch_n_batch/b2_p64/b2_p64.yaml

echo "EXPERIMENT 3 FINISHED"

ganslate train config="mri_to_ct/24_12_13_patch_n_batch/sw_0_5/sw_0_5.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_13_patch_n_batch/sw_0_5/sw_0_5.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_13_patch_n_batch/sw_0_5/sw_0_5.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_13_patch_n_batch/sw_0_5/sw_0_5.yaml

echo "EXPERIMENT 4 FINISHED"

ganslate train config="mri_to_ct/24_12_13_patch_n_batch/aug/aug.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_13_patch_n_batch/aug/aug.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_13_patch_n_batch/aug/aug.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_13_patch_n_batch/aug/aug.yaml

echo "EXPERIMENT 5 FINISHED"

ganslate train config="mri_to_ct/24_12_13_patch_n_batch/vanilla/vanilla.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_13_patch_n_batch/vanilla/vanilla.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_13_patch_n_batch/vanilla/vanilla.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_13_patch_n_batch/vanilla/vanilla.yaml

echo "EXPERIMENT 6 FINISHED"
printf "\n\nFINISHED\n\n"
