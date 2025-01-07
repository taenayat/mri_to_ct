#!/bin/bash

# Job name
#SBATCH --job-name pix2pix
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

# ganslate train config="mri_to_ct/24_12_18_vnet/first_layer/first_layer.yaml"
# python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/first_layer/first_layer.yaml" --selection-metric "$METRIC"
# ganslate test config="mri_to_ct/24_12_18_vnet/first_layer/first_layer.yaml"
# python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/first_layer/first_layer.yaml

# echo "EXPERIMENT 1 FINISHED"

# ganslate train config="mri_to_ct/24_12_18_vnet/2_2_2_2/2_2_2_2.yaml"
# python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/2_2_2_2/2_2_2_2.yaml" --selection-metric "$METRIC"
# ganslate test config="mri_to_ct/24_12_18_vnet/2_2_2_2/2_2_2_2.yaml"
# python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/2_2_2_2/2_2_2_2.yaml

# echo "EXPERIMENT 2 FINISHED"

ganslate train config="mri_to_ct/24_12_18_vnet/1_2_3_4/1_2_3_4.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/1_2_3_4/1_2_3_4.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_18_vnet/1_2_3_4/1_2_3_4.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/1_2_3_4/1_2_3_4.yaml

echo "EXPERIMENT 3 FINISHED"

ganslate train config="mri_to_ct/24_12_18_vnet/4_3_2_1/4_3_2_1.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/4_3_2_1/4_3_2_1.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_18_vnet/4_3_2_1/4_3_2_1.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/4_3_2_1/4_3_2_1.yaml

echo "EXPERIMENT 4 FINISHED"

ganslate train config="mri_to_ct/24_12_18_vnet/3_3_3_3/3_3_3_3.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/3_3_3_3/3_3_3_3.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_18_vnet/3_3_3_3/3_3_3_3.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/3_3_3_3/3_3_3_3.yaml

echo "EXPERIMENT 5 FINISHED"

ganslate train config="mri_to_ct/24_12_18_vnet/1_1_1_1/1_1_1_1.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/1_1_1_1/1_1_1_1.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_18_vnet/1_1_1_1/1_1_1_1.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/1_1_1_1/1_1_1_1.yaml

echo "EXPERIMENT 6 FINISHED"

ganslate train config="mri_to_ct/24_12_18_vnet/2_2_2_2_2/2_2_2_2_2.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/2_2_2_2_2/2_2_2_2_2.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_18_vnet/2_2_2_2_2/2_2_2_2_2.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/2_2_2_2_2/2_2_2_2_2.yaml

echo "EXPERIMENT 7 FINISHED"

ganslate train config="mri_to_ct/24_12_18_vnet/2_2_2/2_2_2.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_18_vnet/2_2_2/2_2_2.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_18_vnet/2_2_2/2_2_2.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_18_vnet/2_2_2/2_2_2.yaml

echo "EXPERIMENT 8 FINISHED"

printf "\n\nFINISHED\n\n"
