#!/bin/bash

# Job name
#SBATCH --job-name pix2pix
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

METRIC="mae_clean_mask"

echo "$PWD"

# ganslate train config="mri_to_ct/24_12_19_patchgan/ndf/ndf.yaml"
# python python_scripts/get_best_model.py --config "mri_to_ct/24_12_19_patchgan/ndf/ndf.yaml" --selection-metric "$METRIC"
# ganslate test config="mri_to_ct/24_12_19_patchgan/ndf/ndf.yaml"
# python python_scripts/test_average.py -d mri_to_ct/24_12_19_patchgan/ndf/ndf.yaml

# echo "EXPERIMENT 1 FINISHED"

# ganslate train config="mri_to_ct/24_12_19_patchgan/n_layers_3/n_layers_3.yaml"
# python python_scripts/get_best_model.py --config "mri_to_ct/24_12_19_patchgan/n_layers_3/n_layers_3.yaml" --selection-metric "$METRIC"
# ganslate test config="mri_to_ct/24_12_19_patchgan/n_layers_3/n_layers_3.yaml"
# python python_scripts/test_average.py -d mri_to_ct/24_12_19_patchgan/n_layers_3/n_layers_3.yaml

# echo "EXPERIMENT 2 FINISHED"

# ganslate train config="mri_to_ct/24_12_19_patchgan/n_layers_4/n_layers_4.yaml"
# python python_scripts/get_best_model.py --config "mri_to_ct/24_12_19_patchgan/n_layers_4/n_layers_4.yaml" --selection-metric "$METRIC"
# ganslate test config="mri_to_ct/24_12_19_patchgan/n_layers_4/n_layers_4.yaml"
# python python_scripts/test_average.py -d mri_to_ct/24_12_19_patchgan/n_layers_4/n_layers_4.yaml

# echo "EXPERIMENT 3 FINISHED"

ganslate train config="mri_to_ct/24_12_19_patchgan/kernel/kernel.yaml"
python python_scripts/get_best_model.py --config "mri_to_ct/24_12_19_patchgan/kernel/kernel.yaml" --selection-metric "$METRIC"
ganslate test config="mri_to_ct/24_12_19_patchgan/kernel/kernel.yaml"
python python_scripts/test_average.py -d mri_to_ct/24_12_19_patchgan/kernel/kernel.yaml

echo "EXPERIMENT 4 FINISHED"


printf "\n\nFINISHED\n\n"
