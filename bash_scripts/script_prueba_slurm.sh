#!/bin/bash


#SBATCH --job-name Prueba         # Process Name

#SBATCH --partition dios   # Queue to run

#SBATCH --gres=gpu:1              # Number of GPUs to use

        

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/Environments/tf2.2py36

export TFHUB_CACHE_DIR=.


python pruebatf2.py          


mail -s "Process Completed" taha.enayat.panacea@gmail.com <<< "The Process has Ended"
