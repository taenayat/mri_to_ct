#!/bin/bash


#SBATCH --job-name grace         # Process Name

#SBATCH --partition dgx   # Queue to run

#SBATCH --gres=gpu:1              # Number of GPUs to use

        

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/tenayat/condaenv38

export TFHUB_CACHE_DIR=.


python neckmask.py

# echo "The Process has Ended" | mail -s "Process Completed" taha.enayat.panacea@gmail.com
# mail -s "Process Completed" taha.enayat.panacea@gmail.com <<< "The Process has Ended"
