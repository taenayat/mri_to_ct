"""
This script will look into your gaslate experiments' log and removes the redundant
models saved in each checkpoint. If there is a best model selected from python_scripts/get_best_model.py,
then this script finds is from the updated yaml file and keeps the best model as well as the final checkpoint.
if the best model coinsides the last checkpoint, only one model is kept. 
"""
 
import os
import yaml
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-e",
    "--experiment",
    type=str,
    help="Name of the experiment you want to remove redundant saved models.",
    required=True
)
args = parser.parse_args()
experiment = args.experiment

PATH = '/mnt/homeGPU/tenayat/mri_to_ct'
# experiment = '24_11_17_cyclegan_firstrun'
# experiment = 'cyclegan_temp'
with open(os.path.join(PATH, 'experiments', experiment+'.yaml'), 'r') as file:
    config = yaml.safe_load(file)

if config['test']['checkpointing']['load_iter'] != 0:
    mid_checkpoint_to_keep = config['test']['checkpointing']['load_iter']
else:
    mid_checkpoint_to_keep = None

checkpoints = os.listdir(os.path.join(PATH, experiment, 'checkpoints'))
checkpoints_reverse_sorted = sorted([item.split('.')[0] for item in checkpoints], key=lambda x: -int(x))

last_checkpoint_to_keep = checkpoints_reverse_sorted[0]

files_to_keep = list(set([f'{f}.pth' for f in [last_checkpoint_to_keep,mid_checkpoint_to_keep] if f is not None]))
files_to_remove = [f for f in checkpoints if f not in files_to_keep]

print("Files to keep:")
for f in files_to_keep:
    print(f"{experiment}  - {f}")

confirm = input("\nDo you want to delete the above files? (yes/no): ").strip().lower()

if confirm in ["yes", "y"]:
    for file_path in files_to_remove:
        file_full_path = os.path.join(PATH, experiment, 'checkpoints', file_path)
        try:
            os.remove(file_full_path)
            print(f"Deleted: {file_full_path}")
        except Exception as e:
            print(f"Failed to delete {file_full_path}: {e}")
    print("\nDeletion complete.")
else:
    print("\nNo files were deleted.")
