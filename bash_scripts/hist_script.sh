cd /mnt/homeGPU/tenayat

# sbatch ganslate.sh mri_to_ct/experiments/cyclegan.yaml mae_clean_mask
sbatch bash_scripts/ganslate.sh mri_to_ct/experiments/cyclegan_temp.yaml

tail -f slurm_archive/slurm-5*.out

python compute_metrics.py --gt data/TEST/CT --syn mri_to_ct/cyclegan_unpaired/infer/saved/ --output mri_to_ct/cyclegan_unpaired/output --mask-path data/TEST/MASKS/

python volumetric_to_png.py --folder mri_to_ct/cyclegan_unpaired/output/difference_images/ --color_mode 1 -n 1

echo $PWD

ganslate infer config="$CONFIG_YAML"

#run this on local to sync here with local machine
# rsync -anv --exclude-from=mri_to_ct/exclude_list.txt tenayat@ngpu.ugr.es:/mnt/homeGPU/tenayat/ mri_to_ct/
rsync -anv conda4ganslate/lib/python3.9/site-packages/ganslate/ ganslate/ganslate
rsync -avzP

# either this or that:
#SBATCH --partition dgx
#SBACTH --nodelist dgx1

#SBATCH --partition dios
python python_scripts/delete_redundant_models.py -e 24_11_19_baseline_400epoch
