cd /mnt/homeGPU/tenayat

# sbatch ganslate.sh mri_to_ct/experiments/cyclegan.yaml mae_clean_mask
sbatch ganslate.sh mri_to_ct/experiments/24_11_19_baseline_400epoch.yaml

tail -f slurm_archive/slurm-5*.out

python compute_metrics.py --gt data/TEST/CT --syn mri_to_ct/cyclegan_unpaired/infer/saved/ --output mri_to_ct/cyclegan_unpaired/output --mask-path data/TEST/MASKS/

python volumetric_to_png.py --folder mri_to_ct/cyclegan_unpaired/output/difference_images/ --color_mode 1 -n 1

echo $PWD

ganslate infer config="$CONFIG_YAML"

#run this on local to sync here with local machine
rsync -anv --exclude-from=mri_to_ct/exclude_list.txt tenayat@ngpu.ugr.es:/mnt/homeGPU/tenayat/ mri_to_ct/

