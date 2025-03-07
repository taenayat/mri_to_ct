import os
import SimpleITK as sitk
from segmentation_functions import segmentation_stats
import numpy as np
import vtk
import tqdm
import pandas as pd

# SCT_PATH = '/home/taha/Downloads/Panacea/mri_to_ct/mri_to_ct/24_12_24_final/infer/saved'
# CT_PATH = '/home/taha/Downloads/Panacea/dataset/TEST/CT'

# SCT_PATH_BASE = '/mnt/homeGPU/tenayat/mri_to_ct/25_02_28_thresh150_correctfunc/infer'
# SCT_PATH_BASE = '/mnt/homeGPU/tenayat/MTT-Net/train_model/MTTNET_wave3DDiscriminator_results/epoch_120_Npatch_24_maxCT_3000/test'
SCT_PATH_BASE = '/mnt/homeGPU/tenayat/mri_to_ct/24_12_04_squeeze/infer'
CT_PATH = '/mnt/homeGPU/tenayat/data/TEST/CT'
sct_dir = sorted([file for file in os.listdir(os.path.join(SCT_PATH_BASE,'saved')) if file[-6:]=='nii.gz'], key = lambda x: x)
ct_dir = sorted(os.listdir(CT_PATH), key = lambda x: x)
ct_full_dir = [os.path.join(CT_PATH, ct_dir[idx]) for idx in range(len(ct_dir))]
sct_full_dir = [os.path.join(SCT_PATH_BASE, 'saved', sct_dir[idx]) for idx in range(len(sct_dir))]

def load_and_threshold(image_path, threshold=400):
    image = sitk.ReadImage(image_path, sitk.sitkInt16)
    binary_image = sitk.BinaryThreshold(image,
                                        lowerThreshold=threshold,
                                        upperThreshold=3000,
                                        insideValue=1,
                                        outsideValue=0)
    return binary_image

def get_file_name(dir):
    return dir.split('/')[-1].split('.')[0]

for ref_dir, seg_dir in zip(ct_full_dir, sct_full_dir):
    ref = load_and_threshold(ref_dir)
    seg = load_and_threshold(seg_dir)
    file_id = get_file_name(ref_dir)
    print(file_id)
    one_row = segmentation_stats(ref, seg, file_id)
    if 'df' not in locals():
        df = one_row
    else:
        df = pd.concat([df, one_row], sort=False)

average_row = df.mean().to_frame().T
std_row = df.std().to_frame().T

# Add labels for the new rows
average_row.index = ["avg"]
std_row.index = ["stddev"]

# Append the new rows to the DataFrame
df = pd.concat([df, average_row, std_row])
os.makedirs(os.path.join(SCT_PATH_BASE,'comparison'), exist_ok=True)
df.to_csv(os.path.join(SCT_PATH_BASE,'comparison', 'skull_metrics.csv'))
