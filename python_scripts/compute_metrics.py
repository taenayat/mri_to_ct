import numpy as np
import argparse
import os
import SimpleITK as sitk
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tqdm
import multiprocessing
import math

def remove_bck(mask, data):
    """Remove background from the data using the mask provided."""
    
    mask_array = sitk.GetArrayFromImage(mask)
    filtered_array = data[mask_array != 0]

    return filtered_array

def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)"""
    mae_value = np.mean(np.abs(gt - pred))
    return float(mae_value)


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)"""
    mse_value = np.mean((gt - pred)**2)
    return float(mse_value)


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    nmse_value = np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2
    return float(nmse_value)


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=gt.max())
    return float(psnr_value)

def padding(image, padding_x, padding_y, padding_z):

    minmax_filter = sitk.MinimumMaximumImageFilter()
    minmax_filter.Execute(image)
    minimun = minmax_filter.GetMinimum()

    size = image.GetSize()
    padding_list = [padding_x, padding_y, padding_z]
    diff = [ (pad - x)/2 for x, pad in zip(size, padding_list)]
    diff_upper = [ math.ceil(d) for d in diff ]
    diff_lower = [ math.floor(d) for d in diff ]

    padder = sitk.ConstantPadImageFilter()
    padder.SetConstant(minimun)
    padder.SetPadUpperBound(diff_upper)
    padder.SetPadLowerBound(diff_lower)

    image_padded = padder.Execute(image)
    image_padded.SetOrigin(image.GetOrigin())

    return image_padded

def ssim(gt: np.ndarray, pred: np.ndarray, maxval: float = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    size = (gt.shape[0] * gt.shape[1]) if gt.ndim == 4 else gt.shape[0]

    ssim_sum = 0
    for channel in range(gt.shape[0]):
        # Format is CxHxW or DxHxW
        if gt.ndim == 3:
            target = gt[channel]
            prediction = pred[channel]
            ssim_sum += structural_similarity(target, prediction, data_range=maxval)

        # Format is CxDxHxW
        elif gt.ndim == 4:
            for slice_num in range(gt.shape[1]):
                target = gt[channel, slice_num]
                prediction = pred[channel, slice_num]
                ssim_sum += structural_similarity(target, prediction, data_range=maxval)
        else:
            raise NotImplementedError(f"SSIM for {gt.ndim} images not implemented")

    return ssim_sum / size

def process_case(gt_path, syn_path, mask_path, output_path):
    file_code = os.path.basename(syn_path.split(".")[0])
        
    gt_image = sitk.ReadImage(gt_path)
    mask = sitk.ReadImage(mask_path)

    if gt_image.GetSize() != mask.GetSize():
        mask = padding(mask, gt_image.GetSize()[0], gt_image.GetSize()[1], gt_image.GetSize()[2])

    gt_array = sitk.GetArrayFromImage(gt_image)
    mask_array = sitk.GetArrayFromImage(mask)

    gt_array = gt_array * mask_array

    syn_image = sitk.ReadImage(syn_path)
    syn_array = sitk.GetArrayFromImage(syn_image)
    syn_array = syn_array * mask_array

    syn_array_filter = remove_bck(mask, syn_array)
    gt_array_filter = remove_bck(mask, gt_array)
    
    mae_value = mae(gt_array_filter, syn_array_filter)
    mse_value = mse(gt_array_filter, syn_array_filter)
    nmse_value = nmse(gt_array_filter, syn_array_filter)
    psnr_value = psnr(gt_array_filter, syn_array_filter)
    ssim_value = ssim(gt_array, syn_array)

    case_id = os.path.basename(gt_path).split(".")[0]

    metrics={0:{"ID": case_id,
        "MAE": mae_value,
        "MSE": mse_value,
        "NMSE": nmse_value,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
    }}
    metrics_df = pd.DataFrame().from_dict(metrics).T

    diff_array = gt_array - syn_array
    diff_image = sitk.GetImageFromArray(diff_array)
    diff_image.CopyInformation(gt_image)
    os.makedirs(os.path.join(output_path,"difference_images"), exist_ok=True)
    sitk.WriteImage(diff_image, os.path.join(output_path,"difference_images",file_code+"_diff.nii.gz"))

    error_array = abs(gt_array - syn_array)
    error_array_image = sitk.GetImageFromArray(error_array)
    error_array_image.CopyInformation(gt_image)
    os.makedirs(os.path.join(output_path,"error_images"), exist_ok=True)
    sitk.WriteImage(error_array_image, os.path.join(output_path,"error_images",file_code+"_error.nii.gz"))

    return metrics_df

def main(gt_files, syn_files, mask_files, output_path):
    # Load the ground truth and synthetic data

    param_list = []
    for gt_path, syn_path,  mask_path in tqdm.tqdm(zip(gt_files, syn_files,mask_files), total=len(gt_files)):
        param_list.append([gt_path, syn_path, mask_path, output_path])

    pool = multiprocessing.Pool(processes=7)
    metrics_list = pool.starmap(process_case, param_list)
    pool.close()

    metrics = pd.concat(metrics_list)
    metrics.reset_index(inplace=True)
        
    metrics.to_csv(os.path.join(output_path,"test_metrics.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute metrics for a given dataset')
    parser.add_argument('--gt', type=str, help='Path to the ground truth data', required=True)
    parser.add_argument('--syn', type=str, help='Path to the synthetic data', required=True)
    parser.add_argument('--mask-path',type=str,help='Path to the folder with the cbct masks',required=True)
    parser.add_argument('--output', type=str, help='Path to the output folder', required=True)


    args = parser.parse_args()

    # Get all the files inside the ground truth and synthetic data folders
    gt_files = [os.path.join(args.gt,f) for f in os.listdir(args.gt) if os.path.isfile(os.path.join(args.gt, f)) and f.endswith('.nii.gz')]
    syn_files = [os.path.join(args.syn,f) for f in os.listdir(args.syn) if os.path.isfile(os.path.join(args.syn, f)) and f.endswith('.nii.gz')]
    mask_files= [os.path.join(args.mask_path,f) for f in os.listdir(args.mask_path) if os.path.isfile(os.path.join(args.mask_path, f)) and f.endswith('.nii.gz')]

    gt_files = sorted(gt_files)
    syn_files = sorted(syn_files)
    mask_files = sorted(mask_files)
    
    main(gt_files, syn_files, mask_files, args.output)