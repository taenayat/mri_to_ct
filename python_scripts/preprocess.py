import SimpleITK as sitk
import numpy as np
import os
print('libraries loaded')

DATA_PARENT_DIR = 'brain/'
dataset_dir = [os.path.join(DATA_PARENT_DIR, mri_path, 'mr.nii.gz') for mri_path in os.listdir(DATA_PARENT_DIR)]
mask_dir = [os.path.join(DATA_PARENT_DIR, mri_path, 'mask.nii.gz') for mri_path in os.listdir(DATA_PARENT_DIR)]
category = [dataset_dir[i].split('/')[-2][2] for i in range(len(dataset_dir))]
category_dir = {'A':[], 'B':[], 'C':[]}
for dir,cat in zip(zip(dataset_dir,mask_dir), category):
    if cat == 'A':
        category_dir['A'].append(dir)
    if cat == 'B':
        category_dir['B'].append(dir)
    if cat == 'C':
        category_dir['C'].append(dir)
SPACING = (1.0, 1.0, 1.0)
print('directories found')

threshold_dict = {'A':100, 'B':40, 'C':40}
NORMALIZATION_FACTOR = 350

def bbox(image_array):
    non_zero_slices_z = np.any(image_array != 0, axis=(1, 2))
    image_array = image_array[non_zero_slices_z, :, :]
    
    non_zero_slices_y = np.any(image_array != 0, axis=(0, 2))
    image_array = image_array[:, non_zero_slices_y, :]
    
    non_zero_slices_x = np.any(image_array != 0, axis=(0, 1))
    image_array = image_array[:, :, non_zero_slices_x]
    return image_array

for cat in ['A','B','C']:
    for mri_img_path, mask_path in category_dir[cat]:

        # read MRI and its mask
        mri_img_sitk = sitk.ReadImage(mri_img_path)
        mri_img = sitk.GetArrayFromImage(mri_img_sitk)
        mask_original = sitk.ReadImage(mask_path)
        mask_original = sitk.GetArrayFromImage(mask_original).astype(bool)

        # create the threshold mask
        threshold = threshold_dict[cat]
        mask_threshold = np.zeros_like(mri_img, dtype=bool)
        mask_threshold[mri_img > threshold] = 1

        # apply the mask and create a 1-D array of intersting voxels
        mri_img_masked = mri_img[mask_original & mask_threshold]

        # calculate mean and standard deviation
        img_mean = np.mean(mri_img_masked)
        img_var = np.std(mri_img_masked)

        # zero out the voxel outside the original mask (filter out non-interesting values) and reduce the dimension using bounding box, and normalize the image
        normalized_img = np.where(~mask_original, 0, mri_img)
        normalized_img = bbox(normalized_img)
        normalized_img = (normalized_img - img_mean) / img_var
        normalized_img += np.abs(np.min(normalized_img))
        normalized_img *= NORMALIZATION_FACTOR

        # convert back to SITK
        normalized_img_sitk = sitk.GetImageFromArray(normalized_img)
        normalized_img_sitk.SetOrigin(mri_img_sitk.GetOrigin())
        normalized_img_sitk.SetSpacing(SPACING)
        normalized_img_sitk.SetDirection(mri_img_sitk.GetDirection())

        print(mri_img_path, 'orig:', mri_img.shape, 'norm:', normalized_img.shape)

        # save to nifti file
        output_path = mri_img_path[:-7] + '_normalized' + mri_img_path[-7:]
        sitk.WriteImage(normalized_img_sitk, output_path)