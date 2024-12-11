import random
from pathlib import Path
import numpy as np
import os
import SimpleITK as sitk
from ganslate.utils.io import make_dataset_of_files
from ganslate.utils import sitk_utils

from ganslate.data.utils.normalization import min_max_normalize, min_max_denormalize, z_score_normalize, z_score_clip, z_score_squeeze

from dataclasses import dataclass

from torch.utils.data import Dataset

from ganslate import configs
from ganslate.data.utils.ops import pad



@dataclass
class SynthRAD2023ValTestDatasetConfig(configs.base.BaseDatasetConfig):
    pass

EXTENSIONS = ['.nii.gz']

class SynthRAD2023ValTestDataset(Dataset):

    def __init__(self, conf):
        root_path = Path(conf[conf.mode].dataset.root).resolve()
        mri_path=root_path / "MRI"
        ct_path=root_path / "CT"
        mask_path=root_path / "MASKS"
        self.mri_path = make_dataset_of_files(mri_path, EXTENSIONS)
        self.ct_path = make_dataset_of_files(ct_path, EXTENSIONS)
        self.mask_path = make_dataset_of_files(mask_path, EXTENSIONS)
        self.num_datapoints = len(self.mri_path)

    def __getitem__(self, index):
        final_index = index % self.num_datapoints
        
        ct_sample=self.ct_path[final_index]
        mri_sample=self.mri_path[final_index]
        mask_sample=self.mask_path[final_index]


        CT_image = sitk_utils.load(ct_sample)
        MRI_image = sitk_utils.load(mri_sample)
        mask = sitk_utils.load(mask_sample)

        CT_tensor = sitk_utils.get_tensor(CT_image)
        MRI_tensor = sitk_utils.get_tensor(MRI_image)
        mask_tensor = sitk_utils.get_tensor(mask)

        self.CT_min_value, self.CT_max_value = -1024, 3000
        CT_tensor = min_max_normalize(CT_tensor, self.CT_min_value, self.CT_max_value)
        MRI_tensor = z_score_squeeze(MRI_tensor)

        # print('before pad', MRI_tensor.shape, CT_tensor.shape)
        CT_tensor = pad(CT_tensor, (262,284,280))
        MRI_tensor = pad(MRI_tensor, (262,284,280))
        mask_tensor = pad(mask_tensor, (262,284,280))
        # print('after pad', MRI_tensor.shape, CT_tensor.shape)

        CT_tensor = CT_tensor.unsqueeze(0)
        MRI_tensor = MRI_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        mask_dict={"clean_mask": mask_tensor}
        return {'A': MRI_tensor, 'B': CT_tensor, "masks": mask_dict,"metadata": {"mri_path": str(mri_sample)}}


    def denormalize(self, tensor):
        return min_max_denormalize(tensor, self.CT_min_value, self.CT_max_value)


    def save(self, tensor, save_dir, metadata=None):
        # """ By default, ganslate logs images in png format. However, if you wish
        # to save images in a different way, then implement this `save()` method. 
        # For example, you could save medical images in their native format for easier
        # inspection or usage.
        # If you do not need this method, remove it.
        # """
        # tensor = min_max_denormalize(tensor, -1024, 3000)

        # np_image = tensor.numpy(force=True)
        # filename = os.path.basename(metadata["mri_path"])

        # image = sitk.GetImageFromArray(np_image[0])
        # os.makedirs(save_dir, exist_ok=True)
        # sitk_utils.write(image, save_dir / filename)
        pass

    def __len__(self):
        return self.num_datapoints