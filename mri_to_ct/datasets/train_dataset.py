import random
from pathlib import Path
import numpy as np
from ganslate.utils.io import make_dataset_of_files
from ganslate.utils import sitk_utils

from ganslate.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler
from ganslate.data.utils.normalization import min_max_normalize, z_score_normalize, z_score_clip, z_score_squeeze

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset

from ganslate import configs

from torchvision.transforms import v2 
from monai import transforms
from ganslate.data.utils.ops import pad

@dataclass
class SynthRAD2023TrainDatasetConfig(configs.base.BaseDatasetConfig):
    # Define other attributes
    patch_size: Tuple[int, int, int] = (16, 128, 128)
    # Proportion of focal region size compared to original volume size
    focal_region_proportion: float = 0
    # Whether to apply data augmentation
    augmentation: bool = False
    # Indicates if the data is unpaired or paired
    unpaired: bool = False

EXTENSIONS = ['.nii.gz']

class SynthRAD2023TrainDataset(Dataset):

    def __init__(self, conf):

        root_path = Path(conf.train.dataset.root).resolve()
        mri_path=root_path / "MRI"
        ct_path=root_path / "CT"
        self.mri_path = make_dataset_of_files(mri_path, EXTENSIONS)
        self.ct_path = make_dataset_of_files(ct_path, EXTENSIONS)
        self.num_datapoints = len(self.mri_path)

        focal_region_proportion = conf.train.dataset.focal_region_proportion
        self.patch_size = np.array(conf.train.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)
        
        if conf.train.dataset.augmentation:
            self.transformations = transforms.Compose([
                transforms.RandFlip(spatial_axis=0, prob=0.5),
                transforms.RandFlip(spatial_axis=1, prob=0.5),
                transforms.RandFlip(spatial_axis=2, prob=0.5),
                transforms.RandRotate(range_x=0.17, range_y=0.17, range_z=0.17, prob=0.3, align_corners=True),
            ])
        else:
            self.transformations = None

        if conf.train.dataset.unpaired:
            self.unpaired = True
        else:
            self.unpaired = False

    def __getitem__(self, index):
        index_A = index % self.num_datapoints

        if self.unpaired:
            index_B= random.randint(0, self.num_datapoints - 1)
        else:
            index_B = index_A
            
        mri_sample=self.mri_path[index_B]
        ct_sample=self.ct_path[index_B]
        
        CT_image = sitk_utils.load(ct_sample)
        MRI_image = sitk_utils.load(mri_sample)

        if (sitk_utils.is_image_smaller_than(CT_image, self.patch_size) or
                sitk_utils.is_image_smaller_than(MRI_image, self.patch_size)):
            raise ValueError("Volume size not smaller than the defined patch size.\
                                \nA: {} \nB: {} \npatch_size: {}."\
                                .format(sitk_utils.get_torch_like_size(CT_image),
                                        sitk_utils.get_torch_like_size(MRI_image),
                                        self.patch_size))

        CT_tensor = sitk_utils.get_tensor(CT_image)
        MRI_tensor = sitk_utils.get_tensor(MRI_image)
        # MRI_min_value, MRI_max_value = MRI_tensor.min(), MRI_tensor.max()
        MRI_tensor = z_score_squeeze(MRI_tensor)
        CT_tensor = min_max_normalize(CT_tensor, -1024, 3000)

        if self.transformations:
            CT_tensor = CT_tensor.unsqueeze(0)
            MRI_tensor = MRI_tensor.unsqueeze(0)

            MRI_tensor = self.transformations(MRI_tensor)
            CT_tensor = self.transformations(CT_tensor)

            CT_tensor = CT_tensor.squeeze(0)
            MRI_tensor = MRI_tensor.squeeze(0)

        CT_patch, MRI_patch = self.patch_sampler.get_patch_pair(CT_tensor, MRI_tensor)

        CT_patch = CT_patch.unsqueeze(0)
        MRI_patch = MRI_patch.unsqueeze(0)

        return {'A': MRI_patch, 'B': CT_patch}

    def __len__(self):
        return self.num_datapoints