from pathlib import Path
import torch
from ganslate.utils.io import make_dataset_of_files
from ganslate.utils import sitk_utils

import os
import SimpleITK as sitk

from ganslate.data.utils.normalization import min_max_normalize, min_max_denormalize

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
from omegaconf import MISSING

from ganslate import configs

import ganslate.utils.sitk_utils as sitk_utils


@dataclass
class SynthRAD2023InferDatasetConfig(configs.base.BaseDatasetConfig):
    pass

EXTENSIONS = ['.nii.gz']

class SynthRAD2023InferDataset(Dataset):

    def __init__(self, conf):
        root_path = Path(conf.infer.dataset.root).resolve()
        mri_path=root_path / "MRI"
        self.mri_path = make_dataset_of_files(mri_path, EXTENSIONS)
        self.num_datapoints = len(self.mri_path)

    def __getitem__(self, index):
        final_index = index % self.num_datapoints

        mri_sample=self.mri_path[final_index]

        MRI_image = sitk_utils.load(mri_sample)

        MRI_tensor = sitk_utils.get_tensor(MRI_image)

        MRI_tensor = min_max_normalize(MRI_tensor, MRI_tensor.min(), MRI_tensor.max())

        MRI_tensor = MRI_tensor.unsqueeze(0)


        # Metadata is optionally returned by this method, explained at the end of the method.
        # Delete if not necessary.
        metadata = {
            'path': str(self.mri_path[final_index]),
            'origin' : MRI_image.GetOrigin(),
            'spacing' : MRI_image.GetSpacing(),
            'direction' : MRI_image.GetDirection(),
        }

        return {
            # Notice that the key for inference input is not "A"
            "input": MRI_tensor,
            # [Optional] metadata - if `save()` is defined *and* if it requires metadata.
            "metadata": metadata,
        }
     
    def __len__(self):
        # Depending on the dataset dir structure, you might want to change it.
        return self.num_datapoints

    def save(self, tensor, save_dir, metadata=None):
        """ By default, ganslate logs images in png format. However, if you wish
        to save images in a different way, then implement this `save()` method. 
        For example, you could save medical images in their native format for easier
        inspection or usage.
        If you do not need this method, remove it.
        """
        tensor_denom = tensor.clone()
        tensor_denom = min_max_denormalize(tensor_denom, -1024, 3000)

        np_image = tensor_denom.numpy(force=True)
        filename = os.path.basename(metadata["path"])
        filename = filename.split(".")[0]
        filename = filename.split("_")[0]
        filename += "_SCT.nii.gz"
        image = sitk.GetImageFromArray(np_image[0])
        image.SetOrigin(metadata["origin"])
        image.SetSpacing(metadata["spacing"])
        image.SetDirection(metadata["direction"])
        os.makedirs(save_dir, exist_ok=True)
        sitk_utils.write(image, save_dir / filename)