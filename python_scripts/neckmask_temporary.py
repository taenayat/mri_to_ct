import os
import numpy as np
from tqdm import tqdm
import torch
import time
from collections import OrderedDict
import nibabel as nib

#load monai functions
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.networks.nets import UNETR
from monai.data import Dataset
from monai.transforms import MapTransform

# device and directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
DATA_PARENT_DIR = 'brain/'
iterable_dir_list = sorted(os.listdir(DATA_PARENT_DIR), key=lambda x: x)[5:6] # A
# iterable_dir_list = sorted(os.listdir(DATA_PARENT_DIR), key=lambda x: x)[70:71] # B
# iterable_dir_list = sorted(os.listdir(DATA_PARENT_DIR), key=lambda x: x)[130:131] # C
test_files = [{'image':os.path.join(DATA_PARENT_DIR, file_name, 'mr.nii.gz'), 'label':file_name} for file_name in iterable_dir_list]
# test_files = [{'image':os.path.join(DATA_PARENT_DIR, file_name, 'mr_normalized.nii.gz'), 'label':file_name} for file_name in os.listdir(DATA_PARENT_DIR)[3:4]]
GRACE_PATH = 'GRACE.pth'


class DynamicScaleIntensityRanged(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        category = d['label'][2]

        # Determine category based on index and set a_max accordingly
        if category == 'A':
            a_max = 1000  # Category A
        else:
            a_max = 255   # Category B or C

        # Apply the scaling
        mri_img = d["image"]
        scaled_img = (mri_img - 0.0) / (a_max - 0.0) # Simulate ScaleIntensityRanged logic

        # Update image in dictionary
        d["image"] = scaled_img
        return d

# defining tranformations
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        # ZScoreNormalization(keys=["image"], threshold_dict=threshold_dict),
        # ScaleToUint8(keys=["image"], original_min=0, original_max=2000),
        # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
        # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=False),
        # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=False),
        DynamicScaleIntensityRanged(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
    )

# creating the dataset
test_ds = Dataset(
    data=test_files, transform=test_transforms, 
)

# define the model and load weights into it
model = UNETR(
    in_channels=1,
    out_channels=12, #12 for all tissues
    img_size=(64,64,64),
    feature_size=16, 
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

grace_load = torch.load(GRACE_PATH, map_location=torch.device('cpu'), weights_only=False)
grace_load = OrderedDict([(k[7:],v) for k,v in grace_load.items()])

model.load_state_dict(grace_load)
model.eval()

# inferencing and save the result
case_num = len(test_ds)
begin_time = time.time()
for i in range(case_num):
    with torch.no_grad():
        img_name = test_ds[i]['label']
        img = test_ds[i]['image']
        print(img_name)
        input = img.unsqueeze(0).to(device)
        start_time = time.time()
        output = sliding_window_inference(input, (64, 64, 64), 1, model, overlap=0.8)
        print("--- %s seconds ---" % (time.time() - start_time))
    result_image = torch.argmax(output, dim=1).detach().cpu().numpy().squeeze(0)
    result_image_nib = nib.Nifti1Image(result_image, affine=np.eye(4), dtype=np.int64)
    output_path = os.path.join(DATA_PARENT_DIR, img_name, 'mr_segmented.nii.gz')
    nib.save(result_image_nib, output_path)
print('TOTAL TIME:')
print("--- %s seconds ---" % (time.time() - begin_time))
print('DONE :)')
