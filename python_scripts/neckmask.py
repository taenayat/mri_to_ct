print('ENTERED THE FILE!')
import os
import numpy as np
from tqdm import tqdm
import torch
import time
from collections import OrderedDict
import nibabel as nib
import logging

# logging.basicConfig(filename='neckmask_log.log', level=logging.INFO)
# logger = logging.StreamHandler()
# logger.setLevel(logging.DEBUG)
# # logger = logging.getLogger(__name__)
# logger.setFormatter(logging.Formatter("%(asctime)s;%(levelname)s;%(message)s"))
logging.basicConfig(filename='neckmask_log.log', level=logging.DEBUG)
logger = logging.getLogger("neck_mask")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info('Started!')

#load monai functions
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ToTensord,
)
from monai.networks.nets import UNETR
from monai.data import Dataset
from monai.transforms import MapTransform

print('LOADED THE LIBRARYIES')
logger.info('LOADED THE LIBRARIES')
# device and directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
logger.info('DEVICE: ' + str(device))
DATA_PARENT_DIR = 'brain/'
iterable_dir_list = sorted(os.listdir(DATA_PARENT_DIR), key=lambda x: x)
test_files = [{'image':os.path.join(DATA_PARENT_DIR, file_name, 'mr.nii.gz'), 'label':file_name} for file_name in iterable_dir_list]
GRACE_PATH = 'GRACE.pth'

class DynamicScaleIntensityRanged(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        category = d['label'][2]
        if category == 'A':
            a_max = 1000  # Category A
        else:
            a_max = 255   # Category B or C

        mri_img = d["image"]
        scaled_img = (mri_img - 0.0) / (a_max - 0.0) # Simulate ScaleIntensityRanged logic
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
        # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
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

print('MODEL LOADED SUCCESSFULLY')
logger.info('MODEL LOADED SUCCESSFULLY')

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
        logger.info('index: ' + str(i) + 'image name:' + str(img_name))
        logger.info("--- %s seconds ---" % (time.time() - start_time))
    result_image = torch.argmax(output, dim=1).detach().cpu().numpy().squeeze(0)
    result_image_nib = nib.Nifti1Image(result_image, affine=np.eye(4), dtype=np.int64)
    output_path = os.path.join(DATA_PARENT_DIR, img_name, 'mr_segmented2.nii.gz')
    nib.save(result_image_nib, output_path)
print('TOTAL TIME:')
print("--- %s seconds ---" % (time.time() - begin_time))
print('DONE :)')
logger.info('TOTAL TIME:')
logger.info("--- %s seconds ---" % (time.time() - begin_time))
logger.info('DONE!')
