# print('we entered the file')

# import numpy as np # type: ignore
# import logging
# import time

# logging.basicConfig(filename='temp_log.log', level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logger.info('Started')
# logger.debug("debog log")
# logger.warning('the first warning')

# start = time.time()
# array = np.array([3,4,5])
# print('array:', array)
# logger.info('array:'+str(array))
# end = time.time()

# print(12/0)

# print("Total time: {:.1f}".format(end-start))
# logger.log(0,'ended')





# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import SimpleITK as sitk

# DATA_PARENT_DIR = '/mnt/homeGPU/tenayat/MTT-Net/data/headneck_train.txt'
# with open(DATA_PARENT_DIR,'r') as f:
#     iterable_dir_list = []
#     for l in f.readlines():
#         iterable_dir_list.append(l[:-1])
# dataset_parent_dir = [os.path.join('/mnt/homeGPU/tenayat/brain/', mri_path) for mri_path in iterable_dir_list]

# def read_img(idx):
#     return sitk.GetArrayFromImage(sitk.ReadImage(dataset_parent_dir[idx]+'/mr.nii.gz'))

# min_values = [1000,1000,1000]
# max_values = [0,0,0]
# for idx in range(len(dataset_parent_dir)):
#     print(idx)
#     img = read_img(idx)
#     shape = img.shape
#     for i in range(3):
#         if shape[i] < min_values[i]:
#             min_values[i] = shape[i]
#         if shape[i] > max_values[i]:
#             max_values[i] = shape[i]
# print('val dataset size:')
# print("min:", min_values)
# print('max:', max_values)





# import os
# train_dir = os.listdir('data/TRAIN/CT')
# val_dir = os.listdir('data/VAL/CT')
# test_dir = os.listdir('data/TEST/CT')

# train_name = [item.split('_')[0] for item in train_dir]
# val_name = [item.split('_')[0] for item in val_dir]
# test_name = [item.split('_')[0] for item in test_dir]

# with open('MTT-Net/data/headneck_train.txt', 'w') as f:
#     for item in train_name:
#         f.write(item + "\n")
# with open('MTT-Net/data/headneck_val.txt', 'w') as f:
#     for item in val_name:
#         f.write(item + "\n")
# with open('MTT-Net/data/headneck_test.txt', 'w') as f:
#     for item in test_name:
#         f.write(item + "\n")




import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

# Load the image
# experiments = ['24_11_28_freshstart','24_12_02_new_lr','24_12_06_new_patch','24_12_04_squeeze',
#     '24_12_09_pix2pix', '24_12_15_dynamic_padding', '24_12_24_final','25_02_28_thresh150_correctfunc']
experiments = ['24_12_18_vnet']
sub_experiment = 'first_layer'

for experiment in experiments:
    # image_path = f"mri_to_ct/{experiment}/test/images/0_real_A-fake_B-real_B-clean_mask.png"
    image_path = f"mri_to_ct/{experiment}/{sub_experiment}/test/images/0_real_A-fake_B-real_B-clean_mask.png"
    img = mpimg.imread(image_path)
    height, width = img.shape[:2]
    new_width = int(width * 0.75)
    cropped_img = img[:, :new_width]
    
    # Save the cropped images in experiment-specific folders
    os.makedirs("saved_images", exist_ok=True)
    plt.imsave(f"saved_images/{experiment}_{sub_experiment}.png", cropped_img)

