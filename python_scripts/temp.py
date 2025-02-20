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





import os
train_dir = os.listdir('data/TRAIN/CT')
val_dir = os.listdir('data/VAL/CT')
test_dir = os.listdir('data/TEST/CT')

train_name = [item.split('_')[0] for item in train_dir]
val_name = [item.split('_')[0] for item in val_dir]
test_name = [item.split('_')[0] for item in test_dir]

with open('MTT-Net/data/headneck_train.txt', 'w') as f:
    for item in train_name:
        f.write(item + "\n")
with open('MTT-Net/data/headneck_val.txt', 'w') as f:
    for item in val_name:
        f.write(item + "\n")
with open('MTT-Net/data/headneck_test.txt', 'w') as f:
    for item in test_name:
        f.write(item + "\n")




