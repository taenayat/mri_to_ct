import os
import yaml

PATH = '/mnt/homeGPU/tenayat/mri_to_ct'
experiment = '24_11_17_cyclegan_firstrun.yaml'
config = yaml.safe_load(os.path.join(PATH, 'experiments', experiment))
print(config)
# print(config['test']['checkpointing']['load_iter'])

