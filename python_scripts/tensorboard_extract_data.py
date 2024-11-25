import os


PATH = "/mnt/homeGPU/tenayat/mri_to_ct"
experiment = '24_11_19_baseline_400epoch'
# train
# file_name = 'events.out.tfevents.1732035357.selene.3105759.0'
# file_name = 'events.out.tfevents.1732175806.selene.3815386.0'

# test
# file_name = 'events.out.tfevents.1732240698.selene.4137699.0'

# validation
file_name = 'events.out.tfevents.1732035358.selene.3105759.1'
# file_name = 'events.out.tfevents.1732175810.selene.3815386.1'

total_path = os.path.join(PATH, experiment, file_name)

# from tensorboard.backend.event_processing import event_accumulator
# ea = event_accumulator.EventAccumulator(total_path,
#       size_guidance={ # see below regarding this argument
#           event_accumulator.COMPRESSED_HISTOGRAMS: 0,
#           event_accumulator.IMAGES: 0,
#           event_accumulator.AUDIO: 0,
#           event_accumulator.SCALARS: 100,
#           event_accumulator.HISTOGRAMS: 0,
#     })
# ea.Reload()
# print(ea.Tags())
# # print(ea.Scalars())
"""
validation scalars:
'Metrics (val)/ssim', 'Metrics (val)/mse', 'Metrics (val)/nmse', 'Metrics (val)/psnr', 
'Metrics (val)/mae', 'Metrics (val)/ssim_clean_mask', 'Metrics (val)/mse_clean_mask', 
'Metrics (val)/nmse_clean_mask', 'Metrics (val)/psnr_clean_mask', 'Metrics (val)/mae_clean_mask', 
'Metrics (val)/cycle_SSIM'

train scalars:
'Learning Rates/lr_G', 'Learning Rates/lr_D', 'Losses/G_AB', 'Losses/D_B',
'Losses/cycle_A', 'Losses/G_BA', 'Losses/D_A', 'Losses/cycle_B', 'Metrics (train)/D_B_real',
'Metrics (train)/D_B_fake', 'Metrics (train)/D_A_real', 'Metrics (train)/D_A_fake'

test scalars:
'Metrics (test)/ssim', 'Metrics (test)/mse', 'Metrics (test)/nmse', 
'Metrics (test)/psnr', 'Metrics (test)/mae', 'Metrics (test)/ssim_clean_mask', 
'Metrics (test)/mse_clean_mask', 'Metrics (test)/nmse_clean_mask', 
'Metrics (test)/psnr_clean_mask', 'Metrics (test)/mae_clean_mask'
"""


import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event
import pandas as pd


# cols = ['Learning Rates/lr_G', 'Learning Rates/lr_D', 'Losses/G_AB', 'Losses/D_B',
# 'Losses/cycle_A', 'Losses/G_BA', 'Losses/D_A', 'Losses/cycle_B', 'Metrics (train)/D_B_real',
# 'Metrics (train)/D_B_fake', 'Metrics (train)/D_A_real', 'Metrics (train)/D_A_fake']
cols = ['Metrics (val)/ssim', 'Metrics (val)/mse', 'Metrics (val)/nmse', 'Metrics (val)/psnr', 
'Metrics (val)/mae', 'Metrics (val)/ssim_clean_mask', 'Metrics (val)/mse_clean_mask', 
'Metrics (val)/nmse_clean_mask', 'Metrics (val)/psnr_clean_mask', 'Metrics (val)/mae_clean_mask', 
'Metrics (val)/cycle_SSIM']

losses = {}
for col in cols:
    losses[col] = []

rec = 0
for record in tf.data.TFRecordDataset(total_path):
    rec += 1
    if rec % 1000 == 0: print(rec)
    event = Event()
    event.ParseFromString(record.numpy())
    # row = {}
    # eve = 0
    if event.summary:
        for value in event.summary.value:
            # eve += 1
            # if value.tag == 'Losses/cycle_A':  # Replace 'loss' with your scalar tag name
                # print(event.step, value.simple_value, value.tag)
            if value.tag in cols:
                # print('record:',rec,'event',eve, 'val', value.tag)
                # row[value.tag] = value.simple_value
                losses[value.tag].append(value.simple_value)
        # print(row)

df = pd.DataFrame(losses)
print(len(df))
print(df.tail())
    # print(df.tail())
df.to_csv(os.path.join(PATH, experiment,'val_losses1.csv'))




# import pandas as pd
# path = '/mnt/homeGPU/tenayat/mri_to_ct/24_11_19_baseline_400epoch/'
# df1 = pd.read_csv(path + 'train_losses.csv')
# df2 = pd.read_csv(path + 'train_losses2.csv')
# df3 = pd.concat((df1,df2), ignore_index=True)
# df3.to_csv(path+'train/losses_tensorboard.csv', index=False)