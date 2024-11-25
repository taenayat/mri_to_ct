import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

PATH = '/mnt/homeGPU/tenayat/mri_to_ct/24_11_19_baseline_400epoch/train/'
data = pd.read_csv(PATH + 'losses_tensorboard.csv')#.drop(['unnamed 0'],axis=1)
columns = data.columns

num_metrics = len(columns)
fig, axes = plt.subplots(nrows=(num_metrics + 2) // 3, ncols=3, figsize=(18, 4 * ((num_metrics + 2) // 3)))
axes = axes.flatten()

# Plot each metric
for i, column in enumerate(columns):
    plot_data = np.array(data[column])
    # plot_data = np.array(data[column][100:])
    # if column in ['Losses/G_AB', 'Losses/D_B']:
    #     plot_data = plot_data[plot_data < 5]
    # plot_data = savgol_filter(plot_data, 20, 3)
    plot_data = savgol_filter(plot_data, 300, 3)
    axes[i].plot(data.index*10/121, plot_data, label=column)
    # axes[i].plot(range(len(plot_data)), plot_data, label=column)
    axes[i].set_title(column)
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Value')
    axes[i].legend()

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout and show the plot
plt.tight_layout()
# plt.show()
plt.savefig(PATH+'loss_vis2.png')
