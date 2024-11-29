import os
import optuna
import pickle

PARENT_PATH = "/mnt/homeGPU/tenayat/mri_to_ct/24_11_29_lr/"
with open(os.path.join(PARENT_PATH, 'study.pkl') , 'rb') as file:
    study = pickle.load(file)

fig1 = optuna.visualization.plot_optimization_history(study)
fig1.write_image(os.path.join(PARENT_PATH, 'optimization_history.png'))

fig2 = optuna.visualization.plot_parallel_coordinate(study)
fig2.write_image(os.path.join(PARENT_PATH, 'parallel_coordinate.png'))

fig3 = optuna.visualization.plot_contour(study)
fig3.write_image(os.path.join(PARENT_PATH, 'contour.png'))
