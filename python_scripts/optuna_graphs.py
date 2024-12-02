import os
import optuna
import pickle

PARENT_PATH = "/mnt/homeGPU/tenayat/mri_to_ct/24_11_29_lr/"
with open(os.path.join(PARENT_PATH, 'study.pkl') , 'rb') as file:
    study = pickle.load(file)

def number2string_formatter(lr):
    base, exponent = f"{lr:.1e}".split('e')
    base = base.replace('.','_')
    return f"{base}e{abs(int(exponent))}"

print('BEST PARAMETERS:')
print(study.best_trial.params)
print('FORMATTED:', [number2string_formatter(v) for k,v in study.best_trial.params.items()])

fig1 = optuna.visualization.plot_optimization_history(study)
fig1.write_image(os.path.join(PARENT_PATH, 'optimization_history.png'))

fig2 = optuna.visualization.plot_parallel_coordinate(study)
fig2.write_image(os.path.join(PARENT_PATH, 'parallel_coordinate.png'))

fig3 = optuna.visualization.plot_contour(study)
fig3.write_image(os.path.join(PARENT_PATH, 'contour.png'))

fig4 = optuna.visualization.plot_slice(study)
fig4.write_image(os.path.join(PARENT_PATH, 'slice.png'))

fig5 = optuna.visualization.plot_param_importances(study)
fig5.write_image(os.path.join(PARENT_PATH, 'param_importance.png'))

fig6 = optuna.visualization.plot_edf(study)
fig6.write_image(os.path.join(PARENT_PATH, 'empirical_distribution.png'))

fig7 = optuna.visualization.plot_rank(study)
fig7.write_image(os.path.join(PARENT_PATH, 'rank.png'))

fig8 = optuna.visualization.plot_timeline(study)
fig8.write_image(os.path.join(PARENT_PATH, 'timeline.png'))
