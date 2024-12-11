import os
import subprocess
import yaml
import optuna
import pandas as pd
import numpy as np
import pickle


YAML_PATH = '/mnt/homeGPU/tenayat/mri_to_ct/experiments/24_12_10_lambda.yaml'
PARENT_PATH = "/mnt/homeGPU/tenayat/mri_to_ct/24_12_10_lambda/"
os.makedirs(PARENT_PATH, exist_ok=True)

with open(YAML_PATH, 'r') as file:
    conf = yaml.safe_load(file)

def exp_name_formatter(lambda_pix2pix):
    return f"{lambda_pix2pix:.1f}"

def objective(trial):

    lambda_pix2pix = trial.suggest_float('lambda_pix2pix', 1, 200)
    conf['train']['gan']['optimizer']['lambda_pix2pix'] = lambda_pix2pix

    experiment_name = exp_name_formatter(lambda_pix2pix)
    output_path = os.path.join(PARENT_PATH, experiment_name)
    config_path = os.path.join(output_path, experiment_name+'.yaml')
    conf['train']['output_dir'] = output_path

    # print('experiment:', experiment_name, 'config:', config_path, 'output:', output_path)
    print('experiment:', experiment_name, 'lambda:', lambda_pix2pix)

    os.makedirs(output_path, exist_ok=True)
    with open(config_path, 'w') as file:
        yaml.dump(conf, file)

        subprocess.run(f"ganslate train config={config_path}", shell=True)
        subprocess.run(f"ganslate test config={config_path}", shell=True)
        subprocess.run(f"python python_scripts/test_average.py -d {config_path}", shell=True)

        metrics_df = pd.read_csv(os.path.join(output_path, 'test', 'metrics.csv'), index_col=0)
        metrics_df = metrics_df.mean()

    mae = metrics_df.mae_clean_mask
    print('MAE CLEAN MASK value:', mae)
    print('------------------------------')
    # mae = np.random.random()
    return mae

study = optuna.create_study(direction="minimize")

# study.enqueue_trial({'channel':32, 'width':160})

study.optimize(objective, n_trials=10, gc_after_trial=True)
with open(os.path.join(PARENT_PATH, 'study.pkl'), 'wb') as file:
    pickle.dump(study, file)
