import os
import subprocess
import yaml
import optuna
import pandas as pd
import numpy as np
import pickle


YAML_PATH = '/mnt/homeGPU/tenayat/mri_to_ct/experiments/24_12_05_patch.yaml'
PARENT_PATH = "/mnt/homeGPU/tenayat/mri_to_ct/24_12_05_patch/"
os.makedirs(PARENT_PATH, exist_ok=True)

with open(YAML_PATH, 'r') as file:
    conf = yaml.safe_load(file)

def exp_name_formatter(channel, width):
    return f"{channel}_{width}_{width}"

def objective(trial):

    # channel = trial.suggest_int('channel', 10, 163)
    # width = trial.suggest_int('width', 10, 172)
    channel = trial.suggest_categorical('channel', [160,128,64,32,16])
    # width = trial.suggest_categorical('width', [16])
    width = trial.suggest_categorical('width', [160,128,64,32,16])
    # print(channel,width)
    conf['train']['dataset']['patch_size'] = [channel, width, width]

    experiment_name = exp_name_formatter(channel, width)
    output_path = os.path.join(PARENT_PATH, experiment_name)
    config_path = os.path.join(output_path, experiment_name+'.yaml')
    conf['train']['output_dir'] = output_path

    # print('experiment:', experiment_name, 'config:', config_path, 'output:', output_path)
    print('experiment:', experiment_name, 'channel:', channel, 'width:', width)

    already_run_sub_experiments = [(32,160), (64,32)]
    illegal_sub_experiments = [(160,160),(160,128),(128,160)]

    if (channel,width) in already_run_sub_experiments:
        metrics_df = pd.read_csv(os.path.join(output_path, 'test', 'metrics.csv'), index_col=0)
        metrics_df = metrics_df.mean()

    elif (channel,width) in illegal_sub_experiments:
        raise optuna.TrialPruned()
    
    else:
        os.makedirs(output_path, exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(conf, file)

        subprocess.run(f"ganslate train config={config_path}", shell=True)
        subprocess.run(f"ganslate test config={config_path}", shell=True)

        metrics_df = pd.read_csv(os.path.join(output_path, 'test', 'metrics.csv'), index_col=0)
        metrics_df = metrics_df.mean()

    mae = metrics_df.mae_clean_mask
    print('MAE CLEAN MASK value:', mae)
    print('------------------------------')
    # mae = np.random.random()
    return mae

study = optuna.create_study(direction="minimize")

study.enqueue_trial({'channel':32, 'width':160})
study.enqueue_trial({'channel':64, 'width':32})

study.optimize(objective, n_trials=10, gc_after_trial=True)
with open(os.path.join(PARENT_PATH, 'study.pkl'), 'wb') as file:
    pickle.dump(study, file)
