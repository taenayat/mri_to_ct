import os
import subprocess
import yaml
import optuna
import pandas as pd
import numpy as np
import pickle


YAML_PATH = '/mnt/homeGPU/tenayat/mri_to_ct/experiments/24_11_29_lr.yaml'
PARENT_PATH = "/mnt/homeGPU/tenayat/mri_to_ct/24_11_29_lr/"
os.makedirs(PARENT_PATH, exist_ok=True)

with open(YAML_PATH, 'r') as file:
    conf = yaml.safe_load(file)

def number2string_formatter(lr):
    base, exponent = f"{lr:.1e}".split('e')
    base = base.replace('.','_')
    return f"{base}e{abs(int(exponent))}"

def objective(trial):

    # gan_lr = trial.suggest_loguniform("gan_lr", 1e-6, 1e-3)
    # disc_lr = trial.suggest_loguniform("disc_lr", 1e-6, 1e-3)
    gan_lr = trial.suggest_float("gan_lr", 1e-6, 1e-3, log=True)
    disc_lr = trial.suggest_float("disc_lr", 1e-6, 1e-3, log=True)

    conf['train']['gan']['optimizer']['lr_G'] = gan_lr
    conf['train']['gan']['optimizer']['lr_D'] = disc_lr

    # print('gen', gan_lr, 'disc', disc_lr)
    # print('formatted:', number2string_formatter(gan_lr), number2string_formatter(disc_lr))

    experiment_name = f"g{number2string_formatter(gan_lr)}_d{number2string_formatter(disc_lr)}"
    output_path = os.path.join(PARENT_PATH, experiment_name)
    config_path = os.path.join(output_path, experiment_name+'.yaml')
    conf['train']['output_dir'] = output_path

    # print('experiment:', experiment_name, 'config:', config_path, 'output:', output_path)
    print('experiment:', experiment_name, 'gan lr:', gan_lr, 'disc lr:', disc_lr)

    os.makedirs(output_path, exist_ok=True)
    with open(config_path, 'w') as file:
        yaml.dump(conf, file)

    subprocess.run(f"ganslate train config={config_path}")
    subprocess.run(f"ganslate test config={config_path}")

    # TEMP
    # output_path = '/mnt/homeGPU/tenayat/mri_to_ct/24_11_28_freshstart'

    metrics_df = pd.read_csv(os.path.join(output_path, 'test', 'metrics.csv'), index_col=0)
    metrics_df = metrics_df.mean()
    # print(metrics_df)

    mae = metrics_df.mae_clean_mask
    # metric = np.random.random()
    print('------------------------------')
    print('MAE CLEAN MASK value:', mae)
    print('------------------------------')
    return mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, gc_after_trial=True)
with open(os.path.join(PARENT_PATH, 'study.pkl'), 'wb') as file:
    pickle.dump(study, file)
