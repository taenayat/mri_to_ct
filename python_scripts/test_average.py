"""
Add a line to the csv file containing the metrics for test data. currently the test
metrics CSV file adds a line for metrics of each image in the test folder. 
This script adds one line at the end with the average value of each column.
"""

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compute metrics for a given dataset')
parser.add_argument('-d', '--dir', type=str, help='Path to the test CSV file', required=True)
args = parser.parse_args()

YAML_PATH = args.dir 
OUTPUT_PATH =  "/mnt/homeGPU/tenayat/mri_to_ct/meta_test_stats.csv"
OUTPUT_PATH_HR =  "/mnt/homeGPU/tenayat/mri_to_ct/meta_test_stats_hr.csv"

def generate_metrics_path(filepath):
    if ".yaml" in filepath:
        parts = filepath.split('/')
        if "experiments" in parts:
            experiment_name = parts[-1].replace('.yaml', '')
            base_path = '/'.join(parts[:-2])  # Up to "mri_to_ct"
            return f"{base_path}/{experiment_name}/test/metrics.csv"
        else:
            # folder_name = parts[-2]
            experiment_name = parts[-1].replace('.yaml', '')
            base_path = '/'.join(parts[:-2])  # Up to "mri_to_ct"
            return f"{base_path}/{experiment_name}/test/metrics.csv"
    return "Invalid Path"

def extract_name(filepath):
    if ".yaml" in filepath:
        parts = filepath.split('/')
        if "experiments" in parts:
            return parts[-1].replace('.yaml', '')
        else:
            return parts[-3] + "_" + parts[-1].replace('.yaml', '')
    return "Invalid Path"

print(YAML_PATH)
print(generate_metrics_path(YAML_PATH))
print(extract_name(YAML_PATH))

metrics_df = pd.read_csv(generate_metrics_path(YAML_PATH))

try:
    meta_test_df = pd.read_csv(OUTPUT_PATH)
except (pd.errors.EmptyDataError, FileNotFoundError):
    print("CSV file is empty. Creating an empty DataFrame.")
    meta_test_df = pd.DataFrame()
try:
    meta_test_df_human_readable = pd.read_csv(OUTPUT_PATH)
except (pd.errors.EmptyDataError, FileNotFoundError):
    print("CSV file is empty. Creating an empty DataFrame.")
    meta_test_df_human_readable = pd.DataFrame()

# meta_test_df = pd.read_csv(OUTPUT_PATH) 
# meta_test_df_human_readable = pd.read_csv(OUTPUT_PATH_HR)

# meta_test_df = pd.DataFrame() 
# meta_test_df_human_readable = pd.DataFrame()


# avg_df = metrics_df.mean(axis=0)
# std_df = metrics_df.std(axis=0)
# # print(avg_df.to_frame().T)
# # print(final.tail())
# # final.to_csv('../mri_to_ct/24_11_19_baseline_400epoch/test/metrics2.csv', index=False)

# updated = pd.concat((meta_test_df, avg_df.to_frame().T, std_df.to_frame().T), ignore_index=True)
# updated.to_csv(PATH, index=False)
# updated.to_csv(PATH, index=False)


name = extract_name(YAML_PATH)

avg_df = metrics_df.mean(axis=0).to_frame().T.drop(['Unnamed: 0'], axis=1)
std_df = metrics_df.std(axis=0).to_frame().T.drop(['Unnamed: 0'], axis=1)

# formatted_df = pd.DataFrame({col: f"{avg:.2f} ± {std:.2f}" for col, avg, std in zip(avg_df.columns, avg_df.iloc[0], std_df.iloc[0])})
formatted_df = pd.Series({col: f"{avg:.2f} ± {std:.2f}" for col, avg, std in zip(avg_df.columns, avg_df.iloc[0], std_df.iloc[0])}).to_frame().T
formatted_df.insert(0,'exp_name', name)
meta_test_df_human_readable_updated = pd.concat((meta_test_df_human_readable, formatted_df))
meta_test_df_human_readable_updated.to_csv(OUTPUT_PATH_HR, index=False)

avg_df_renamed = avg_df.rename(columns=lambda x: f"{x}_avg")
std_df_renamed = std_df.rename(columns=lambda x: f"{x}_std")

# Combining the dataframes horizontally
combined_df = pd.concat([avg_df_renamed, std_df_renamed], axis=1)
combined_df.insert(0,'exp_name', name)
meta_test_df = pd.concat((meta_test_df, combined_df))
meta_test_df.to_csv(OUTPUT_PATH, index=False)




