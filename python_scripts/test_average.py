"""
Add a line to the csv file containing the metrics for test data. currently the test
metrics CSV file adds a line for metrics of each image in the test folder. 
This script adds one line at the end with the average value of each column.
"""

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compute metrics for a given dataset')
parser.add_argument('-d', '--dir', type=str, help='Path to the test/val CSV file', required=True)
args = parser.parse_args()

PATH = args.dir 
metrics_df = pd.read_csv(PATH)
avg_df = metrics_df.mean(axis=0)
# print(avg_df.to_frame().T)
final = pd.concat((metrics_df, avg_df.to_frame().T), ignore_index=True)
# print(final.tail())
# final.to_csv('../mri_to_ct/24_11_19_baseline_400epoch/test/metrics2.csv', index=False)
final.to_csv(PATH, index=False)
