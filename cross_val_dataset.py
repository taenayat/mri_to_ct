import argparse
import matplotlib.pyplot as plt
import glob
import os
import cv2
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import shutil
# from preprocess import main as preprocess
import yaml
import tqdm

def read_yaml(config):
    with open(config, 'r') as file:
        config_params = yaml.safe_load(file)

    if config_params is None:
        config_params = {}

    return config_params

def cases_folds_maker(cases, n_folds, random_state=3):
    for train_ids, test_ids in KFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    ).split(cases):
        yield [cases[i] for i in train_ids], [cases[i] for i in test_ids]


def copy_data(df, output, folder_name):

   for i, row in df.iterrows():
        mri_path = row["mri_paths"]
        mri_patient_name = mri_path.split("/")[-2]
        ct_path = row["ct_paths"]
        ct_patient_name = ct_path.split("/")[-2]
        mask_path = row["mask_paths"]
        mask_patient_name = mask_path.split("/")[-2]

        os.makedirs(os.path.join(output, folder_name, "MRI"), exist_ok=True)
        dst_mri = os.path.join(output, folder_name, "MRI", mri_patient_name + "_" + os.path.basename(mri_path))

        # print(mri_path, dst_mri)
        shutil.copy(mri_path, dst_mri)

        os.makedirs(os.path.join(output, folder_name, "CT"), exist_ok=True)
        dst_ct = os.path.join(output, folder_name, "CT", ct_patient_name + "_" + os.path.basename(ct_path))

        # print(ct_path, dst_ct)
        shutil.copy(ct_path, dst_ct)

        # if folder_name != "TRAIN":
        os.makedirs(os.path.join(output, folder_name, "MASKS"), exist_ok=True)
        dst_mask = os.path.join(output, folder_name, "MASKS", mask_patient_name + "_" + os.path.basename(mask_path))

        # print(mask_path, dst_mask)
        shutil.copy(mask_path, dst_mask)

def main(parent_path, output, folds=0, leave_one_out=False): #, prep_config, __dev_bool, num_processes):

    # Create output folders
    os.makedirs(output, exist_ok=True)
    output_path = output

    # Read data paths
    mri_paths = glob.glob("{}/*/mr.nii.gz".format(parent_path), recursive=True)
    # mri_paths = glob.glob("{}/*.nii.gz".format(mri_path), recursive=True)
    mri_paths.sort()

    # ct_paths = glob.glob("{}/*.nii.gz".format(ct_path), recursive=True)
    ct_paths = glob.glob("{}/*/ct.nii.gz".format(parent_path), recursive=True)
    ct_paths.sort()

    # mask_paths = glob.glob("{}/*.nii.gz".format(mask_path), recursive=True)
    mask_paths = glob.glob("{}/*/mask.nii.gz".format(parent_path), recursive=True)
    mask_paths.sort()

    # # Get the name of each subject involved
    # mri_names = [os.path.basename(n).split("_")[0] for n in mri_paths]
    # unique_names_mri = list(set(mri_names))
    # unique_names_mri.sort()

    # ct_names= [os.path.basename(n).split("_")[0] for n in ct_paths]
    # unique_names_ct = list(set(ct_names))
    # unique_names_ct.sort()

    # mask_names= [os.path.basename(n).split("_")[0] for n in mask_paths]
    # unique_names_mask = list(set(mask_names))
    # unique_names_mask.sort()

    # # Check if mri and ct have the same subjects
    # assert unique_names_mri == unique_names_ct == unique_names_mask

    # Dictionary of GT images   
    data_dict= {"mri_paths": mri_paths, "ct_paths": ct_paths, "mask_paths": mask_paths}
        
    # Create a dataframe
    df=pd.DataFrame.from_dict(data_dict)
    df.reset_index(inplace=True, drop=True)

    # Split the data in train and test
    train_data, test_data = train_test_split(mri_paths, test_size=0.25, random_state=RANDOM_STATE, shuffle=True)

    # Exclude the test data from the mri_paths
    mri_paths = train_data

    # if __dev_bool:
    #     print(df.head())

    if leave_one_out:
        folds = len(mri_paths)

    if folds > 1:
        train_folds, val_folds = [], []
        for train_cases, val_cases in cases_folds_maker(mri_paths, folds, RANDOM_STATE):
            train_folds.append(train_cases)
            val_folds.append(val_cases)
    else: 
        train_folds, val_folds= train_test_split(mri_paths, test_size=0.1, random_state=RANDOM_STATE, shuffle=True)
        train_folds = [train_folds]
        val_folds = [val_folds]

    i=0
    for train_cases, val_cases in tqdm.tqdm(zip(train_folds, val_folds), total=len(train_folds)):
        # Create directories
        if folds > 1:
            os.makedirs(os.path.join(output, "fold_{}".format(i)), exist_ok=True)
            output_path = os.path.join(output, "fold_{}".format(i))
           
        # Create a dataframe for each dataset
        df_train = df.loc[df.mri_paths.isin(train_cases)]
        df_val= df.loc[df.mri_paths.isin(val_cases)]
        
        # if __dev_bool:
        #     print(df_train.head())
        #     print(df_val.head())
        #     print(df_test.head())

        #Save csv files
        df_train.to_csv(os.path.join(output_path, "train.csv"), index=None)
        df_val.to_csv(os.path.join(output_path, "val.csv"), index=None)
    
        i+=1

        copy_data(df_train, output_path, "TRAIN")
        copy_data(df_val, output_path, "VAL")

    # Save test data separately 
    df_test=df.loc[df.mri_paths.isin(test_data)]
    df_test.to_csv(os.path.join(output, "test.csv"), index=None)
    copy_data(df_test, output, "TEST") 

    # if prep_config is not None:
    #     preprocess(output, prep_config, output, n_process=num_processes, no_confirm=True)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add basic arguments
    
    parser.add_argument("--parent-path",
                        type=str,
                        help="Path to the parent directory of images",
                        required=True)
    
    # parser.add_argument("--ct-path",
    #                     type=str,
    #                     help="Path to the CT target images",
    #                     required=True)
    
    # parser.add_argument("--mask-path",
    #                     type=str,
    #                     help="Path to the mask target images",
    #                     required=True)
    
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output folder to save the resulting dataset split.",
                        required=True)
    
    # parser.add_argument("-f", "--folds",
    #                     type=int,
    #                     default=5)
    
    # parser.add_argument("--leave-one-out",
    #                     action="store_true",
    #                     help="Use leave one out cross-validation",
    #                     default=False)
    
    # parser.add_argument("-d", "--debug",
    #                     action="store_true",
    #                     help="Print debugging information",
    #                     default=False)
    
    # parser.add_argument("-s", "--seed",
    #                     type=int,
    #                     help="Set random seed for reproducibility",
    #                     default=161195)
    
    # parser.add_argument("--prep-config",
    #                     type=str,
    #                     help="Path to the configuration file for the preprocessing steps.")
    
    # parser.add_argument( "-p", "--n-processes", 
    #                     type=int,
    #                     help="Number of processes to execute in parallel", 
    #                     required=False, 
    #                     default=5)

    # Parse arguments
    args = parser.parse_args()

    # RANDOM_STATE = args.seed
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)

    # __dev_bool = args.debug

    # main(args.mri_path, args.ct_path, args.mask_path, args.output, args.folds, args.leave_one_out, args.prep_config, __dev_bool, args.n_processes)
    main(args.parent_path, args.output)
