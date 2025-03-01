import argparse
import logging
import yaml
import pandas as pd
import os

def main(config,selection_metric):
    with open(config, 'r') as file:
        config_params = yaml.safe_load(file)

    val_folder = os.path.join(config_params["train"]["output_dir"],"val", "iter_metrics")

    metrics = []
    
    for root, folders, files in os.walk(val_folder):
        for file in files:
            if "metrics" in file and file.endswith(".csv"):
                iteration = file.split("_")[1]
                iteration = iteration.split(".")[0]
                iter_df = pd.read_csv(os.path.join(root, file), index_col=0)
                iter_df = iter_df.mean()
                iter_df["iter"] = int(iteration)
                metrics.append(iter_df)

    print(metrics)
    metrics_df = pd.concat(metrics, axis=1).T
    metrics_df = metrics_df.sort_values("iter")
    metrics_df = metrics_df.reset_index(drop=True)

    if "mae" or "mse" in selection_metric:
        best_iteration = metrics_df.iloc[metrics_df[selection_metric].idxmin()]["iter"]
    else: 
        best_iteration = metrics_df.iloc[metrics_df[selection_metric].idxmax()]["iter"]

    print("BEST ITERATION {} FOR METRIC {}".format(best_iteration, selection_metric))
                
    config_params["infer"]["checkpointing"]["load_iter"] = int(best_iteration)
    config_params["test"]["checkpointing"]["load_iter"] = int(best_iteration)

    # Save yaml file in the same file 
    with open(config, 'w') as file:
        yaml.dump(config_params, file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the experiment configuration YAML file",
        required=True
    )

    parser.add_argument(
        "-v", 
        "--verbose", 
        type=int, 
        required=False, 
        help="Set the verbosity level of the logger",
        default=0
    )

    parser.add_argument(
        "--selection-metric",
        type=str,
        default="mae_clean_mask",
        help="Metric to consider to select the best epoch"
    )

    args = parser.parse_args()

    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    main(args.config,args.selection_metric)