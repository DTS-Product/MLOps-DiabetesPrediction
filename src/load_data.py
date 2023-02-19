# read data from local/remote data source
# save it in data/raw for further processing


import os
from get_data import read_params, get_data
import argparse
import mlflow


def load_and_save(config_path):
    with mlflow.start_run() as mlrun:
        config = read_params(config_path)
        df = get_data(config_path)
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        print(raw_data_path)
        df.to_csv(raw_data_path, sep=",", index=False)


if __name__=="__main__":
    print("Running load data script")
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)