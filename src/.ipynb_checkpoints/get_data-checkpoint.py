# read params
# process them
# return dataframe

import os
import yaml
import pandas as pd
import argparse
import mlflow


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    with mlflow.start_run() as mlrun:
        config = read_params(config_path)
        # print(config)
        data_path = config["data_source"]["original_source"]
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        df = pd.read_csv(data_path, sep=",", encoding='utf-8')
        # df.to_csv(raw_data_path,"diabetes.csv",index=False)
        # print(df.head())
        mlflow.log_artifact(file_path, "data-csv-dir")
        return df

if __name__=="__main__":
    print("Running get data script")
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
        