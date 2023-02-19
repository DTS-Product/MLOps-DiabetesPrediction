# load the train and test dataset
# Training algorithm
# save the metrics, params

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from get_data import read_params
import argparse
import joblib
import json
import mlflow
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return accuracy, precision, recall, roc_auc

def train_and_evaluate(config_path):
    with mlflow.start_run() as parent_run:
        config = read_params(config_path)
        train_data_path = config["split_data"]["train_path"]
        test_data_path = config["split_data"]["test_path"]
        random_state = config["base"]["random_state"]
        model_dir = config["model_dir"]

        penalty = config["estimators"]["LogisticRegression"]["params"]["penalty"]
        l1_ratio = config["estimators"]["LogisticRegression"]["params"]["l1_ratio"]
        solver = config["estimators"]["LogisticRegression"]["params"]["solver"]

        target = [config["base"]["target_col"]]

        train = pd.read_csv(train_data_path, sep=",")
        test = pd.read_csv(test_data_path, sep=",")
        
        df = pd.concat([test, train])
        
        tabular_train_data = Tabular(train, target_column='Outcome')
        tabular_test_data = Tabular(test, target_column='Outcome')
        
        tabular_data = Tabular(df, target_column='Outcome')
        
        transformer = TabularTransform(
            target_transform=Identity()
        ).fit(tabular_data)
        
        train_x = transformer.transform(tabular_train_data)
        test_x = transformer.transform(tabular_test_data)
                
        
        train_y = train_x[:, -1]
        test_y = test_x[:, -1]
        preprocess = lambda z: transformer.transform(z)
        
        # train = transformer.transform(train)
        # test = transformer.transform(test)
        
        train_x = train_x[:, :-1]
        test_x = test_x[:, :-1]
        
        
        
        lr = LogisticRegression(
            solver=solver,
            penalty=penalty,
            l1_ratio=l1_ratio,
            random_state=random_state)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        
        train_data = transformer.invert(train_x)
        test_data = transformer.invert(test_x)
        (accuracy, precision, recall, roc_auc) = eval_metrics(test_y, predicted_qualities)
        
        # Initialize a TabularExplainer
        explainers = TabularExplainer(
            explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
            mode="regression",
            data=train_data,
            model=lr,
            preprocess=preprocess,
            params={
                "lime": {"kernel_width": 3},
                "shap": {"nsamples": 100}
            }
        )
        # Generate explanations
        test_instances = test_data[0:5]
        local_explanations = explainers.explain(X=test_instances)
        global_explanations = explainers.explain_global(
            params={"pdp": {"features": list(train).remove('Outcome')}}
        )
        
        index=0
        print("LIME results:")
        fig1 = local_explanations["lime"]._plotly_figure(index)
        fig1.write_image('./images/local_lime_exp_1.png')
        mlflow.log_artifact('./images/local_lime_exp_1.png', "Local Lime Explanation")
        print("SHAP results:")
        fig2=local_explanations["shap"]._plotly_figure(index)
        fig2.write_image('./images/local_shap_exp_2.png')
        mlflow.log_artifact('./images/local_shap_exp_2.png', "Local Shap Explanation")
        print("Sensitivity results:")
        fig3=global_explanations["sensitivity"]._plotly_figure()
        fig3.write_image('./images/global_sensitivity_exp_3.png')
        mlflow.log_artifact('./images/global_sensitivity_exp_3.png', "Global Sensitivity Explanation")
        print("PDP results:")
        fig4=global_explanations["pdp"]._plotly_figure()
        fig4.write_image('./images/global_pdp_exp_4.png')
        mlflow.log_artifact('./images/global_pdp_exp_4.png', "Global Partial Dependency Plot")
        
        
        print("LogisticRegression model (solver = %s)" % solver)
        print("LogisticRegression model (penalty = %s)" % penalty)
        print("LogisticRegression model (l1_ratio = %f)" % l1_ratio)
        print("  accuracy : %f" % accuracy)
        print("  precision: %f" % precision)
        print("  recall: %f" % recall)
        print("  area under roc: %f" % roc_auc)

        scores_file = config["reports"]["scores"]   #updating scores from scores.json
        params_file = config["reports"]["params"]   #updating parameters from params.json

        with open(scores_file, "w") as f:
            scores = {
                "accuracy_score": accuracy,
                "precision_score": precision,
                "recall_score": recall,
                "roc_auc_score": roc_auc
            }
            json.dump(scores, f, indent=4)

        with open(params_file, "w") as f:
            params = {
                "solver": solver,
                "penalty": penalty,
                "l1_ratio": l1_ratio
            }
            json.dump(params, f, indent=4)


        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")

        joblib.dump(lr, model_path) #saving the model
        mlflow.log_param("training_random_state", random_state)
        mlflow.log_param("Logistic Regression - Solver", solver)
        mlflow.log_param("Logistic Regression - Penalty", penalty)
        mlflow.log_param("Logistic Regression - l1 ratio", l1_ratio)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.sklearn.log_model(lr, "model")



if __name__=="__main__":
    print("Running training and evaluation script")
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)