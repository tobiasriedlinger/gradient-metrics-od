import numpy as np
import os
import pandas as pd
import tqdm

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.models.gradient_boosting import gb_classifier_from_parameter_dict
from src.models.logistic_regression import logistic_regression_from_parameter_dict
from src.uncertainty_quantification.performance_evaluation import evaluate_classifier_performance


get_model_from_parameter_dict = {"gb_classifier": gb_classifier_from_parameter_dict,
                                 "logistic": logistic_regression_from_parameter_dict}


def threshold_variables_and_targets(var_df, target_df, column="s", threshold=0.3):
    id_str = "dataset_box_id"
    variables_thresh = var_df[var_df[column] >= threshold]
    targets_thresh = target_df[target_df[id_str].isin(
        variables_thresh[id_str])]

    variables_thresh = variables_thresh.drop(id_str, axis=1)
    targets_thresh = targets_thresh.drop(id_str, axis=1)

    return variables_thresh, targets_thresh


def train_and_evaluate_classifiers(var_df, target_df, parameter_dict, path, augmentation="smote"):
    model_name = parameter_dict["model"]

    print("Evaluation...")
    for score_thr in parameter_dict["parameters"].keys():
        print("Score threshold = {}".format(score_thr))
        res_path = f"{path}/{model_name}/{score_thr}"
        os.makedirs(res_path, exist_ok=True)
        params = parameter_dict["parameters"][score_thr]
        print(f"Parameters: {params}")

        variables_thresh, targets_thresh = threshold_variables_and_targets(
            var_df, target_df, threshold=float(score_thr))
        variables_thresh, targets_thresh = np.array(
            variables_thresh), np.ravel(targets_thresh)
        targets_thresh = targets_thresh >= 0.45

        acc_ls,  auroc_ls = [], []

        for i in tqdm.tqdm(range(10)):
            x_train, x_test, y_train, y_test = train_test_split(
                variables_thresh, targets_thresh, test_size=0.2)
            if augmentation:
                oversample = SMOTE()
                x_train, y_train = oversample.fit_resample(x_train, y_train)

            model = get_model_from_parameter_dict[parameter_dict["model"]](
                params)
            print("Fitting model...")
            model.fit(x_train, y_train)
            if "gb" in model_name:
                model.save_model(f"{res_path}/model_{i}.model")
            print("Predicting...")
            y_pred = model.predict_proba(x_test)[:, 1]

            comp_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
            comp_df.to_csv(f"{res_path}/eval_data_{i}.csv")

            print("Evaluating...")
            acc, auroc = evaluate_classifier_performance(y_test, y_pred)
            acc_ls.append(acc)
            auroc_ls.append(auroc)

        eval_results = pd.DataFrame({"accuracy": acc_ls, "auroc": auroc_ls})
        eval_results.loc["mean"] = eval_results.mean()
        eval_results.loc["std"] = eval_results.std()
        eval_results.to_csv(f"{res_path}/evaluation_results.csv")

        print(f"Evaluation results for eps_s = {score_thr}: ")
        print(eval_results)
