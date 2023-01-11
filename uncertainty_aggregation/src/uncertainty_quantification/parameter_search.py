import json
import os
import time

import numpy as np
import pandas as pd

from api.pre_process import center_pd_columns
import configs.models_config as models_cfg
from src.uncertainty_quantification.train_and_evaluate import threshold_variables_and_targets
from src.meta_detect_metrics.md_metrics import add_meta_detect_metrics
from data_tools.sampling_methods import get_augmented_vars_and_targets
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from configs import uq_config as uq_conf


def default_scoring(model):
    if model in models_cfg.CLASSIFICATION_MODELS:
        return "accuracy"
    elif model in models_cfg.REGRESSION_MODELS:
        return "r2"
    else:
        return ""


def default_augmentation(model):
    if model in models_cfg.CLASSIFICATION_MODELS:
        return False
    elif model in models_cfg.REGRESSION_MODELS:
        return None
    else:
        return ""


def get_scoring_and_augmentation(scoring_method, augmentation_method, model):
    scoring = scoring_method if scoring_method is not None else default_scoring(
        model)
    augmentation = augmentation_method if augmentation_method is not None else default_augmentation(
        model)

    return scoring, augmentation


def parameter_search(variables_df, targets_df,
                     model="gb_classifier",
                     score_thresholds=[0.1, 0.3, 0.5],
                     scoring="accuracy",
                     augmentation="smote",
                     pca_transform=15,
                     metrics_const=["score"],
                     n_jobs=16):
    """parse_gradient_metrics
    Performs parameter search for given model on given variable and target data. Employs sklearn.model_selection.GridSearchCV.
    Parameter ranges are specified in src.uncertainty_quantification.uq_config.
    :param variables_df: (pandas DataFrame) with uncertainty metrics, contains at least [file_path, dataset_box_id, s]
    :param targets_df: (pandas DataFrame) containing dataset_box_id and meta-classification (tp_label) or meta-regression (true_iou)
                        targets.
    :param model: (str) model type for which to tune parameters. Options are listed in src.uncertainty_quantification.uq_config
                        as entries of CLASSIFICATION_MODELS and REGRESSION_MODELS
    :param score_thresholds: (list[float]) confidence score thresholds above which entries from variables_df and targets_df are
                        considered (filtered in threshold_variables_and_targets).
    :param scoring: (str) model performance measure for which to tune parameters. Either contained in https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                        or needs to be specified accordingly. Common choices: "accuracy", "precision", "recall", "jaccard", "roc_auc"
    :param augmentation: (str) data up/down-sampling method to employ (so-far implemented: None, "smote")
    :param metrics_const: (list[str]) Identifiers for uncertainty metrics classes to use as defined in src.uncertainty_quantification.uq_config.
    :param n_jobs: (int) number of parallel jobs used for parameter optimization (may be sent to GPU)
    :return dict: (dict{"model": , "parameters": , "log": }) return dictionary containing relevant data from parameter search.
        "model" : model,
        "parameters" : {thr : clf.best_params_(thr)},
        "log" : log_dict (specified below)
    """
    log_dict = {"score_thresholds": score_thresholds,
                "scoring": scoring,
                "augmentation": augmentation,
                "pca_transform": pca_transform,
                "parameter_ranges": models_cfg.PARAMETER_SEARCH_OPTIONS[model],
                "metrics": metrics_const}
    results = {}
    for thr in score_thresholds:
        print(f"Score threshold = {thr}")
        variables_thresh, targets_thresh = threshold_variables_and_targets(
            variables_df, targets_df, threshold=thr)

        # This automatically forgets the entry "file_path" in variables_thresh.
        # todo: metrics already computed or compute+save now.
        variables_thresh = add_meta_detect_metrics(
            variables_thresh, metrics_const)
        if "score" not in metrics_const:
            variables_thresh = variables_thresh.drop("s", axis=1)

        ids = list(variables_thresh.columns)
        log_dict["metrics_identifier"] = ids
        print(f"Metrics for parameter search ({len(ids)}): {ids}")

        if augmentation:
            print("Using sampling method {}...".format(augmentation))
            variables_thresh, targets_thresh = get_augmented_vars_and_targets(
                variables_thresh, targets_thresh, augmentation)
            if model in models_cfg.CLASSIFICATION_MODELS:
                print(
                    f"Total data points: {len(targets_thresh)}, tp data points: {np.sum(targets_thresh)}")
            elif model in models_cfg.REGRESSION_MODELS:
                print(f"Data points: {len(targets_thresh)}")

        # todo: outsource pca transformation
        if pca_transform and "gradient_metrics" in metrics_const:
            print(
                f"Applying Principal Component Analysis ({pca_transform} comps) to gradient metrics...")
            gradient_identifiers = [
                s for s in variables_thresh.columns if "grad" in s]
            grads_df = variables_thresh[gradient_identifiers].copy(deep=True)
            variables_thresh = variables_thresh.drop(
                gradient_identifiers, axis=1)
            grads_df = center_pd_columns(grads_df)

            pca = PCA(n_components=pca_transform)
            x_trans = pca.fit_transform(grads_df)
            trans_grads_df = pd.DataFrame(data=x_trans, columns=[
                                          f"grad_{i}" for i in range(pca_transform)], index=variables_thresh.index)

            variables_thresh = pd.concat(
                [variables_thresh, trans_grads_df], axis=1, ignore_index=False)

        assert len(variables_thresh.index) == len(targets_thresh.index)
        targets_thresh = np.ravel(targets_thresh)
        print(20 * "#" + "\nGrid Search: {}\n".format(model) + 20 * "#")
        tunable_model = models_cfg.PARAMETER_SEARCH_MODELS[model]()
        tuned_parameters = models_cfg.PARAMETER_SEARCH_OPTIONS[model]
        clf = GridSearchCV(tunable_model, tuned_parameters,
                           scoring=scoring, verbose=1, n_jobs=n_jobs)

        print("Tuning hyper parameters...")
        clf.fit(variables_thresh, targets_thresh)

        print("Best parameters on development set:")
        print(clf.best_params_)
        results[thr] = clf.best_params_

    return {"model": model, "parameters": results, "log": log_dict}


def optimize_params(var_df, tar_df,
                    model=uq_conf.aggregation_model,
                    score_thresholds=uq_conf.PARAMETER_SEARCH_SCORE_THRESHOLDS,
                    scoring_method=None,
                    augmentation_method=None,
                    pca_transform=False,
                    n_jobs=8,
                    metrics_const=["score"],
                    path="/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64"):

    param_dir = f"{path}/{model}/{'+'.join(metrics_const)}"
    os.makedirs(param_dir, exist_ok=True)

    scoring, augmentation = get_scoring_and_augmentation(
        scoring_method, augmentation_method, model)

    best_params = parameter_search(var_df, tar_df,
                                   model,
                                   score_thresholds,
                                   scoring=scoring,
                                   augmentation=augmentation,
                                   pca_transform=pca_transform,
                                   n_jobs=n_jobs,
                                   metrics_const=metrics_const)
    with open(f"{param_dir}/{time.strftime('%m_%d_%Y_%H:%M:%S')}_params.json", "w") as f:
        json.dump(best_params, f)

    return best_params
