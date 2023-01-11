import os

from api.loading_data import load_or_prepare_metrics, load_metrics_and_targets
from configs.uq_config import df_path, metrics_constellation
import configs.uq_config as uq_conf
from src.uncertainty_quantification.train_and_evaluate import train_and_evaluate
from uncertainty_quantification.parameter_search import optimize_params


def uncertainty_evaluation(var_df, tar_df,
                           model=uq_conf.aggregation_model,
                           score_thresholds=uq_conf.PARAMETER_SEARCH_SCORE_THRESHOLDS,
                           scoring_method=None,
                           augmentation_method=None,
                           pca_transform=False,
                           metrics_const=uq_conf.metrics_constellation,
                           path=uq_conf.df_path,
                           n_jobs=16):
    """

    :param var_df:
    :param tar_df:
    :param model:
    :param score_thresholds:
    :param scoring_method:
    :param augmentation_method:
    :param pca_transform:
    :param metrics_const:
    :param path:
    :param n_jobs:
    """
    param_dir = f"{df_path}/{model}/{'+'.join(metrics_constellation)}"
    os.makedirs(param_dir, exist_ok=True)

    best_params = optimize_params(var_df, tar_df, model, score_thresholds, scoring_method,
                                  augmentation_method, pca_transform, n_jobs, metrics_const, path)
    train_and_evaluate(var_df, tar_df, best_params, path)


d = {"model": "linear_regression",
     "score_thresholds": [0.0],
     "scoring_method": "r2",
     "augmentation_method": False,
     "pca_transform": False,
     "metrics_const": ["output", "meta_detect"],
     "path": "/home/riedlinger/MetaDetect-TestEvaluation/test_preparation",
     "n_jobs": 1}

if __name__ == "__main__":
    p, m, c = d["path"], d["model"], d["metrics_const"]
    variables_df, targets_df = load_metrics_and_targets(p, m, c)

    c += ["output_metrics"]
    variables_df, target_df = load_or_prepare_metrics(p, m, c)
    uncertainty_evaluation(variables_df, target_df, **d)
