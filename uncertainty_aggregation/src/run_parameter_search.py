import argparse

import configs.uq_config as uq_conf
from api.loading_data import load_or_prepare_metrics
from uncertainty_quantification.parameter_search import optimize_params

parser = argparse.ArgumentParser(
    description="Perform parameter search for metrics constellation as given in src.uncertainty_quantification.uq_config.")
parser.add_argument("--df-path", dest="df_path")

df_path = "/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64"
gt_path = "/home/riedlinger/datasets_ground_truth/KITTI/csv"

if __name__ == "__main__":
    d = {"model": "logistic",
         "score_thresholds": [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5],
         "scoring_method": "roc_auc",
         "augmentation_method": False,
         "pca_transform": 15,
         "metrics_const": ["score", "gradient_metrics"],
         "path": "/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64",
         "n_jobs": 16}

    metrics_constellation = ["output"]
    # Get variables and targets to perform parameter search for.
    variables_df, target_df = load_or_prepare_metrics(
        df_path, uq_conf.aggregation_model, metrics_constellation)
    optimize_params(variables_df, target_df, augmentation_method="smote",
                    metrics_const=metrics_constellation, n_jobs=16)
