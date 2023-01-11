import os

import pandas as pd
import random
import xgboost as xgb

from sacred import Experiment
from sklearn.metrics import r2_score

from src.api.loading_data import load_single_frame
from src.api.pre_process import standardize_columns

scatter_ex = Experiment("scatter_evaluation")


@scatter_ex.config
def scatter_config():
    setup_path = "/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain"
    # setup_path = "/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti_tracking/0001"
    # setup_path = "/home/riedlinger/UQ/MetaDetect-TestEvaluation/test_preparation/fusion_metrics"
    metrics_id = "two_norms"
    regression_params = dict(max_depth=6,
                             n_estimators=30,
                             reg_alpha=1.0,
                             reg_lambda=0.5,
                             learning_rate=0.3)


def meta_score(metrics_df, iou_df, regression_params, random_seed=0, setup_path=None):
    images = list(set(metrics_df["file_path"]))

    random.seed(random_seed)
    random.shuffle(images)

    split_1 = images[:int(len(images) / 2)]
    split_2 = images[int(len(images) / 2):]

    df_post_1 = metrics_df[metrics_df['file_path'].isin(split_1)]
    iou_post_1 = iou_df[iou_df['file_path'].isin(split_1)]
    df_post_2 = metrics_df[metrics_df['file_path'].isin(split_2)]
    iou_post_2 = iou_df[iou_df['file_path'].isin(split_2)]

    model_1 = xgb.XGBRegressor(
        tree_method="gpu_hist", gpu_id=0, **regression_params)
    model_2 = xgb.XGBRegressor(
        tree_method="gpu_hist", gpu_id=0, **regression_params)

    droppables = [s for s in list(
        df_post_1.columns) if "Unnamed" in s or "file_path" in s or "dataset_box_id" in s]
    model_1.fit(df_post_1.drop(droppables, axis=1), iou_post_1['true_iou'])
    model_2.fit(df_post_2.drop(droppables, axis=1), iou_post_2['true_iou'])

    model_1.save_model(f"{setup_path}/post_nms_regression/latest_model.json")

    iou_1 = model_1.predict(df_post_2.drop(droppables, axis=1))
    iou_2 = model_2.predict(df_post_1.drop(droppables, axis=1))

    df_post_2['end_score'] = iou_1
    df_post_1['end_score'] = iou_2

    return pd.concat([df_post_2, df_post_1], ignore_index=True, axis=0)


@scatter_ex.automain
def scatter_main(setup_path, metrics_id, regression_params, random_seed=0):
    print(f"Experiment with {metrics_id}.")
    os.makedirs(f"{setup_path}/post_nms_regression", exist_ok=True)

    metrics_df = load_single_frame(f"{setup_path}/{metrics_id}_post_nms.csv")
    if (metrics_id != "gradients"):
        grad_ids = [s for s in list(
            metrics_df.columns) if "gradient_metrics" in s]
        metrics_df = metrics_df.drop(grad_ids, axis=1)
    iou_df = load_single_frame(f"{setup_path}/true_iou_post_nms.csv")

    fusion_df = meta_score(metrics_df, iou_df, regression_params, random_seed,
                           setup_path=setup_path).sort_values(by="dataset_box_id")

    fusion_df["r2"] = r2_score(iou_df["true_iou"], fusion_df["end_score"])

    fusion_df[["file_path", "dataset_box_id", "end_score", "r2"]].to_csv(
        f"{setup_path}/post_nms_regression/{metrics_id}_uncertainty.csv")
