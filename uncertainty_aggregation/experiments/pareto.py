import os

import pandas as pd
import numpy as np
import random
import xgboost as xgb

from sacred import Experiment
from sklearn.metrics import roc_auc_score, r2_score, average_precision_score

from src.api.loading_data import load_single_frame
from src.api.pre_process import standardize_columns

pareto_ex = Experiment("pareto_evaluation")

@pareto_ex.config
def pareto_config():
    setup_path = "/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain"
    # metrics_id = "mc_dropout_std"
    # metrics_id = "score_baseline"
    # metrics_id = "gradients"
    # metrics_id = "meta_detect"
    metrics_id = "all_norms"
    metrics_id = "two_norms"
    metrics_id = "md+grads"
    metrics_id = "grads+mc_dropout"
    metrics_id = "md+mc_dropout"
    metrics_id = "grads+md+mc_dropout"
    classifier_params = dict(max_depth=6,
                             n_estimators=30,
                             reg_alpha=1.5,
                             reg_lambda=0.0,
                             learning_rate=0.3)
    # classifier_params = "thresholding"

def meta_score(metrics_df, iou_df, classifier_params, random_seed=0):
    images = list(set(metrics_df["file_path"]))

    random.seed(random_seed)
    random.shuffle(images)

    split_1 = images[:int(len(images) / 2)]
    split_2 = images[int(len(images) / 2):]

    df_post_1 = metrics_df[metrics_df['file_path'].isin(split_1)]
    iou_post_1 = iou_df[iou_df['file_path'].isin(split_1)]
    df_post_2 = metrics_df[metrics_df['file_path'].isin(split_2)]
    iou_post_2 = iou_df[iou_df['file_path'].isin(split_2)]

    model_1 = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0, use_label_encoder=False, **classifier_params)
    model_2 = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0, use_label_encoder=False, **classifier_params)

    droppables = [s for s in list(df_post_1.columns) if "Unnamed" in s or "file_path" in s or "dataset_box_id" in s]
    print(df_post_1.size, iou_post_1.size)
    model_1.fit(df_post_1.drop(droppables, axis=1), iou_post_1['true_iou'].round(0))
    model_2.fit(df_post_2.drop(droppables, axis=1), iou_post_2['true_iou'].round(0))

    iou_1 = model_1.predict_proba(df_post_2.drop(droppables, axis=1))[:, 1]
    iou_2 = model_2.predict_proba(df_post_1.drop(droppables, axis=1))[:, 1]

    df_post_2['end_score'] = iou_1
    df_post_1['end_score'] = iou_2

    return pd.concat([df_post_2, df_post_1], ignore_index=True, axis=0)

def make_decisions(fusion_df, iou_df):
    eval_preds = fusion_df["end_score"]
    eval_targets = iou_df["true_iou"]
    print(len(eval_targets), len(eval_preds))

    auroc = roc_auc_score(eval_targets.round(0), fusion_df["end_score"])
    r2 = r2_score(eval_targets, fusion_df["end_score"])
    avg_prec = average_precision_score(eval_targets.round(0), fusion_df["end_score"])

    dec_threshs = np.arange(0, 1, 0.0005)
    err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp", "auroc", "avg_prec.", "r2"])
    err_df.loc[dec_threshs[0], "auroc"] = auroc
    err_df.loc[dec_threshs[0], "avg_prec."] = avg_prec
    err_df.loc[dec_threshs[0], "r2"] = r2

    for eps_dec in dec_threshs:
        pos = (eval_preds >= eps_dec)
        neg = (eval_preds < eps_dec)
        err_df.loc[eps_dec, "fp"] = np.sum(np.logical_and(pos, eval_targets.values < 0.5))
        err_df.loc[eps_dec, "fn"] = np.sum(np.logical_and(neg, eval_targets.values >= 0.5))

    return err_df

@pareto_ex.automain
def pareto_main(setup_path, metrics_id, classifier_params, random_seed=0):
    print(f"Experiment with {metrics_id}.")
    os.makedirs(f"{setup_path}/post_nms_evaluation", exist_ok=True)

    metrics_df = standardize_columns(load_single_frame(f"{setup_path}/{metrics_id}_post_nms.csv"))
    iou_df = load_single_frame(f"{setup_path}/true_iou_post_nms.csv")

    print(metrics_df, iou_df)

    if type(classifier_params) is str:
        fusion_df = metrics_df
        fusion_df.columns = ["file_path", "dataset_box_id", "end_score"]
        err_df = make_decisions(fusion_df, iou_df)
        err_df.to_csv(f"{setup_path}/post_nms_evaluation/{metrics_id}_thresholding_errs.csv")
    else:
        fusion_df = meta_score(metrics_df, iou_df, classifier_params, random_seed).sort_values(by="dataset_box_id")

        fusion_df[["file_path", "dataset_box_id", "end_score"]].to_csv(f"{setup_path}/post_nms_evaluation/{metrics_id}_uncertainty.csv")
        err_df = make_decisions(fusion_df, iou_df)
        err_df.to_csv(f"{setup_path}/post_nms_evaluation/{metrics_id}_errs.csv")
