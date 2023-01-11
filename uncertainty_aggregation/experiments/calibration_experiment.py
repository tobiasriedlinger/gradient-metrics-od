import os
import pandas as pd
import numpy as np
from sacred import Experiment
from src.api.loading_data import load_single_frame

calib_ex = Experiment("Calibration check")

@calib_ex.config
def calib_config():
    setup_path = "/home/riedlinger/OD/faster-rcnn-torch-lfs/uncertainty_metrics/coco/retrain/"
    metrics_id = "score_baseline"

@calib_ex.automain
def calib_main(setup_path,
               metrics_id):
    metrics_df = load_single_frame(os.path.join(setup_path, "metrics_post_nms.csv"))
    iou_df = load_single_frame(os.path.join(setup_path, "true_iou_post_nms.csv"))
    unc_df = load_single_frame(os.path.join(setup_path, f"post_nms_classification_cv/{metrics_id}_uncertainty.csv"))

    calib_df = pd.DataFrame(columns=[f"hist_{i}" for i in range(10)]+["ece_mean", "ece_std", "mce_mean", "mce_std", "ace_mean", "ace_std"])

    threshs = np.arange(0.0, 1.0, 0.1)

    ece, mce, ace = [], [], []

    for r in range(10):
        accs, mean_conf = [], []
        bin_sizes = []
        for t in threshs:
            if metrics_id == "score_baseline":
                bin = metrics_df[metrics_df["s"].between(t, t+0.1, inclusive=False)]
                conf = bin["s"]
            else:
                bin = unc_df[unc_df[f"end_score_{r}"].between(t, t+0.1, inclusive=False)]
                conf = bin[f"end_score_{r}"]
            ious = iou_df[iou_df["dataset_box_id"].isin(bin["dataset_box_id"])]
            true_samples = len(ious[ious["true_iou"] > 0.5])
            acc = 1.0 * true_samples / max(len(bin), 1)
            bin_sizes.append(len(bin))
            accs.append(acc)
            mean_conf.append(np.mean(conf))
            calib_df.loc[t, f"hist_{r}"] = acc

        accs = np.array(accs)
        mean_conf = np.array(mean_conf)
        bin_sizes = np.array(bin_sizes)

        ece.append(np.sum(bin_sizes / len(unc_df) * np.abs(accs - mean_conf)))
        mce.append(np.max(np.abs(accs - mean_conf)))
        ace.append(np.mean(np.abs(accs - mean_conf)))

    calib_df.loc[threshs[0], "ece_mean"] = np.mean(ece)
    calib_df.loc[threshs[0], "ece_std"] = np.std(ece)
    calib_df.loc[threshs[0], "mce_mean"] = np.mean(mce)
    calib_df.loc[threshs[0], "mce_std"] = np.std(mce)
    calib_df.loc[threshs[0], "ace_mean"] = np.mean(ace)
    calib_df.loc[threshs[0], "ace_std"] = np.std(ace)

    calib_df.to_csv(os.path.join(setup_path, f"post_nms_classification_cv/{metrics_id}_calibration.csv"))
