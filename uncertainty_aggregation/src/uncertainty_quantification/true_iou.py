import numpy as np
import pandas as pd
import tqdm

from src.bbox_tools.iou_metrics import bbox_iou
from configs.data_config import default_df_path


def contains_true_iou(dataframe):
    return "true_iou" in dataframe.columns


def extract_tp_labels(iou_df, iou_threshold):
    if "file_path" in iou_df.columns:
        iou_df = iou_df.drop("file_path", axis=1)
    labels = np.array(iou_df["true_iou"]) > iou_threshold
    iou_df["true_iou"] = labels
    iou_df.columns = ["dataset_box_id", "tp_label"]

    return iou_df


def get_true_iou(dataframe, gt_folder, keep_ids=True):
    target_path = f"{default_df_path}/true_iou.csv"
    if contains_true_iou(dataframe):
        print("True IoU column already present in dataframe!")
        df = dataframe[["dataset_box_id", "true_iou"]]
        df.to_csv(target_path)

        return df
    else:
        print("Computing true IoUs...")
        ious = []
        for row in tqdm.tqdm(dataframe.index):
            box_coords = np.copy(
                np.array(dataframe.loc[row, ["xmin", "ymin", "xmax", "ymax"]]))
            box_category = dataframe.loc[row, "category_idx"]

            img_idx = dataframe.loc[row, "file_path"].split(
                "/")[-1].split(".")[0]
            gt_df = pd.read_csv(
                f"{gt_folder}/{img_idx}_gt.csv").drop("Unnamed: 0", axis=1)
            gt_df = gt_df[gt_df["category_idx"].isin([box_category])]
            gt_coords = np.copy(
                np.array(gt_df[["xmin", "ymin", "xmax", "ymax"]]))

            if len(gt_coords) > 0:
                iou_arr = bbox_iou(box_coords, gt_coords)
                ious.append(np.amax(iou_arr))
            else:
                ious.append(0.)

        assert len(ious) == len(dataframe.index)

        if keep_ids:
            df = dataframe[["file_path", "dataset_box_id"]].copy(deep=True)
        else:
            df = dataframe[["file_path"]].copy(deep=True)
        df["true_iou"] = ious

        return df
