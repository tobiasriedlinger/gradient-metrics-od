import json
import os

import pandas as pd
import glob
import time

import tqdm

import configs.data_config as data_cfg
import configs.models_config as model_cfg
import configs.uq_schedule_dict as uq_conf
from uncertainty_quantification.true_iou import get_true_iou, extract_tp_labels
from meta_detect_metrics.md_metrics import output_based_metrics


def load_frames_from_folder(path):
    try:
        frames = [pd.read_csv(file).drop("Unnamed: 0", axis=1)
                  for file in glob.glob(f"{path}/*")]
    except:
        frames = [pd.read_csv(file) for file in glob.glob(f"{path}/*")]

    return pd.concat(frames, axis=0)


def load_single_frame(path, verbose=False):
    time_0 = time.time()
    if verbose:
        print("Loading dataframe...")
    try:
        df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
    except:
        df = pd.read_csv(path)
    if verbose:
        print("Done. ({:.3}s)".format(time.time() - time_0))

    return df


def load_or_prepare_metrics(path=data_cfg.default_df_path,
                            model="gb_classifier",
                            metrics_const=["score"]):
    """
    Utilizes information from config file src/uncertainty_quantification/uq_config.py
    to set up dataframes containing respective variables and targets.
    :return df (pandas DataFrame): Dataframe [dataset_box_id, s, ...] with uncertainty metrics; contains
                                        at least confidence score.
    :return target_df (pandas DataFrame): Dataframe [dataset_box_id, true_iou] with meta classification
                                        (binary) / meta regression (float) targets.
    """
    df, iou_df, grads_df, meta_detect_df = combine_data_frames(
        path, metrics_const)
    # todo: this is still a little ugly; what if we want to consider additional baselines?
    if "output" in metrics_const:
        baseline_identifiers = data_cfg.OUTPUT_METRICS
    else:
        baseline_identifiers = ["s"]
    df = df[["dataset_box_id", "file_path"] + baseline_identifiers]

    if "gradient_metrics" in metrics_const:
        df = pd.concat([df, meta_detect_df, grads_df.drop(
            ["dataset_box_id", "file_path"], axis=1)], axis=1)
    target_df = choose_training_targets(model, iou_df)

    return df, target_df


def combine_data_frames(folder, metrics_const=["output", "gradient_metrics", "meta_detect"]):
    """
    Loads uncertainty metric frame (metrics_cat.csv), gradient metrics frame (gradient_metrics.csv) and
    IoU frame (true_iou.csv) from passed folder if present. Otherwise will compute and sort the data from
    subfolder "csv" where image-wise .csv-files are located and save combined frames as
        - metrics_cat.csv
        - gradient_metrics.csv
        - true_iou.csv ,
    respectively.
    :param folder: (str), folder with subfolder "csv" or the required .csv-files present to load data from.
    :param metrics_const: (list[str]) Identifiers for uncertainty metrics classes to use as defined in src.uncertainty_quantification.uq_config.
    :return cat: (pandas DataFrame) Concatenated DataFrame with basic uncertainty metrics.
                Contains at least [file_path, xmin, ymin, xmax, ymax, s, category_idx, prob_sum, prob_i,
                                    dataset_box_id]
    :return iou_df: (pandas DataFrame) DataFrame [file_path, dataset_box_id, true_iou]
    :return grads_df: (pandas DataFrame) DataFrame containing parsed gradient metrics.
                Contains at least [file_path, dataset_box_id]
    """
    folder = os.path.normpath(folder)
    cat_path = f"{folder}/metrics_cat.csv"
    # todo: dynamic score threshold
    meta_detect_path = f"{folder}/md_metrics_0.0.csv"
    grads_path = f"{folder}/gradient_metrics.csv"
    iou_path = f"{folder}/true_iou.csv"
    if glob.glob(cat_path):
        print("Loading combined DataFrame...")
        cat = load_single_frame(cat_path)
        print("Done.")
    else:
        file_list = glob.glob(f"{folder}/csv/*.csv")

        print("Loading DataFrames...")
        df_list = [load_single_frame(p) for p in file_list]

        print("Accumulating DataFrames...")
        cat = pd.concat(df_list, axis=0, ignore_index=True)
        cat.to_csv(cat_path)
        print("Done.")

    if glob.glob(meta_detect_path):
        print("Loading MetaDetect metrics DataFrame...")
        md_metrics = load_single_frame(meta_detect_path)
        print("Done.")
    else:
        print("Computing MetaDetect metrics...")
        md_metrics = output_based_metrics(cat, 0.0)

    if "gradient_metrics" in metrics_const:
        if glob.glob(grads_path):
            print("Loading gradient DataFrame...")
            grads_df = load_single_frame(grads_path)
            print("Done.")
        else:
            grads_df = cat["gradient_metrics"].copy(deep=True)
            grads_df = parse_gradient_metrics(grads_df, cat)
            grads_df.to_csv(grads_path)

    else:
        grads_df = cat[["file_path", "dataset_box_id"]].copy(deep=True)

    if "gradient_metrics" in cat.columns:
        cat.drop("gradient_metrics", axis=1)

    if "true_iou" in cat.columns:
        iou_df = cat[["file_path", "true_iou",
                      "dataset_box_id"]].copy(deep=True)
        cat = cat.drop("true_iou", axis=1)
        iou_df.to_csv(iou_path)
    elif glob.glob(iou_path):
        print("Loading iou DataFrame...")
        iou_df = load_single_frame(iou_path)
        print("Done.")
    else:
        iou_df = get_true_iou(cat, data_cfg.default_gt_path)
        iou_df.to_csv(iou_path)

    return cat, iou_df, grads_df, md_metrics


def parse_gradient_metrics(grads_df, cat_df):
    parsed_df = cat_df[["file_path", "dataset_box_id"]].copy(deep=True)
    print("Collecting gradient metrics...")
    for i, s in tqdm.tqdm(enumerate(grads_df)):
        s = s.replace("'", '"')
        grad_dict = json.loads(s)
        for loss in grad_dict:
            for layer in grad_dict[loss]:
                for metric in grad_dict[loss][layer]:
                    parsed_df.loc[i,
                                  f"grad_{loss}_{layer}_{metric}"] = grad_dict[loss][layer][metric]

    return parsed_df


def load_metrics_and_targets(path=data_cfg.default_df_path, model=uq_conf.d["model"], metrics_const=uq_conf.d["metrics_const"], nms=True):
    nms_dict = {True: "post_nms", False: "pre_nms"}
    metrics_df = pd.read_csv(f"{path}/metrics_{nms_dict[nms]}.csv")
    iou_df = pd.read_csv(f"{path}/true_iou_{nms_dict[nms]}.csv")
    targets_df = choose_training_targets(model, iou_df)


def choose_training_targets(aggr_model, iou_df):
    if aggr_model in model_cfg.CLASSIFICATION_MODELS:
        return extract_tp_labels(iou_df, 0.45)
    elif aggr_model in model_cfg.REGRESSION_MODELS:
        return iou_df.drop("file_path", axis=1)
    else:
        raise ValueError(
            f"Given uncertainty aggregation model '{aggr_model}' is not implemented!\nPossible classifiers: {model_cfg.CLASSIFICATION_MODELS}\nPossible regression models: {model_cfg.REGRESSION_MODELS}")


def get_gt(gt_path):
    for p in glob.glob(str(gt_path)+"/*.csv"):
        gt_df = load_single_frame(p, verbose=False)
        try:
            gt = gt.append(gt_df, ignore_index=True)
        except:
            gt = gt_df

    return gt
