import glob
import numpy as np
import pandas as pd
from sacred import Experiment
from tqdm import tqdm

from src.api.loading_data import load_single_frame, parse_gradient_metrics
from src.meta_detect_metrics.md_metrics import output_based_metrics
from src.uncertainty_quantification.true_iou import get_true_iou
from src.bbox_tools.nms_algorithms import perform_nms_on_dataframe, select_rows_from_other

ex = Experiment("metrics_setup")


@ex.config
def setup_config():
    setup_dir = "/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/deep_grads"
    gt_folder = "/home/riedlinger/dataset_ground_truth/KITTI/csv"


def setup_gradients(setup_dir, gt_folder):
    csv_list = glob.glob(f"{setup_dir}/gradient_metrics/csv/*.csv")
    print("Loading and accumulating single image frames...")
    df_list = [load_single_frame(p) for p in csv_list]

    metrics_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Saved accumulation.")

    if "gradient_metrics" in metrics_df.columns:
        grads_df = load_single_frame(f"{setup_dir}/gradients_pre_nms.csv")
        metrics_df = metrics_df.drop("gradient_metrics", axis=1)
    else:
        grads_df = pd.DataFrame()

    metrics_df.to_csv(f"{setup_dir}/metrics_pre_nms.csv")
    grads_df.to_csv(f"{setup_dir}/gradients_pre_nms.csv")

    # Meta Detect metrics
    md_df = output_based_metrics(metrics_df, 0.0)
    md_df = pd.concat([metrics_df, md_df], axis=1)
    md_df.to_csv(f"{setup_dir}/meta_detect_pre_nms.csv")

    score_df = metrics_df[["file_path", "dataset_box_id", "s"]].copy(deep=True)
    score_df.to_csv(f"{setup_dir}/score_baseline_pre_nms.csv")

    # True IoU
    iou_df = load_single_frame(f"{setup_dir}/true_iou_post_nms.csv")

    # Perform NMS on metrics_df to obtain ids for survivors
    pnms_df = perform_nms_on_dataframe(
        metrics_df[["file_path", "xmin", "ymin", "xmax", "ymax", "s", "category_idx", "dataset_box_id"]])
    # Save post-NMS metrics
    pnms_metrics_df = select_rows_from_other(metrics_df, pnms_df)
    pnms_metrics_df.to_csv(f"{setup_dir}/metrics_post_nms.csv")
    pnms_grads_df = select_rows_from_other(grads_df, pnms_df)
    pnms_grads_df.to_csv(f"{setup_dir}/gradients_post_nms.csv")
    pnms_md_df = select_rows_from_other(md_df, pnms_df)
    pnms_md_df.to_csv(f"{setup_dir}/meta_detect_post_nms.csv")
    pnms_score_df = select_rows_from_other(score_df, pnms_df)
    pnms_score_df.to_csv(f"{setup_dir}/score_baseline_post_nms.csv")
    pnms_iou_df = select_rows_from_other(iou_df, pnms_df)
    pnms_iou_df.to_csv(f"{setup_dir}/true_iou_post_nms.csv")

    # Classification-based metrics
    prob_cols = [p for p in metrics_df.columns if (
        "prob_" in p) and (not "sum" in p)]
    if prob_cols:
        probs = np.array(metrics_df[prob_cols])
        entropy = - 1.0 * np.sum(probs * np.log(probs), axis=1)
        logits = np.log(np.clip(probs, 1e-8, 1e8) / (1. - probs))
        energy = - 100.0 * \
            np.log(np.clip(np.sum(np.exp(logits / 100.), axis=1), 1e-8, 1e8))
        metrics_df[["file_path", "dataset_box_id"] +
                   prob_cols].to_csv((f"{setup_dir}/softmax_pre_nms.csv"))
        pnms_metrics_df[["file_path", "dataset_box_id"] +
                        prob_cols].to_csv((f"{setup_dir}/softmax_post_nms.csv"))

        entropy_df = metrics_df.copy(
            deep=True)[["file_path", "dataset_box_id"]]
        entropy_df["entropy"] = entropy
        entropy_df.to_csv((f"{setup_dir}/entropy_pre_nms.csv"))
        pnms_entropy_df = select_rows_from_other(entropy_df, pnms_df)
        pnms_entropy_df.to_csv((f"{setup_dir}/entropy_post_nms.csv"))

        energy_df = metrics_df.copy(deep=True)[["file_path", "dataset_box_id"]]
        energy_df["energy"] = energy
        energy_df.to_csv((f"{setup_dir}/energy_pre_nms.csv"))
        pnms_energy_df = select_rows_from_other(energy_df, pnms_df)
        pnms_energy_df.to_csv((f"{setup_dir}/energy_post_nms.csv"))

    # Combination of gradient metrics and Output metrics
    pd.concat([metrics_df, grads_df.drop(["file_path", "dataset_box_id"],
              axis=1)], axis=1).to_csv(f"{setup_dir}/output+grads_pre_nms.csv")
    pd.concat([pnms_metrics_df, pnms_grads_df.drop(["file_path", "dataset_box_id"],
              axis=1)], axis=1).to_csv(f"{setup_dir}/output+grads_post_nms.csv")

    # Combination of gradient metrics and Meta Detect metrics
    pd.concat([md_df, grads_df.drop(["file_path", "dataset_box_id"],
              axis=1)], axis=1).to_csv(f"{setup_dir}/md+grads_pre_nms.csv")
    pd.concat([pnms_md_df, pnms_grads_df.drop(["file_path", "dataset_box_id"],
              axis=1)], axis=1).to_csv(f"{setup_dir}/md+grads_post_nms.csv")

    # 2-norm gradient metrics
    grad_columns = list(grads_df.columns)
    two_norm_ids = [s for s in grad_columns if "2-norm" in s]
    grads_df[["file_path", "dataset_box_id"] +
             two_norm_ids].to_csv(f"{setup_dir}/two_norms_pre_nms.csv")
    pnms_grads_df[["file_path", "dataset_box_id"] +
                  two_norm_ids].to_csv(f"{setup_dir}/two_norms_post_nms.csv")

    # All norm gradient metrics
    all_norm_ids = [s for s in grad_columns if "norm" in s]
    grads_df[["file_path", "dataset_box_id"] +
             all_norm_ids].to_csv(f"{setup_dir}/all_norms_pre_nms.csv")
    pnms_grads_df[["file_path", "dataset_box_id"] +
                  all_norm_ids].to_csv(f"{setup_dir}/all_norms_post_nms.csv")


def setup_var_inf(setup_dir, id):
    assert id in ["mcdropout", "ensemble"]

    metrics_df = load_single_frame(f"{setup_dir}/metrics_post_nms.csv")
    csv_list = glob.glob(f"{setup_dir}/{id}/csv/*.csv")

    ls = []
    print(f"Loading {id} frames...")
    for p in tqdm(csv_list):
        ls.append(select_rows_from_other(load_single_frame(p), metrics_df))
    cat = pd.concat(ls, axis=0, ignore_index=True)

    std_cols = [s for s in list(cat.columns) if "std" in s]
    pd.concat([metrics_df, cat[std_cols]], axis=1).to_csv(
        f"{setup_dir}/{id}_all_post_nms.csv")
    pd.concat([metrics_df[["file_path", "dataset_box_id"]], cat[std_cols]],
              axis=1).to_csv(f"{setup_dir}/{id}_std_post_nms.csv")


def combine_all_data(setup_dir):
    mc_df = load_single_frame(f"{setup_dir}/mcdropout_std_post_nms.csv")
    mc_all_df = load_single_frame(f"{setup_dir}/mcdropout_all_post_nms.csv")
    ens_df = load_single_frame(f"{setup_dir}/ensemble_std_post_nms.csv")
    ens_all_df = load_single_frame(f"{setup_dir}/ensemble_all_post_nms.csv")

    md_df = load_single_frame(f"{setup_dir}/meta_detect_post_nms.csv")
    grad_df = load_single_frame(f"{setup_dir}/gradients_post_nms.csv")
    start_ids = ["file_path", "dataset_box_id"]
    mc_ids = list(mc_df.columns)[2:]
    mc_all_ids = list(mc_all_df.columns)[2:]
    ens_ids = list(ens_df.columns)[2:]
    ens_all_ids = list(ens_all_df.columns)[2:]
    md_ids = list(md_df.columns)[2:]
    grad_ids = list(grad_df.columns)[2:]

    # MC + ENS
    print("MC + Ensembles")
    pd.concat([mc_df, ens_df[ens_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+ens_std_post_nms.csv")
    pd.concat([mc_df, ens_all_df[ens_all_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+ens_all_post_nms.csv")
    pd.concat([mc_all_df, ens_df[ens_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+ens_std_post_nms.csv")
    pd.concat([mc_all_df, ens_all_df[ens_all_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+ens_all_post_nms.csv")

    # MC + MD
    print("MC + MetaDetect")
    pd.concat([mc_df, md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+md_post_nms.csv")
    pd.concat([mc_all_df, md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+md_post_nms.csv")

    # MC + GRADS
    print("MC + Gradients")
    pd.concat([mc_df, grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+grad_post_nms.csv")
    pd.concat([mc_all_df, grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+grad_post_nms.csv")

    # ENS + MD
    print("Ensembles + MetaDetect")
    pd.concat([ens_df, md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/ens_std+md_post_nms.csv")
    pd.concat([ens_all_df, md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/ens_all+md_post_nms.csv")

    # ENS + GRADS
    print("Ensembles + Gradients")
    pd.concat([ens_df, grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/ens_std+grad_post_nms.csv")
    pd.concat([ens_all_df, grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/ens_all+grad_post_nms.csv")

    # MD + GRADS
    print("MetaDetect + Gradients")
    pd.concat([md_df, grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/md+grad_post_nms.csv")

    # MC + ENS + MD
    print("MC + Ensembles + MetaDetect")
    pd.concat([mc_df, ens_df[ens_ids], md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+ens_std+md_post_nms.csv")
    pd.concat([mc_df, ens_all_df[ens_all_ids], md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+ens_all+md_post_nms.csv")
    pd.concat([mc_all_df, ens_df[ens_ids], md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+ens_std+md_post_nms.csv")
    pd.concat([mc_all_df, ens_all_df[ens_all_ids], md_df[md_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+ens_all+md_post_nms.csv")

    # MC + ENS + GRADS
    print("MC + Ensembles + Gradients")
    pd.concat([mc_df, ens_df[ens_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+ens_std+grad_post_nms.csv")
    pd.concat([mc_df, ens_all_df[ens_all_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+ens_all+grad_post_nms.csv")
    pd.concat([mc_all_df, ens_df[ens_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+ens_std+grad_post_nms.csv")
    pd.concat([mc_all_df, ens_all_df[ens_all_ids], grad_df[grad_ids]],
              axis=1).to_csv(f"{setup_dir}/mc_all+ens_all+grad_post_nms.csv")

    # MC + MD + GRADS
    print("MC + MetaDetect + Gradients")
    pd.concat([mc_df, md_df[md_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_std+md+grad_post_nms.csv")
    pd.concat([mc_all_df, md_df[md_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/mc_all+md+grad_post_nms.csv")

    # ENS + MD + GRADS
    print("Ensembles + MetaDetect + Gradients")
    pd.concat([ens_df, md_df[md_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/ens_std+md+grad_post_nms.csv")
    pd.concat([ens_all_df, md_df[md_ids], grad_df[grad_ids]], axis=1).to_csv(
        f"{setup_dir}/ens_all+md+grad_post_nms.csv")

    # MC + ENS + MD + GRADS
    print("MC + Ensembles + MetaDetect + Gradients")
    pd.concat([mc_df, ens_df[ens_ids], md_df[md_ids], grad_df[grad_ids]],
              axis=1).to_csv(f"{setup_dir}/mc_std+ens_std+md+grad_post_nms.csv")
    pd.concat([mc_df, ens_all_df[ens_all_ids], md_df[md_ids], grad_df[grad_ids]],
              axis=1).to_csv(f"{setup_dir}/mc_std+ens_all+md+grad_post_nms.csv")
    pd.concat([mc_all_df, ens_df[ens_ids], md_df[md_ids], grad_df[grad_ids]],
              axis=1).to_csv(f"{setup_dir}/mc_all+ens_std+md+grad_post_nms.csv")
    pd.concat([mc_all_df, ens_all_df[ens_all_ids], md_df[md_ids], grad_df[grad_ids]],
              axis=1).to_csv(f"{setup_dir}/mc_all+ens_all+md+grad_post_nms.csv")


@ex.automain
def main_setup(setup_dir, gt_folder):
    setup_gradients(setup_dir, gt_folder)
