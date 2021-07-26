import os
import torch
import pandas as pd
import numpy as np
from sacred import Experiment

from prettytable import PrettyTable
from src.access import config_device, load_yolov3_model, load_dataset, save_checkpoint_weight_file
from src.evaluation import calculate_map

from production.get_forwards import ForwardConfig, GetForwards

eval_ex = Experiment("Evaluate YOLOv3")


@eval_ex.config
def eval_cfg():
    # Detector Settings
    detector_settings = dict(num_classes=80,
                             weight_path="/home/OD/yolov3-torch-lfs/weights/yolov3_original.pt",
                             from_ckpt=False,
                             # last_n_layers="tail",
                             # reset_weights=False,
                             # unfreeze_all=True,
                             # reset_head=False
                             )

    data_settings = dict(test_set="coco",
                         # train_dir="/home/datasets/COCO/2017/train2017",
                         # train_annot="/home/datasets/COCO/2017/annotations/instances_train2017.json",
                         # train_bs=32,
                         # n_cpu=8,
                         img_size=608,
                         # augmentation=True,
                         eval_dir="/home/datasets/COCO/2017/val2017",
                         score_thr=0.1,
                         gt_path="/home/dataset_ground_truth/COCO/2017/csv",
                         target_dir=None
                         )


@eval_ex.automain
def run_evaluation(detector_settings,
                   data_settings
                   ):
    det_s = detector_settings
    data_s = data_settings

    if data_s["target_dir"] is not None:
        os.makedirs(data_s["target_dir"], exist_ok=True)
    dev = torch.device("cuda:0")
    model = load_yolov3_model(det_s["weight_path"],
                              dev,
                              det_s["from_ckpt"],
                              mode="eval",
                              num_classes=det_s["num_classes"],
                              transfer=False)
    print(f"Loaded checkpoint from {det_s['weight_path']}.")

    fwd_cfg = ForwardConfig()
    predictions = pd.concat(GetForwards(fwd_cfg, ignore_config=True).run_forward(model=model,
                                                                                 img_dir=data_s["eval_dir"],
                                                                                 n_classes=det_s["num_classes"],
                                                                                 save=False),
                            axis=0)
    classes = open(f"../data/{data_s['test_set']}.names", "r").readlines()
    ap, f1 = calculate_map(predictions[predictions["s"] >= data_s["score_thr"]],
                           data_s["gt_path"],
                           num_classes=det_s["num_classes"],
                           print_results=False)
    mean_ap, mean_f1 = np.nanmean(ap), np.nanmean(f1)
    print(5*"#" + " Evaluation Table " + 5*"#")
    t = PrettyTable()
    t.add_column("Class", classes)
    t.add_column("AP", ap)
    t.add_column("F1", f1)
    print(t)
    print("mAP = {:.4} \t mF1 = {:.4}".format(mean_ap, mean_f1))
