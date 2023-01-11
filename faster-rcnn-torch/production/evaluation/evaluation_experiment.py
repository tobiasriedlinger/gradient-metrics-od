import os
import torch
import pandas as pd
import numpy as np
from sacred import Experiment

from torch.utils.data.dataloader import DataLoader
import torchvision as tv
from src.datasets.transforms import collate_fn

from prettytable import PrettyTable
from src.access import load_frcnn_model, load_dataset
from src.evaluation.evaluation import calculate_map

from production.get_forwards import ForwardConfig, GetForwards

eval_ex = Experiment("Evaluate Faster R-CNN")


@eval_ex.config
def eval_cfg():
    # Detector Settings
    detector_settings = dict(num_classes=80,
                             weight_path="/home/riedlinger/OD/faster-rcnn-torch-lfs/weights/coco/retrain/ckpt_lr_1e-6e+00_ep_2",
                             from_ckpt=True,
                             )

    data_settings = dict(test_set="coco",
                         name="coco",
                         val_bs=1,
                         img_size=800,
                         eval_dir="/home/datasets/COCO/2017/val2017",
                         score_thr=0.01,
                         gt_path="/home/riedlinger/dataset_ground_truth/COCO/2017/csv",
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
    model = load_frcnn_model(det_s["weight_path"],
                             dev,
                             det_s["from_ckpt"],
                             mode="eval",
                             num_classes=det_s["num_classes"],
                             box_score_thr=data_s["score_thr"],
                             transfer=False)
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                        progress=True,
                                                        num_classes=91
                                                        )
    model.to(dev)
    print(f"Loaded checkpoint from {det_s['weight_path']}.")

    model.roi_heads.score_thresh = 0.00001
    model.roi_heads.box_predictor.dropout = 0.0
    model.roi_heads.detections_per_img = 150

    # Check evaluation pipeline
    train_ds, val_ds = load_dataset(name=data_s["name"],
                                    )
    val_dl = DataLoader(val_ds,
                        batch_size=data_s["val_bs"],
                        collate_fn=collate_fn,
                        )

    fwd_cfg = ForwardConfig()
    predictions = pd.concat(GetForwards(fwd_cfg, ignore_config=True).run_forward(model=model,
                                                                                 img_dir=data_s["eval_dir"],
                                                                                 n_classes=det_s["num_classes"],
                                                                                 save=False),
                            axis=0)
    classes = open(f"../../data/{data_s['test_set']}.names", "r").readlines()
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
