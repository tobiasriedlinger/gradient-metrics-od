import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

from tqdm import tqdm
from sacred import Experiment

from src.access import config_device, load_retinanet_model, load_dataset
import src.model.boxes as box_ops

mc_ex = Experiment("Compute MC dropout metrics for RetinaNet")


@mc_ex.config
def mc_config():
    detector_settings = dict(num_classes=8,
                             weight_path="/home/weights/kitti/retrain/ckpt_lr_1e-8e+00_ep_9_map_0.882_mf1_0.431.pt",
                             from_ckpt=True,
                             score_thresh=1e-4
                             )

    data_settings = dict(img_dir="/home/datasets/KITTI/test_split/images",
                         target_dir="/home/uncertainty_metrics/kitti")

    dir_id = "retrain2"
    num_runs = 30


@mc_ex.automain
def mc_main(detector_settings,
               data_settings,
               dir_id=None,
               num_runs=30
               ):
    det_s = detector_settings
    data_s = data_settings

    dl = load_dataset(name="image_folder",
                      img_dir=data_s["img_dir"],
                      annot_dir=None,
                      img_size=800,
                      batch_size=1,
                      n_cpu=4,
                      shuffle=False,
                      train=False
                      )

    dev = torch.device("cuda:0")

    model = load_retinanet_model(weight_path=det_s["weight_path"],
                                 device=dev,
                                 ckpt=det_s["from_ckpt"],
                                 num_classes=det_s["num_classes"]
                                 )

    num_det = 200
    model.topk_candidates = 200000
    model.detections_per_img = num_det
    model.score_thresh = det_s["score_thresh"]
    model.nms_thresh = 0.85
    model.head.regression_head.dropout = 0.0
    model.head.classification_head.dropout = 0.0


    id_count = 0

    for batch in tqdm(dl):
        file_name = batch[0]

        imgs = [i.to(dev) for i in batch[1]]

        model.eval()
        model.head.regression_head.dropout = 0.0
        model.head.classification_head.dropout = 0.0

        original_image_sizes = []
        for img in imgs:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, _ = model.transform(imgs)

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        features = list(features.values())

        xmins = np.zeros((num_det, num_runs))
        ymins = np.zeros((num_det, num_runs))
        xmaxs = np.zeros((num_det, num_runs))
        ymaxs = np.zeros((num_det, num_runs))
        s = np.zeros((num_det, num_runs))
        cats = np.zeros((num_det, num_runs))

        anchors = model.anchor_generator(images, features)
        head_outputs = model.head(features)

        # RetinaNet: computation of detection, lines 529 ff.
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        hw = 0
        for v in num_anchors_per_level:
            hw += v
        hwa = head_outputs["cls_logits"].size(1)
        a = hwa // hw
        num_anchors_per_level = [hw * a for hw in num_anchors_per_level]

        split_head_outputs = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        _, rankings, keeps = model.postprocess_detections(split_head_outputs,
                                                  split_anchors,
                                                  images.image_sizes,
                                                  nms=False
                                                  )

        for i in range(num_runs):
            model.head.regression_head.dropout = 0.5
            model.head.classification_head.dropout = 0.5
            head_outputs = model.head(features)

            # RetinaNet: computation of detection, lines 529 ff.
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            hw = 0
            for v in num_anchors_per_level:
                hw += v
            hwa = head_outputs["cls_logits"].size(1)
            a = hwa // hw
            num_anchors_per_level = [hw * a for hw in num_anchors_per_level]

            split_head_outputs = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))


            detections = model.postprocess_detections(split_head_outputs,
                                                      split_anchors,
                                                      images.image_sizes,
                                                      nms=False,
                                                      rankings=rankings,
                                                      keeps=keeps
                                                      )
            print(split_head_outputs["cls_logits"], len(detections))
            detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            n_det = min(len(detections[0]["boxes"]), len(detections[0]["scores"]))
            if i == 0:
                min_num_det = n_det
            else:
                if n_det < min_num_det:
                    min_num_det = n_det

            xmins[:n_det, i] = detections[0]["boxes"][:n_det, 0].detach().cpu().numpy()
            ymins[:n_det, i] = detections[0]["boxes"][:n_det, 1].detach().cpu().numpy()
            xmaxs[:n_det, i] = detections[0]["boxes"][:n_det, 2].detach().cpu().numpy()
            ymaxs[:n_det, i] = detections[0]["boxes"][:n_det, 3].detach().cpu().numpy()
            s[:n_det, i] = detections[0]["scores"].detach().cpu().numpy()
            cats[:n_det, i] = detections[0]["labels"].detach().cpu().numpy()

        raw_name = file_name[0].split("/")[-1].split(".")[0]
        df = pd.DataFrame()
        df["file_path"] = [file_name[0] for _ in range(num_det)]
        df["xmin_mean"] = np.mean(xmins[:min_num_det, :], axis=1)
        df["ymin_mean"] = np.mean(ymins[:min_num_det, :], axis=1)
        df["xmax_mean"] = np.mean(xmaxs[:min_num_det, :], axis=1)
        df["ymax_mean"] = np.mean(ymaxs[:min_num_det, :], axis=1)
        df["s_mean"] = np.mean(s[:min_num_det, :], axis=1)
        df["xmin_std"] = np.std(xmins[:min_num_det, :], axis=1)
        df["ymin_std"] = np.std(ymins[:min_num_det, :], axis=1)
        df["xmax_std"] = np.std(xmaxs[:min_num_det, :], axis=1)
        df["ymax_std"] = np.std(ymaxs[:min_num_det, :], axis=1)
        df["s_std"] = np.std(s[:min_num_det, :], axis=1)
        df["dataset_box_id"] = range(id_count, id_count + min_num_det)
        id_count += num_det

        if dir_id is not None:
            tgt_path = f"{data_s['target_dir']}/{dir_id}/mc_uncertainty/csv"
        else:
            tgt_path = f"{data_s['target_dir']}/mc_uncertainty/csv"
        os.makedirs(tgt_path, exist_ok=True)
        df.to_csv(f"{tgt_path}/{raw_name}_mc.csv")