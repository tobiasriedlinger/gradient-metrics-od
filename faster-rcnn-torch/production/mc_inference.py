import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

from tqdm import tqdm
from sacred import Experiment

from src.access import load_frcnn_model, load_dataset
from src.model.rpn import concat_box_prediction_layers

mc_ex = Experiment("Compute MC dropout metrics for Faster R-CNN")


@mc_ex.config
def mc_config():
    detector_settings = dict(num_classes=80,
                             weight_path="/home/riedlinger/OD/faster-rcnn-torch-lfs/weights/coco/retrain/ckpt_lr_1e-6e+00_ep_3.pt",
                             from_ckpt=True,
                             score_thresh=0.01
                             )

    data_settings = dict(img_dir="/home/datasets/COCO/2017/val2017",
                         target_dir="/home/riedlinger/OD/faster-rcnn-torch-lfs/uncertainty_metrics/coco")

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

    columns = ["dataset_box_id", "file_path", "xmin", "ymin",
               "xmax", "ymax", "s", "category_idx", "gradient_metrics"]

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

    model = load_frcnn_model(weight_path=det_s["weight_path"],
                             device=dev,
                             ckpt=det_s["from_ckpt"],
                             num_classes=det_s["num_classes"]
                             )

    model.rpn.nms_thresh = 0.6
    model.roi_heads.score_thresh = det_s["score_thresh"]
    model.roi_heads.nms_thresh = 0.95
    det_per_img = 150
    model.roi_heads.detections_per_img = det_per_img

    id_count = 0

    for batch in tqdm(dl):
        file_name = batch[0]

        imgs = [i.to(dev) for i in batch[1]]

        model.eval()

        original_image_sizes = []
        for img in imgs:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, _ = model.transform(imgs)

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        rpn_fts = list(features.values())
        rpn_objectness, rpn_pred_bbox_deltas = model.rpn.head(rpn_fts)
        rpn_anchors = model.rpn.anchor_generator(images, rpn_fts)

        rpn_num_images = len(rpn_anchors)
        rpn_num_anchors_per_level_shape_tensors = [
            o[0].shape for o in rpn_objectness]
        rpn_num_anchors_per_level = [s[0] * s[1] * s[2]
                                     for s in rpn_num_anchors_per_level_shape_tensors]
        rpn_objectness, rpn_pred_bbox_deltas = \
            concat_box_prediction_layers(rpn_objectness, rpn_pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        rpn_proposals = model.rpn.box_coder.decode(
            rpn_pred_bbox_deltas.detach(), rpn_anchors)
        rpn_proposals = rpn_proposals.view(rpn_num_images, -1, 4)
        proposals, prop_scores = model.rpn.filter_proposals(rpn_proposals,
                                                            rpn_objectness,
                                                            images.image_sizes,
                                                            rpn_num_anchors_per_level
                                                            )
        # end of rpn stage in forward of generalized_rcnn
        # roi stage now takes (features, rpn_proposals, images.image_sizes, "targets")

        roi_features = model.roi_heads.box_roi_pool(features,
                                                    proposals,
                                                    images.image_sizes
                                                    )
        roi_features = model.roi_heads.box_head(roi_features)

        xmins = np.zeros((det_per_img, num_runs))
        ymins = np.zeros((det_per_img, num_runs))
        xmaxs = np.zeros((det_per_img, num_runs))
        ymaxs = np.zeros((det_per_img, num_runs))
        s = np.zeros((det_per_img, num_runs))
        cats = np.zeros((det_per_img, num_runs))

        model.roi_heads.box_predictor.dropout = 0.0
        zclass_logits, zbox_regression = model.roi_heads.box_predictor(
            roi_features)
        min_num_det = 0

        for i in range(num_runs):
            model.roi_heads.box_predictor.dropout = 0.5
            class_logits, box_regression = model.roi_heads.box_predictor(
                roi_features)

            result = []
            boxes, _, labels = model.roi_heads.postprocess_detections(zclass_logits,
                                                                      box_regression,
                                                                      proposals,
                                                                      images.image_sizes,
                                                                      )
            _, scores, _ = model.roi_heads.postprocess_detections(class_logits,
                                                                  zbox_regression,
                                                                  proposals,
                                                                  images.image_sizes)

            num_images = len(boxes)
            for j in range(num_images):
                result.append(
                    {
                        "boxes": boxes[j],
                        "labels": labels[j],
                        "scores": scores[j],
                    }
                )

            detections = model.transform.postprocess(result,
                                                     images.image_sizes,
                                                     original_image_sizes
                                                     )

            n_det = min(len(detections[0]["boxes"]),
                        len(detections[0]["scores"]))
            if i == 0:
                min_num_det = n_det
            else:
                if n_det < min_num_det:
                    min_num_det = n_det
            xmins[:n_det, i] = detections[0]["boxes"][:n_det,
                                                      0].detach().cpu().numpy()
            ymins[:n_det, i] = detections[0]["boxes"][:n_det,
                                                      1].detach().cpu().numpy()
            xmaxs[:n_det, i] = detections[0]["boxes"][:n_det,
                                                      2].detach().cpu().numpy()
            ymaxs[:n_det, i] = detections[0]["boxes"][:n_det,
                                                      3].detach().cpu().numpy()
            s[:n_det, i] = detections[0]["scores"][:n_det].detach().cpu().numpy()
            cats[:n_det, i] = detections[0]["labels"][:n_det].detach().cpu().numpy()

        raw_name = file_name[0].split("/")[-1].split(".")[0]
        df = pd.DataFrame()
        df["file_path"] = [file_name[0] for _ in range(min_num_det)]
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
        id_count += det_per_img

        if dir_id is not None:
            tgt_path = f"{data_s['target_dir']}/{dir_id}/mc_uncertainty/csv"
        else:
            tgt_path = f"{data_s['target_dir']}/mc_uncertainty/csv"
        os.makedirs(tgt_path, exist_ok=True)
        df.to_csv(f"{tgt_path}/{raw_name}_mc.csv")
