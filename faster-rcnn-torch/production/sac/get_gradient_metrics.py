import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

from tqdm import tqdm
from sacred import Experiment

from src.access import load_frcnn_model, load_dataset
from src.model.roi_heads import fastrcnn_loss
from src.model.rpn import concat_box_prediction_layers

grads_ex = Experiment("Compute Gradients for Faster R-CNN")


@grads_ex.config
def grads_config():
    detector_settings = dict(num_classes=80,
                             weight_path="/home/riedlinger/OD/faster-rcnn-torch-lfs/weights/coco/retrain/ckpt_lr_1e-6e+00_ep_3.pt",
                             from_ckpt=True,
                             score_thresh=0.01
                             )

    data_settings = dict(img_dir="/home/datasets/COCO/2017/val2017",
                         target_dir="/home/riedlinger/OD/faster-rcnn-torch-lfs/uncertainty_metrics/coco")

    dir_id = "retrain2"


def map_grad_tensor_to_numbers(v):
    """
    Reduces gradient tensor to a set of uncertainty metrics.
    Returns:
        (dict [str -> float]): Gradient metrics generated from input tensor v.
    """
    d = {"1-norm": float(torch.norm(v, p=1).cpu().numpy()),
         "2-norm": float(torch.norm(v, p=2).cpu().numpy()),
         "min": float(v.min().cpu().numpy()),
         "max": float(v.max().cpu().numpy()),
         "mean": float(torch.mean(v).cpu().numpy()),
         "std": float(torch.std(v).cpu().numpy())}
    return d


@grads_ex.automain
def grads_main(detector_settings,
               data_settings,
               dir_id=None
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

    loss_contributions = ["rpn_obj", "rpn_reg", "roi_class", "roi_reg"]

    weight_dict = [{"rpn_conv": model.rpn.head.conv.weight,
                    "rpn_cls": model.rpn.head.cls_logits.weight},
                   {"rpn_conv": model.rpn.head.conv.weight,
                       "rpn_reg": model.rpn.head.bbox_pred.weight},
                   {"roi_fc_7": model.roi_heads.box_head.fc7.weight,
                       "roi_fc_cls": model.roi_heads.box_predictor.cls_score.weight},
                   {"roi_fc_7": model.roi_heads.box_head.fc7.weight,
                       "roi_fc_reg": model.roi_heads.box_predictor.bbox_pred.weight}]

    id_count = 0

    for batch in tqdm(dl):
        file_name = batch[0][0]

        imgs = [i.to(dev) for i in batch[1]]

        model.eval()
        model.rpn.nms_thresh = 0.6
        model.roi_heads.score_thresh = det_s["score_thresh"]
        model.roi_heads.nms_thresh = 0.95
        model.roi_heads.box_predictor.dropout = 0.0
        model.roi_heads.detections_per_img = 150

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
                                                            rpn_num_anchors_per_level,
                                                            )

        roi_features = model.roi_heads.box_roi_pool(features,
                                                    proposals,
                                                    images.image_sizes
                                                    )
        roi_features = model.roi_heads.box_head(roi_features)
        class_logits, box_regression = model.roi_heads.box_predictor(
            roi_features)

        result = []
        boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits,
                                                                       box_regression,
                                                                       proposals,
                                                                       images.image_sizes,
                                                                       )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        detections = model.transform.postprocess(result,
                                                 images.image_sizes,
                                                 original_image_sizes
                                                 )[0]
        gradient_list = []

        for det_id in tqdm(range(len(boxes[0]))):
            model.zero_grad()
            pgt = [{"boxes": detections["boxes"][det_id].unsqueeze(0),
                   "labels": detections["labels"][det_id].unsqueeze(0)}]
            # RPN loss:
            _, pgt = model.transform([imgs[0]], pgt)
            rpn_labels, rpn_matched_gt_boxes = model.rpn.assign_targets_to_anchors(rpn_anchors,
                                                                                   pgt,
                                                                                   cand_filter=True
                                                                                   )
            rpn_reg_targets = model.rpn.box_coder.encode(rpn_matched_gt_boxes,
                                                         rpn_anchors
                                                         )
            rpn_obj_loss, rpn_reg_loss = model.rpn.compute_loss(rpn_objectness,
                                                                rpn_pred_bbox_deltas,
                                                                rpn_labels,
                                                                rpn_reg_targets
                                                                )

            _, _, roi_labels, roi_reg_tgts, sampled = model.roi_heads.select_training_samples(proposals,
                                                                                              pgt,
                                                                                              True)
            cand_ids = sampled[0][:-1]
            logits = torch.index_select(class_logits, 0, cand_ids)
            reg = torch.index_select(box_regression, 0, cand_ids)
            roi_class_loss, roi_reg_loss = fastrcnn_loss(logits,
                                                         reg,
                                                         [roi_labels[0][:-1]],
                                                         [roi_reg_tgts[0][:-1]]
                                                         )

            instance_dict = {}
            for loss_id, t in enumerate([rpn_obj_loss, rpn_reg_loss, roi_class_loss, roi_reg_loss]):
                g = torch.autograd.grad(
                    t, list(weight_dict[loss_id].values()), grad_outputs=None, retain_graph=True)
                instance_dict[loss_contributions[loss_id]] = dict([(list(weight_dict[loss_id].keys())[ind],
                                                                    map_grad_tensor_to_numbers(v))
                                                                   for ind, v in enumerate(g)])
            gradient_list.append(instance_dict)

        coords = detections["boxes"]
        cpu_coords = coords.detach().cpu()
        score = detections["scores"].detach().cpu().numpy()
        num_entries = len(score)
        category_ids = (detections["labels"] - 1).cpu().numpy()

        # xmin, ymin, xmax, ymax = cpu_coords
        xmin = cpu_coords[..., 0]
        ymin = cpu_coords[..., 1]
        xmax = cpu_coords[..., 2]
        ymax = cpu_coords[..., 3]

        df = pd.DataFrame(columns=columns)
        df["file_path"] = [file_name for _ in range(num_entries)]
        df["xmin"] = xmin
        df["ymin"] = ymin
        df["xmax"] = xmax
        df["ymax"] = ymax
        df["s"] = score
        df["category_idx"] = category_ids
        current_index = list(df.index)
        df["dataset_box_id"] = range(id_count, id_count + len(current_index))
        df["gradient_metrics"] = gradient_list
        id_count += 150

        raw_name = file_name.split("/")[-1].split(".")[0]

        if dir_id is not None:
            csv_folder = f"{data_s['target_dir']}/{dir_id}/gradient_metrics/csv"
            os.makedirs(csv_folder, exist_ok=True)
            df.to_csv(f"{csv_folder}/{raw_name}_grads.csv")
        else:
            print("No directory identifier specified.")
