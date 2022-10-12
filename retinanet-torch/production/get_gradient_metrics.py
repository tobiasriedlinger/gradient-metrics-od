import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

from tqdm import tqdm
from sacred import Experiment

from src.access import config_device, load_retinanet_model, load_dataset
import src.model.boxes as box_ops
from src.model.roi_heads import fastrcnn_loss
from src.utils import untransform_bboxes

grads_ex = Experiment("Compute Gradients for RetinaNet")


@grads_ex.config
def grads_config():
    detector_settings = dict(num_classes=8,
                             weight_path="/home/weights/kitti/retrain/ckpt_lr_1e-8e+00_ep_9_map_0.882_mf1_0.431.pt",
                             from_ckpt=True,
                             score_thresh=1e-4
                             )

    data_settings = dict(img_dir="/home/datasets/KITTI/test_split/images",
                         target_dir="/home/uncertainty_metrics/kitti")

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

    columns = ["dataset_box_id", "file_path", "xmin", "ymin", "xmax", "ymax", "s", "category_idx", "gradient_metrics"]

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

    loss_contributions = ["rpn_obj", "rpn_reg", "roi_class", "roi_reg"]

    weight_dict = [{"cls_logits": model.head.classification_head.cls_logits.weight,
                    "cls_conv": model.head.classification_head.conv._modules["6"].weight},
                   {"bbox_reg": model.head.regression_head.bbox_reg.weight,
                    "bbox_conv": model.head.regression_head.conv._modules["6"].weight}]


    id_count = 0

    for batch in tqdm(dl):
        file_name = batch[0][0]

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
        features = list(features.values())

        head_outputs = model.head(features)
        anchors = model.anchor_generator(images, features)

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

        detections = model.postprocess_detections(split_head_outputs,
                                                  split_anchors,
                                                  images.image_sizes,
                                                  )
        detections = model.transform.postprocess(detections,
                                                 images.image_sizes,
                                                 original_image_sizes)[0]

        # RetinaNet: compute_loss

        gradient_list = []

        for det_id in range(len(detections["boxes"])):
            model.zero_grad()
            pgt = [{"boxes": detections["boxes"][det_id].unsqueeze(0),
                   "labels": detections["labels"][det_id].unsqueeze(0)}]
            _, pgt = model.transform([imgs[0]], pgt)

            matched_idxs = []
            match_quality_matrix = box_ops.box_iou(pgt[0]["boxes"],
                                                   anchors[0])
            matched_idxs.append(model.proposal_matcher(match_quality_matrix))
            cls_loss = model.head.classification_head.compute_loss(pgt,
                                                                   head_outputs,
                                                                   matched_idxs,
                                                                   filter_cands=True
                                                                   )
            reg_loss = model.head.regression_head.compute_loss(pgt,
                                                               head_outputs,
                                                               anchors,
                                                               matched_idxs
                                                               )

            instance_dict = {}
            for loss_id, t in enumerate([cls_loss, reg_loss]):
                g = torch.autograd.grad(t, list(weight_dict[loss_id].values()), grad_outputs=None, retain_graph=True)
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
        id_count += num_det  # len(current_index)

        raw_name = file_name.split("/")[-1].split(".")[0]

        if dir_id is not None:
            csv_folder = f"{data_s['target_dir']}/{dir_id}/gradient_metrics/csv"
            os.makedirs(csv_folder, exist_ok=True)
            df.to_csv(f"{csv_folder}/{raw_name}_grads.csv")
        else:
            print("No directory identifier specified.")

