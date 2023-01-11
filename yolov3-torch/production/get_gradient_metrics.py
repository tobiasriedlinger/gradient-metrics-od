"""
Implements a pipeline for computing gradient uncertainty metrics.
"""
import os
import torch
import numpy as np
import pandas as pd

from torch.nn.functional import binary_cross_entropy_with_logits as bce, mse_loss as mse
from tqdm import tqdm
from production.get_forwards import cxcywh_to_border
from src.main import load_yolov3_model, load_dataset
from src.utils import untransform_bboxes
from src.inference import iou
from src.yolov3_loss import get_targets


class GradientsConfig(object):
    """
    Container class for gradient uncertainty metric configuration.
    """
    weight_path = "/home/OD/yolov3-torch-lfs/weights/coco/retrain/" \
        + "ckpt_lr_1e-5e+00_ep_2_map_0.546_mf1_0.291.pt"
    img_size = 608
    img_dir = "/home/datasets/COCO/2017/val2017"

    batch_size = 1
    dev = torch.device("cuda:0")

    target_folder = "/home/OD/yolov3-torch-lfs/uncertainty_metrics/coco/"
    score_thresh = 1e-2
    iou_thresh = 0.5
    num_classes = 80
    num_attrib = num_classes + 5
    last_layer_dim = 3 * num_attrib
    dropout_rate = 0.0

    loss_contributions = ["loc", "score", "prob"]
    probabilities = [f"prob_{i}" for i in range(num_classes)]
    columns = ["dataset_box_id", "file_path", "xmin", "ymin", "xmax", "ymax", "s", "category_idx", "prob_sum"]\
        + probabilities + ["gradient_metrics"]


class GetGradientMetrics(object):
    """
    Container class for gradient metric computation pipeline.
    """

    def __init__(self, config):
        self.cfg = config
        # Load YOLO model in train mode, such that we get the prediction transformed to image dimensions,
        # as well as the raw prediction used in the loss function.
        self.model = load_yolov3_model(weight_path=self.cfg.weight_path,
                                       device=self.cfg.dev,
                                       cfg=config,
                                       ckpt=True,
                                       mode="train"
                                       )
        for layer in self.model.yolo_last_n_layers("tail"):
            for p in layer.parameters():
                p.requires_grad_()

        self.weight_dict = {"pred_s": self.model.yolo_tail.detect1.conv7.weight,
                            "pred_m": self.model.yolo_tail.detect2.conv7.weight,
                            "pred_l": self.model.yolo_tail.detect3.conv7.weight,
                            "bridge_s": self.model.yolo_tail.detect1.conv6.conv.weight,
                            "bridge_m": self.model.yolo_tail.detect2.conv6.conv.weight,
                            "bridge_l": self.model.yolo_tail.detect3.conv6.conv.weight}

        self.dataloader = load_dataset(name="image_folder",
                                       img_dir=self.cfg.img_dir,
                                       annot_dir=None,
                                       img_size=self.cfg.img_size,
                                       batch_size=self.cfg.batch_size,
                                       n_cpu=4,
                                       shuffle=False,
                                       augment=False)

    def run_gradients(self, dir_id=None):
        dev = self.cfg.dev

        print(f"Total images in dataset: {len(self.dataloader)}")

        id_count = 0

        for i, item in enumerate(self.dataloader):
            file_name = item[0][0]
            print(f"Image number {i}: {file_name}")

            img = item[1].to(dev)
            scale = item[2].to(dev)
            padding = item[3].to(dev)
            detection = self.model(img)

            raw_pred = torch.cat([detection[i][0].view(
                1, -1, self.cfg.num_attrib) for i in range(3)], dim=1)
            output = torch.cat([detection[i][1] for i in range(3)], dim=1)

            # Transformed prediction w.r.t. original image dimensions:
            coords = untransform_bboxes(
                output[..., :4].clone(), scale[0], padding[0]).detach()

            # Remember that we need as ground truth boxed w.r.t. network input dimensions, though.
            total_num_predictions = coords.shape[1]

            soft_thr_mask = (output[:, :, 4] >= self.cfg.score_thresh)
            soft_thr_coords = coords[soft_thr_mask, :]
            soft_thr_det = output[soft_thr_mask, :].detach()
            num_detections = soft_thr_det.shape[0]

            gradient_list = []

            for instance_id in tqdm(range(num_detections)):
                pgt = soft_thr_det[instance_id, ...].view(1, 1, -1).clone()
                pgt[..., 4] = 1.0
                pgt_class = torch.argmax(pgt[..., 5:], dim=2).squeeze(1)
                pgt[..., 5:] = torch.nn.functional.one_hot(
                    pgt_class, num_classes=self.cfg.num_classes)
                # Get candidates for pgt
                candidate_ids = (iou(output, pgt).squeeze(2) >= self.cfg.iou_thresh) & \
                                (torch.argmax(
                                    output[..., 5:], dim=2) == pgt_class.squeeze().item())

                pgt_t, pos_mask, neg_mask = get_targets(
                    pgt, torch.tensor([[1]], dtype=torch.long), self.cfg.img_size)

                loc_bce = torch.sum(
                    bce(raw_pred[:, :, :2], pgt_t[:, :, :2], reduction="none"), dim=2)
                loc_mse = torch.sum(
                    mse(raw_pred[:, :, 2:4], pgt_t[:, :, 2:4], reduction="none"), dim=2)
                # Use candidate indices instead of pos bbox indices for loss computation
                loc_loss = 2 * \
                    torch.sum(loc_bce[candidate_ids] + loc_mse[candidate_ids])

                conf_bce = bce(
                    raw_pred[:, :, 4], pgt_t[:, :, 4].float(), reduction="none")
                conf_loss = torch.sum(conf_bce[candidate_ids])

                prob_bce = torch.sum(
                    bce(raw_pred[:, :, 5:], pgt_t[:, :, 5:].float(), reduction="none"), dim=2)
                prob_loss = torch.sum(prob_bce[candidate_ids])

                losses = torch.stack([loc_loss, conf_loss, prob_loss])

                instance_dict = {}
                for loss_id, t in enumerate(losses):
                    g = torch.autograd.grad(
                        t, list(self.weight_dict.values()), grad_outputs=None, retain_graph=True)
                    instance_dict[self.cfg.loss_contributions[loss_id]] = dict([(list(self.weight_dict.keys())[ind],
                                                                                map_grad_tensor_to_numbers(v))
                                                                               for ind, v in enumerate(g)])
                gradient_list.append(instance_dict)

            xmin, ymin, xmax, ymax = cxcywh_to_border(soft_thr_coords)

            df = pd.DataFrame(columns=self.cfg.columns)
            df["file_path"] = [file_name for _ in range(num_detections)]
            df["xmin"] = xmin.cpu().numpy()
            df["ymin"] = ymin.cpu().numpy()
            df["xmax"] = xmax.cpu().numpy()
            df["ymax"] = ymax.cpu().numpy()
            df["s"] = soft_thr_det[:, 4].cpu().numpy()

            probs = soft_thr_det[:, 5:].cpu().numpy()
            df["category_idx"] = np.argmax(probs, axis=1)
            df["probs_sum"] = np.sum(probs, axis=1)
            df[self.cfg.probabilities] = probs
            df["dataset_box_id"] = np.arange(
                id_count, id_count+total_num_predictions)[soft_thr_mask[0].cpu().numpy()]
            df["gradient_metrics"] = gradient_list
            id_count += total_num_predictions

            file_id = file_name.split("/")[-1].split(".")[0]

            if dir_id is not None:
                csv_folder = f"{self.cfg.target_folder}/{dir_id}/gradient_metrics/csv"
                os.makedirs(csv_folder, exist_ok=True)
                df.to_csv(f"{csv_folder}/{file_id}_grads.csv")
            else:
                print("No directory identifier specified.")


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


if __name__ == "__main__":
    cfg = GradientsConfig()
    GetGradientMetrics(cfg).run_gradients(dir_id="retrain2")
