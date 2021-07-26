"""
Routine for full Monte-Carlo Dropout inference (all possible output boxes).
"""

import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from src.access import load_yolov3_model, load_dataset
from src.utils import untransform_bboxes
from production.get_forwards import cxcywh_to_border


class MCConfig(object):
    weight_path = "/home/OD/yolov3-torch-lfs/weights/coco/retrain/" \
                 "ckpt_lr_1e-5e+00_ep_2_map_0.546_mf1_0.291.pt"
    img_size = 608
    # img_dir = "/home/datasets/KITTI/test_split/images"
    img_dir = "/home/datasets/COCO/2017/val2017"
    # img_dir = "/home/datasets/PASCAL_VOC/test/VOCdevkit/VOC2007/JPEGImages"
    batch_size = 4
    dev = torch.device("cuda:0")
    target_folder = "/home/OD/yolov3-torch-lfs/uncertainty_metrics/coco"
    num_classes = 80
    num_attrib = num_classes + 5
    last_layer_dim = 3 * num_attrib
    dropout_rate = 0.5
    num_runs = 30


class MCInference(object):
    def __init__(self, config):
        self.cfg = config
        self.num_runs = config.num_runs
        self.model = load_yolov3_model(weight_path=config.weight_path,
                                       device=config.dev,
                                       cfg=config,
                                       ckpt=True)
        self.dataloader = load_dataset(name="image_folder",
                                       img_dir=config.img_dir,
                                       annot_dir=None,
                                       img_size=config.img_size,
                                       batch_size=config.batch_size,
                                       n_cpu=4,
                                       shuffle=False,
                                       augment=False)

    def run_inference(self, dir_id=None):
        print(f"Dropout rate: {self.cfg.dropout_rate}")
        id_count = 0
        for batch in tqdm(self.dataloader):
            file_names = batch[0]
            imgs = batch[1].to(self.cfg.dev)
            scales = batch[2].to(self.cfg.dev)
            paddings = batch[3].to(self.cfg.dev)
            num_pred = 3 * 21 * (self.cfg.img_size // 32)**2
            self.model.eval()

            xmins = np.zeros((self.cfg.batch_size, num_pred, self.num_runs))
            ymins = np.zeros((self.cfg.batch_size, num_pred, self.num_runs))
            xmaxs = np.zeros((self.cfg.batch_size, num_pred, self.num_runs))
            ymaxs = np.zeros((self.cfg.batch_size, num_pred, self.num_runs))
            s = np.zeros((self.cfg.batch_size, num_pred, self.num_runs))

            prob = np.zeros((self.cfg.batch_size, num_pred, self.cfg.num_classes, self.num_runs))
            prob_sum = np.zeros((self.cfg.batch_size, num_pred, self.num_runs))
            for i in range(self.num_runs):
                with torch.no_grad():
                    det = torch.cat(self.model(imgs), dim=1)
                for j, (name, det, scale, padding) in enumerate(zip(file_names, det, scales, paddings)):
                    coords = untransform_bboxes(det[..., :4], scale, padding)
                    cpu_det = det.detach().cpu().numpy()
                    cpu_coords = coords.detach().cpu().numpy()

                    s[j, :, i] = cpu_det[:, 4]
                    prob[j, :, :, i] = cpu_det[:, 5:]
                    prob_sum[:, :, i] = np.sum(prob, axis=2)[:, :, i]

                    xmin, ymin, xmax, ymax = cxcywh_to_border(cpu_coords)
                    xmins[j, :, i] = xmin
                    ymins[j, :, i] = ymin
                    xmaxs[j, :, i] = xmax
                    ymaxs[j, :, i] = ymax

            for b in range(self.cfg.batch_size):
                raw_name = file_names[b].split("/")[-1].split(".")[0]
                df = pd.DataFrame()
                df["file_path"] = [file_names[b] for _ in range(num_pred)]
                df["xmin_mean"] = np.mean(xmins[b, :, :], axis=1)
                df["ymin_mean"] = np.mean(ymins[b, :, :], axis=1)
                df["xmax_mean"] = np.mean(xmaxs[b, :, :], axis=1)
                df["ymax_mean"] = np.mean(ymaxs[b, :, :], axis=1)
                df["s_mean"] = np.mean(s[b, :, :], axis=1)
                df[[f"prob_{i}_mean" for i in range(self.cfg.num_classes)]] = np.mean(prob[b, :, :, :], axis=2)
                df["prob_sum_mean"] = np.mean(prob_sum[b, :, :], axis=1)
                df["xmin_std"] = np.std(xmins[b, :, :], axis=1)
                df["ymin_std"] = np.std(ymins[b, :, :], axis=1)
                df["xmax_std"] = np.std(xmaxs[b, :, :], axis=1)
                df["ymax_std"] = np.std(ymaxs[b, :, :], axis=1)
                df["s_std"] = np.std(s[b, :, :], axis=1)
                df[[f"prob_{i}_std" for i in range(self.cfg.num_classes)]] = np.std(prob[b, :, :, :], axis=2)
                df["prob_sum_std"] = np.std(prob_sum[b, :, :], axis=1)
                df["dataset_box_id"] = range(id_count, id_count + num_pred)
                id_count += num_pred

                if dir_id is not None:
                    tgt_path = f"{self.cfg.target_folder}/{dir_id}/mc_uncertainty/csv"
                else:
                    tgt_path = f"{self.cfg.target_folder}/mc_uncertainty/csv"
                os.makedirs(tgt_path, exist_ok=True)
                df.to_csv(f"{tgt_path}/{raw_name}_mc.csv")


if __name__ == "__main__":
    conf = MCConfig()
    MCInference(conf).run_inference(dir_id="retrain2")
