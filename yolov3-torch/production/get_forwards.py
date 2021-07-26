"""
Routine for network forward pass (without NMS!), e.g. for evaluation or MC Dropout inference.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.access import load_yolov3_model, load_dataset
from src.utils import untransform_bboxes
from src.inference import cxcywh_to_border


class ForwardConfig(object):
    """
    Dummy class containing the necessary settings for a forward pass.
    """

    # Detector settings
    num_classes = 80
    num_attrib = num_classes + 5
    last_layer_dim = 3 * num_attrib
    dropout_rate = 0.0
    score_thresh = 1e-4
    weight_path = "/home/OD/yolov3-torch-lfs/weights/yolov3_original.pt"
    weight_is_ckpt = True

    # Data settings
    img_size = 608
    img_dir = "/home/datasets/COCO/2017/val2017"
    batch_size = 16

    target_folder = "/home/OD/yolov3-torch-lfs/detection_eval/coco2017"
    dev = torch.device("cuda:0")


class GetForwards(object):
    """
    Forward class importable in other places.
    Implements run_forward
    """
    def __init__(self, config, ignore_config=False):
        self.cfg = config
        self.csv_folder = None
        if ignore_config:
            pass
        else:
            self.model = load_yolov3_model(weight_path=self.cfg.weight_path,
                                           device=self.cfg.dev,
                                           cfg=config,
                                           ckpt=self.cfg.weight_is_ckpt)
            self.dataloader = load_dataset(name="image_folder",
                                           img_dir=self.cfg.img_dir,
                                           annot_dir=None,
                                           img_size=self.cfg.img_size,
                                           batch_size=self.cfg.batch_size,
                                           n_cpu=4,
                                           shuffle=False,
                                           augment=False)

    def run_forward(self, model=None, img_dir=None, n_classes=None, dir_id=None, save=True):
        """
        Performs forward pass on a set of images located in given directory and saves computed predictions as .csv-files
        Args:
            model (YoloNetV3): YOLOv3 model passed for inference. Defaults to the one generated in __init__ from
            settings.
            img_dir (str): Directory path where images for inference are located. If None, Dataloader generated in
            __init__ is used for inference.
            n_classes (int): Number of classes in data set used to construct column names for result.
            dir_id (str): Subdirectory identifier in case inference should be saved to new subdirectory of
            self.cfg.target_folder.
            save (bool): Flag indicating whether .csv-files are supposed to be saved to self.cfg.target_folder or
            self.cfg.target_folder/dir_id.

        Returns (list[DataFrame]):
            Collection of inferred Pandas DataFrames containing the complete forward pass over the image folder.
        """
        if n_classes is None:
            n_classes = self.cfg.num_classes
        if img_dir is None:
            dl = self.dataloader
        else:
            dl = load_dataset(name="image_folder",
                              img_dir=img_dir,
                              annot_dir=None,
                              img_size=self.cfg.img_size,
                              batch_size=self.cfg.batch_size,
                              n_cpu=4,
                              shuffle=False,
                              augment=False)
        columns = ["dataset_box_id", "file_path", "xmin", "ymin", "xmax", "ymax", "s", "category_idx", "prob_sum"]\
            + [f"prob_{i}" for i in range(n_classes)]

        if model is None:
            model = self.model
        if dir_id is not None:
            print(f"Dir id: {dir_id}")
        id_count = 0
        df_list = []
        for batch in tqdm(dl):
            file_names = batch[0]
            imgs = batch[1].to(self.cfg.dev)
            scales = batch[2].to(self.cfg.dev)
            paddings = batch[3].to(self.cfg.dev)
            model.eval()

            with torch.no_grad():
                detections = torch.cat(model(imgs), dim=1)

            # Sort out batch detection to image-wise DataFrame.
            for name, det, scale, padding in zip(file_names, detections, scales, paddings):
                coords = untransform_bboxes(det[..., :4], scale, padding)
                cpu_det = det.detach().cpu().numpy()
                cpu_coords = coords.detach().cpu().numpy()

                score = cpu_det[..., 4]
                # Mask for score thresholding
                score_mask = (score >= self.cfg.score_thresh)
                num_entries = int(np.sum(score_mask))
                score = score[score_mask]
                probs = cpu_det[score_mask, 5:]
                prob_sum = np.sum(probs, axis=1)
                category_ids = np.argmax(probs, axis=1)

                xmin, ymin, xmax, ymax = cxcywh_to_border(cpu_coords)
                xmin = xmin[score_mask]
                ymin = ymin[score_mask]
                xmax = xmax[score_mask]
                ymax = ymax[score_mask]

                df = pd.DataFrame(columns=columns)
                df["file_path"] = [name for _ in range(num_entries)]
                df["xmin"] = xmin
                df["ymin"] = ymin
                df["xmax"] = xmax
                df["ymax"] = ymax
                df["s"] = score
                df["category_idx"] = category_ids
                df["prob_sum"] = prob_sum
                df.loc[:, "prob_0":] = probs
                current_index = list(df.index)
                df["dataset_box_id"] = range(id_count, id_count+len(current_index))
                id_count += len(current_index)

                raw_name = name.split("/")[-1].split(".")[0]

                if (dir_id is not None) and save:
                    self.csv_folder = f"{self.cfg.target_folder}/{dir_id}/csv"
                    os.makedirs(self.csv_folder, exist_ok=True)
                else:
                    self.csv_folder = self.cfg.target_folder
                if save:
                    df.to_csv(f"{self.csv_folder}/{raw_name}.csv")
                df_list.append(df)

        return df_list


if __name__ == "__main__":
    forward_config = ForwardConfig()
    GetForwards(config=forward_config).run_forward(dir_id="forward")
