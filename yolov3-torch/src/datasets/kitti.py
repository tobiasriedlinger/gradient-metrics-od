import numpy as np
import torch
from PIL import Image


from src.datasets.transforms import default_transform_fn, random_transform_fn_xr
from src.utils import xywh_to_cxcywh


class KittiDetectionBoundingBox(torch.utils.data.Dataset):
    KITTI_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person", "Cyclist", "Tram", "Misc")
    str_to_id = dict(zip(KITTI_CLASSES, range(len(KITTI_CLASSES))))

    def __init__(self,
                 img_root,
                 annot_dir="/home/datasets/KITTI/training/kitti_train.txt",
                 img_size=608,
                 transform="default"
                 ):
        self._img_size = img_size
        self._split = "test" if "test" in annot_dir else "train"

        if self._split == "test":
            self.img_folder = f"{img_root}/test_split/images"
        elif self._split == "train":
            self.img_folder = f"{img_root}/training/image_2"
        with open(f"{annot_dir}", "r") as f:
            self.labels = f.readlines()

        if transform == "default":
            self._tf = default_transform_fn(img_size)
        elif transform == "random":
            self._tf = random_transform_fn_xr(img_size)
        else:
            raise ValueError("input transform can only be 'default' or 'random'.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label_str = self.labels[index].split(".png ")
        img_id = label_str[0].split("/")[-1]
        img = Image.open(f"{self.img_folder}/{img_id}.png").convert("RGB")

        labels_ls = []
        for lab in label_str[1].split(" "):
            entries = lab.split(",")
            category_id = int(entries[-1])
            xmin, ymin, xmax, ymax = float(entries[0]), float(entries[1]), float(entries[2]), float(entries[3])
            bbox = torch.tensor([xmin, ymin, xmax-xmin, ymax-ymin])
            # One hot vector for class list of length 8 (can be made neater by using one-hot function implemented by
            # PyTorch.
            one_hot = torch.from_numpy(np.eye(8, dtype="uint8")[category_id]).float()
            conf = torch.tensor([1.])

            label = torch.cat([bbox, conf, one_hot])
            labels_ls.append(label)

        if labels_ls:
            label_tensor = torch.stack(labels_ls)
        else:
            label_tensor = torch.zeros((0, 13))

        trans_img, label_tensor = self._tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)
        # After this line, the coordinates are encoded in cx (horizontal), cy (vertical), w, h w.r.t. rescaled
        # (img_size) image coordinates.

        return trans_img, label_tensor, label_tensor.size(0)
