import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from src.datasets.kitti import KittiDetectionBoundingBox
from src.utils import xywh_to_border


class KittiDetBBoxTV(KittiDetectionBoundingBox):
    KITTI_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person", "Cyclist", "Tram", "Misc")
    str_to_id = dict(zip(KITTI_CLASSES, range(len(KITTI_CLASSES))))

    def __init__(self,
                 img_root,
                 annot_dir="/home/datasets/KITTI/training/kitti_train.txt",
                 img_size=608,
                 transform="default"
                 ):
        super(KittiDetBBoxTV, self).__init__(img_root,
                                             annot_dir,
                                             img_size,
                                             transform)

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
            label_tensor = torch.zeros((0, 13+1))

        trans_img, label_tensor = self._tf(img, label_tensor)
        areas = label_tensor[:, 2] * label_tensor[:, 3]
        label_tensor = xywh_to_border(label_tensor)

        target = dict(boxes=label_tensor[..., :4].long(),
                      labels=torch.argmax(label_tensor[..., 5:], dim=1)+1,
                      image_id=torch.tensor([int(index)]),
                      area=areas,
                      iscrowd=torch.zeros_like(areas, dtype=torch.int64))

        return trans_img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def kitti_dataloader(img_root,
                     annotation_file="/home/datasets/KITTI/training/kitti_train.txt",
                     img_size=800,
                     train=True,
                     batch_size=4):
    transform = "random" if train else "default"
    ds = KittiDetBBoxTV(img_root,
                        annotation_file,
                        img_size,
                        transform)

    return DataLoader(ds, batch_size, train, collate_fn=collate_fn)
