import numpy as np
import torch
from torchvision.datasets import VOCDetection

from src.datasets.transforms import default_transform_fn, random_transform_fn_mr
from src.utils import xywh_to_cxcywh


class VocDetectionBoundingBox(VOCDetection):
    VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor')
    str_to_id = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __init__(self, img_root, year="2012", img_size=608, transform="default"):
        img_set = "test" if year == "2007" else "train"
        super(VocDetectionBoundingBox, self).__init__(img_root, year, img_set)
        self._img_size = img_size
        if transform == "default":
            self._tf = default_transform_fn(img_size)
        elif transform == "random":
            self._tf = random_transform_fn_mr(img_size)
        else:
            raise ValueError("input transform can only be 'default' or 'random'.")

    def __getitem__(self, index):
        img, targets = super(VocDetectionBoundingBox, self).__getitem__(index)
        anns = targets["annotation"]["object"]
        labels = []
        for target in anns:
            category_id = self.str_to_id[target["name"]]
            bdict = target["bndbox"]
            xmin, ymin, xmax, ymax = float(bdict["xmin"]), float(bdict["ymin"]), float(bdict["xmax"]), \
                float(bdict["ymax"])
            bbox = torch.tensor([xmin, ymin, xmax-xmin, ymax-ymin])
            one_hot = _voc_one_hot_label(category_id)
            conf = torch.tensor([1.])

            label = torch.cat([bbox, conf, one_hot])
            labels.append(label)

        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, 25))

        trans_img, label_tensor = self._tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)

        return trans_img, label_tensor, label_tensor.size(0)


def _voc_one_hot_label(category_id):
    return torch.from_numpy(np.eye(20, dtype=np.float)[category_id]).float()
