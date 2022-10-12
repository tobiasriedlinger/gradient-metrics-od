import numpy as np
import torch
from src.datasets.pascal_voc import VocDetectionBoundingBox
from torch.utils.data.dataloader import DataLoader

from src.utils import xywh_to_border
from src.datasets.kitti_torchvision import collate_fn


class VocDetBBoxTV(VocDetectionBoundingBox):
    VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor')
    str_to_id = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __init__(self,
                 img_root,
                 year="2012",
                 img_size=608,
                 transform="default"
                 ):
        super(VocDetBBoxTV, self).__init__(img_root,
                                           year,
                                           img_size,
                                           transform)

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
        areas = label_tensor[:, 2] * label_tensor[:, 3]
        label_tensor = xywh_to_border(label_tensor)

        target = dict(boxes=label_tensor[..., :4].float(),
                      labels=torch.argmax(label_tensor[..., 5:], dim=1)+1,
                      image_id=torch.tensor([int(index)]),
                      area=areas,
                      iscrowd=torch.zeros_like(areas, dtype=torch.int64))

        return trans_img, target


def _voc_one_hot_label(category_id):
    return torch.from_numpy(np.eye(20, dtype=np.float)[category_id]).float()


def voc_dataloader(img_root,
                   annotation="2012",
                   img_size=800,
                   train=True,
                   batch_size=4):
    transform = "random" if train else "default"
    ds = VocDetBBoxTV(img_root,
                      annotation,
                      img_size,
                      transform)

    return DataLoader(ds, batch_size, train, collate_fn=collate_fn)
