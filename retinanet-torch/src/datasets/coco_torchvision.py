import torch
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import DataLoader

from .transforms import default_transform_fn, random_transform_fn_mr
from src.utils import xywh_to_border
from src.datasets.kitti_torchvision import collate_fn
from src.datasets.coco import _coco_category_to_one_hot, _delete_coco_empty_category

NUM_CLASSES_COCO = 80
NUM_ATTRIB = 4 + 1 + NUM_CLASSES_COCO


class CocoDetBBoxTV(CocoDetection):

    def __init__(self, img_root, ann_file_name, img_size, transform='default', category='all'):
        super(CocoDetBBoxTV, self).__init__(img_root, ann_file_name)
        self._img_size = img_size
        if transform == 'default':
            self._tf = default_transform_fn(img_size)
        elif transform == 'random':
            self._tf = random_transform_fn_mr(img_size)
        else:
            raise ValueError("input transform can only be 'default' or 'random'.")
        if category == 'all':
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category

    def __getitem__(self, index):
        img, targets = super(CocoDetBBoxTV, self).__getitem__(index)
        labels = []
        classes = []
        for target in targets:
            bbox = torch.tensor(target['bbox'], dtype=torch.float32)  # in xywh format
            category_id = target['category_id']
            if (not self.all_categories) and (category_id != self.category_id):
                continue
            one_hot_label = _coco_category_to_one_hot(category_id, dtype='float32')
            classes.append(_delete_coco_empty_category(category_id) + 1)
            conf = torch.tensor([1.])
            label = torch.cat((bbox, conf, one_hot_label))
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((1, NUM_ATTRIB))
        transformed_img_tensor, label_tensor = self._tf(img, label_tensor)
        areas = label_tensor[:, 2] * label_tensor[:, 3]
        label_tensor = xywh_to_border(label_tensor)

        try:
            target = dict(boxes=label_tensor[..., :4].long(),
                          labels=torch.tensor(classes, dtype=torch.int64),
                          image_id=torch.tensor([int(index)]),
                          area=areas,
                          iscrowd=torch.zeros_like(areas, dtype=torch.int64))
        except RuntimeError:
            print(label_tensor)

        return transformed_img_tensor, target


def coco_dataloader(img_root,
                    annotation_file="/home/datasets/COCO/2017/annotations/instances_train2017.json",
                    img_size=800,
                    train=True,
                    batch_size=4):
    transform = "random" if train else "default"
    ds = CocoDetBBoxTV(img_root,
                       annotation_file,
                       img_size,
                       transform)

    return DataLoader(ds, batch_size, train, collate_fn=collate_fn)
