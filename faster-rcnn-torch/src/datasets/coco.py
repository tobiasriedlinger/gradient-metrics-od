import numpy as np
import torch
from torchvision.datasets import CocoDetection

from .transforms import default_transform_fn, random_transform_fn_mr
from src.utils import xywh_to_cxcywh

MISSING_IDS = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
NUM_CLASSES_COCO = 80
NUM_ATTRIB = 4 + 1 + NUM_CLASSES_COCO


class CocoDetectionBoundingBox(CocoDetection):

    def __init__(self, img_root, ann_file_name, img_size, transform='default', category='all'):
        super(CocoDetectionBoundingBox, self).__init__(img_root, ann_file_name)
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
        img, targets = super(CocoDetectionBoundingBox, self).__getitem__(index)
        labels = []
        for target in targets:
            bbox = torch.tensor(target['bbox'], dtype=torch.float32)  # in xywh format
            category_id = target['category_id']
            if (not self.all_categories) and (category_id != self.category_id):
                continue
            one_hot_label = _coco_category_to_one_hot(category_id, dtype='float32')
            conf = torch.tensor([1.])
            label = torch.cat((bbox, conf, one_hot_label))
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, NUM_ATTRIB))
        transformed_img_tensor, label_tensor = self._tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)
        return transformed_img_tensor, label_tensor, label_tensor.size(0)


def _coco_category_to_one_hot(category_id, dtype="uint"):
    """ convert from a category_id to one-hot vector, considering there are missing IDs in coco dataset."""
    new_id = _delete_coco_empty_category(category_id)
    return _category_to_one_hot(new_id, NUM_CLASSES_COCO, dtype)


def _category_to_one_hot(category_id, num_classes, dtype="uint"):
    """ convert from a category_id to one-hot vector """
    return torch.from_numpy(np.eye(num_classes, dtype=dtype)[category_id])


def _delete_coco_empty_category(old_id):
    """The COCO dataset has 91 categories but 11 of them are empty.
    This function will convert the 80 existing classes into range [0-79].
    Note the COCO original class index starts from 1.
    The converted index starts from 0.
    Args:
        old_id (int): The category ID from COCO dataset.
    Return:
        new_id (int): The new ID after empty categories are removed. """
    starting_idx = 1
    new_id = old_id - starting_idx
    for missing_id in MISSING_IDS:
        if old_id > missing_id:
            new_id -= 1
        elif old_id == missing_id:
            raise KeyError("illegal category ID in coco dataset! ID # is {}".format(old_id))
        else:
            break
    return new_id
