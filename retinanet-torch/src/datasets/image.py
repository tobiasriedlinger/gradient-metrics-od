import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .transforms import default_transform_fn
from src.datasets.kitti_torchvision import collate_fn


def _get_padding(h, w):
    """Generate the size of the padding given the size of the image,
    such that the padded image will be square.
    Args:
        h (int): the height of the image.
        w (int): the width of the image.
    Return:
        A tuple of size 4 indicating the size of the padding in 4 directions:
        left, top, right, bottom. This is to match torchvision.transforms.Pad's parameters.
        For details, see:
            https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad
        """
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    return (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)


class ImageFolder(Dataset):
    """The ImageFolder Dataset class."""

    def __init__(self, folder_path, img_size=416, sort_key=None):
        self.files = sorted(glob.glob('{}/*.*'.format(folder_path)), key=sort_key)
        self.img_shape = (img_size, img_size)
        self._img_size = img_size
        self._transform = default_transform_fn(img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        max_size = max(w, h)
        _padding = _get_padding(h, w)
        # Add padding
        transformed_img_tensor, _ = self._transform(img)

        scale = self._img_size / max_size

        return img_path, transformed_img_tensor, scale, np.array(_padding)

    def __len__(self):
        return len(self.files)


def image_dataloader(img_dir,
                     img_size,
                     batch_size=4):
    ds = ImageFolder(img_dir, img_size)

    return DataLoader(ds, batch_size, collate_fn=collate_fn)
