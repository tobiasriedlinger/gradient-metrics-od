"""
Collection of accessibility routines for data sets and model.
"""

import torch
import collections
import logging
from torch.utils.data import DataLoader

from src.yolov3_config import YOLOV3Config
from src.model import YoloNetV3
from src.datasets.utils import collate_img_label_fn
from src.datasets.image import ImageFolder
from src.datasets.caltech import CaltechPedDataset
from src.datasets.coco import CocoDetectionBoundingBox
from src.datasets.pascal_voc import VocDetectionBoundingBox
from src.datasets.kitti import KittiDetectionBoundingBox


def config_device(cpu_only: bool):
    """
    Set main torch device for model and dataloader based on configuration settings.
    Args:
        cpu_only (bool): Flag indicating whether to exclusively use CPU.
    Returns:
        (torch.device).
    """
    if not cpu_only:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logging.warning('CUDA device is not available. Will use CPU')
    else:
        use_cuda = False
    _device = torch.device("cuda:0" if use_cuda else "cpu")
    return _device


def load_yolov3_model(weight_path,
                      device,
                      ckpt=False,
                      mode='eval',
                      cfg=YOLOV3Config(),
                      num_classes=None,
                      transfer=False):
    """
    Loading routine for YOLOv3 for accessibility.
    Args:
        weight_path (str): Path to checkpoint or weight file.
        device (torch.device): Device to which to send model.
        ckpt (bool): Flag indicating whether file in weight_path is a training checkpoint or a pure weight file.
        mode (str): Either "eval" or "train" for respective model setting.
        cfg (Object): Optional Configuration object implementing the attributes [num_classes, num_attrib,
                        last_layer_dim, dropout_rate]
        num_classes (int): Optional indicator of dataset class count in case cfg is not given or transfer learning.
        transfer (bool): Opitonal flag indicating whether backbone weights should be used with the detection heads
                        re-initialized.
    Returns:
        _model (YoloNetV3): Model with given settings.
    """
    if num_classes is not None:
        cfg.num_classes = num_classes
        cfg.num_attrib = num_classes + 5
        cfg.last_layer_dim = 3 * cfg.num_attrib
    _model = YoloNetV3(cfg)
    if not ckpt:
        if transfer:
            default_weights = torch.load(weight_path)
            state_dict = collections.OrderedDict()
            for k in list(default_weights.keys()):
                if "darknet" in k:
                    state_dict[k.split("net.")[1]] = default_weights[k]
            _model.darknet.load_state_dict(state_dict)
        else:
            _model.load_state_dict(torch.load(weight_path))
    else:
        _model.load_state_dict(torch.load(weight_path)['model_state_dict'])
    _model.to(device)
    if mode == 'eval':
        _model.eval()
    elif mode == 'train':
        _model.train()
    else:
        raise ValueError("YoloV3 model can be only loaded in 'train' or 'eval' mode.")
    return _model


def load_dataset(name,
                 img_dir,
                 annot_dir,
                 img_size,
                 batch_size,
                 n_cpu,
                 shuffle,
                 augment,
                 **kwargs):
    """
    Loading routine for datasets in Dataloaders for accessibility.
    Args:
        name (str): Name of an implemented data set. So-far implemented: ["image_folder", "coco", "voc", "kitti",
                            "caltech"].
        img_dir (str): Path to folder containing the respective images for training or inference.
        annot_dir (str): Path to folder/file containing annotations for img_dir or string indicator for ground truth
                        (VOC).
        img_size (int): Square side length to which to rescale images.
        batch_size (int): Batch size for Dataloader.
        n_cpu (int): Number of CPU workers for Dataloader.
        shuffle (bool): Flag indicating whether to apply shuffling to data.
        augment (bool): Flag indicating whether to apply data augmentation to data.
        **kwargs:

    Returns:

    """
    if name == "image_folder":
        _dataset = ImageFolder(img_dir, img_size=img_size)
        _collate_fn = None
    elif name == "coco":
        _transform = 'random' if augment else 'default'
        _dataset = CocoDetectionBoundingBox(img_dir, annot_dir, img_size=img_size, transform=_transform)
        _collate_fn = collate_img_label_fn
    elif name == "voc":
        _transform = "random" if augment else "default"
        _dataset = VocDetectionBoundingBox(img_dir, year=annot_dir, img_size=img_size, transform=_transform)
        _collate_fn = collate_img_label_fn
    elif name == "kitti":
        _transform = "random" if augment else "default"
        _dataset = KittiDetectionBoundingBox(img_dir, annot_dir, img_size=img_size, transform=_transform)
        _collate_fn = collate_img_label_fn
    elif name == "caltech":
        _dataset = CaltechPedDataset(img_dir, img_size, **kwargs)
        _collate_fn = collate_img_label_fn
    else:
        raise TypeError("dataset types can only be 'image_folder', 'coco' or 'caltech'.")
    if _collate_fn is not None:
        _dataloader = DataLoader(_dataset, batch_size, shuffle, num_workers=n_cpu, collate_fn=_collate_fn)
    else:
        _dataloader = DataLoader(_dataset, batch_size, shuffle, num_workers=n_cpu)
    return _dataloader


def save_checkpoint_weight_file(model,
                                optimizer,
                                epoch,
                                batch,
                                loss,
                                weight_file_path):
    """
    Saving training checkpoint.
    Args:
        model (nn.Module)): Detector model to save.
        optimizer (optim.Optimizer): Optimizer for saving optimizer state.
        epoch (int): Current number of trained epochs.
        batch (int): Current batch number in epoch.
        loss (float): Current loss value.
        weight_file_path (str): Checkpoint file path.
    """
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, weight_file_path)
    logging.info("saving model at epoch {}, batch {} to {}".format(epoch, batch, weight_file_path))
    return
