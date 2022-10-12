"""
Collection of accessibility routines for data sets and model.
"""

import torch
import collections
import logging

from src.model.retina_net import _validate_trainable_layers, RetinaNet
from src.model.backbone_utils import resnet_fpn_backbone
from src.model.fpn import LastLevelP6P7
from src.datasets.image import image_dataloader
from src.datasets.coco_torchvision import coco_dataloader
from src.datasets.voc_torchvision import voc_dataloader
from src.datasets.kitti_torchvision import kitti_dataloader


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


def load_retinanet_model(weight_path,
                         device,
                         ckpt=False,
                         mode='eval',
                         num_classes=None,
                         box_score_thr=0.0,
                         transfer=True,
                         unfreeze=False):
    """
    Loading routine for YOLOv3 for accessibility.
    Args:
        weight_path (str): Path to checkpoint or weight file.
        device (torch.device): Device to which to send model.
        ckpt (bool): Flag indicating whether file in weight_path is a training checkpoint or a pure weight file.
        mode (str): Either "eval" or "train" for respective model setting.
        num_classes (int): Optional indicator of dataset class count in case cfg is not given or transfer learning.
        transfer (bool): Opitonal flag indicating whether backbone weights should be used with the detection heads
                        re-initialized.
        unfreeze (bool):
        box_score_thr (float):
    Returns:
        _model (YoloNetV3): Model with given settings.
    """
    layers = 3 if unfreeze else 0
    trainable_backbone_layers = _validate_trainable_layers(True, layers, 5, 3)
    backbone = resnet_fpn_backbone("resnet50",
                                   True,
                                   returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256),
                                   trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone,
                      num_classes+1,
                      score_thresh=box_score_thr)

    if not ckpt:
        default_weights = torch.load(weight_path)
        if transfer:
            state_dict = collections.OrderedDict()
            for k in list(default_weights.keys()):
                if not "fc" in k:
                    state_dict[k] = default_weights[k]
            model.backbone.body.load_state_dict(state_dict)
        else:
            model.load_state_dict(default_weights)
    else:
        model.load_state_dict(torch.load(weight_path)['model_state_dict'])
    model.to(device)
    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()
    return model


def load_dataset(name,
                 img_dir,
                 annot_dir,
                 img_size,
                 batch_size,
                 train,
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
        train (bool): Flag indicating whether to apply data augmentation to data.
        **kwargs:

    Returns:

    """
    if name == "image_folder":
        return image_dataloader(img_dir,
                                img_size,
                                batch_size=batch_size)
    elif name == "coco":
        return coco_dataloader(img_dir,
                               annot_dir,
                               img_size,
                               train,
                               batch_size)
    elif name == "voc":
        return voc_dataloader(img_dir,
                              annot_dir,
                              img_size,
                              train,
                              batch_size)
    elif name == "kitti":
        return kitti_dataloader(img_dir,
                                annot_dir,
                                img_size,
                                train,
                                batch_size)
    else:
        raise TypeError("dataset types can only be 'image_folder', 'coco', 'voc' or 'kitti'.")


def save_checkpoint_weight_file(model,
                                optimizer,
                                epoch,
                                batch,
                                loss,
                                weight_file_path):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, weight_file_path)
    logging.info("saving model at epoch {}, batch {} to {}".format(epoch, batch, weight_file_path))
    return
