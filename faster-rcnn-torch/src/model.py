import torch
import torchvision as tv
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class FRCNN(torch.nn.Module):
    def __init__(self,
                 backbone="resnet50",
                 pretrained_backbone=True):
        self.backbone = resnet_fpn_backbone(backbone,
                                            pretrained_backbone)
        
