from typing import Dict

import torch
import torch.nn as nn
import torchvision
from config import Config
from PIL import Image
from torchvision import transforms

from models.backbone.res_fpn import Res50FPN
from models.proposal.rpn import RPN
from models.roi_head.roi_head import RoIHead


class GenerallizedRCNN(nn.Module):
    ''' generalized rcnn '''
    def __init__(
            self,
            backbone,
            rpn,
            roi_heads
        ):
        super(GenerallizedRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(
            self,
            images: torch.Tensor,
            targets: Dict[str, torch.Tensor]
        ):
        '''
            Args:
                images  ([b, c, w, h])
                targets (List(boxes))
        '''
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, targets)

        losses = {}
        # losses.update(detector_losses)
        losses.update(proposal_losses)
    
        return proposals, losses


class MaskRCNN(GenerallizedRCNN):
    ''' mask r cnn '''
    def __init__(self, cfg: Config):
        backbone = Res50FPN(cfg)

        rpn = RPN(cfg)
        roi_heads = RoIHead()

        super(MaskRCNN, self).__init__(backbone, rpn, roi_heads)
