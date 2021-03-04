from typing import Dict, List

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
            roi_heads,
            cfg: Config,
        ):
        super(GenerallizedRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.cfg = cfg

    def forward(
            self,
            images: torch.Tensor,
            annotations: List[Dict[str, torch.Tensor]]
        ):
        '''
            Args:
                images  ([b, c, w, h])
                targets (List(boxes))
        '''
        
        gt_boxes = [] # List[Tensor]
        gt_labels = [] # List[Tensor]
        for annot in annotations:
            gt_boxes.append(annot["boxes"])
            gt_labels.append(annot["labels"])

        if self.cfg.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        if self.cfg.freeze_rpn:
            with torch.no_grad():
                proposals, proposal_losses = self.rpn(images, features, gt_boxes)
        else:
            proposals, proposal_losses = self.rpn(images, features, gt_boxes)

        detections, detector_losses = self.roi_heads(features, proposals, gt_boxes, gt_labels)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
    
        return detections, losses


class MaskRCNN(GenerallizedRCNN):
    ''' mask r cnn '''
    def __init__(self, cfg: Config):
        self.cfg = cfg

        backbone = Res50FPN(cfg)

        rpn = RPN(cfg)
        roi_heads = RoIHead(cfg)

        super(MaskRCNN, self).__init__(backbone, rpn, roi_heads, cfg)
