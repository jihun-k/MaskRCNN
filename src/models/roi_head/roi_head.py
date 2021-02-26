import numpy as np
import torch.nn as nn
import torchvision
from torch import ceil, floor


class RoIHead(nn.Module):
    ''' roi head '''
    def __init__(self):
        super(RoIHead, self).__init__()
        self.box_head = nn.Sequential(
                            nn.Linear(7*7*256, 1024),
                            nn.Linear(1024, 1024)
        )
        self.cls_score = nn.Linear(1024, 80)
        self.bbox_pred = nn.Linear(1024, 80*4)

    def forward(self, features, proposals):
        x = self.roi_align(features, proposals)
        for x in features:
            x = torchvision.ops.roi_align(x, proposals)
        x = self.box_head(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        detections = {}
        detections["boxes"] = bbox_pred
        detections["labels"] = cls_score

        # todo: calculate losses
        losses = 0

        return detections, losses
