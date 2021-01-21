import torch
from torch.nn.modules import padding
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models.backbone.res_fpn import Res50FPN

class GenerallizedRCNN(nn.Module):
    ''' generalized rcnn '''
    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GenerallizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets):
        '''
            Args:
                images  ([b, c, h, w])
                targets ([b, Dict["boxes", "labels"]])
        '''

        features = self.backbone(images, targets)
        
        # proposals, proposal_losses = self.rpn(features, targets)
        # detections, detector_losses = self.roi_heads(features, proposals, targets)

        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)
    
        # return losses, detections


class MaskRCNN(GenerallizedRCNN):
    ''' mask r cnn '''
    def __init__(self):
        transform = transforms.Compose(
            transforms.Resize(1024, interpolation=Image.BILINEAR)
            
        )
        backbone = Res50FPN()
        rpn = nn.Module()
        roi_heads = nn.Module()

        super(MaskRCNN, self).__init__(backbone, rpn, roi_heads, transform)