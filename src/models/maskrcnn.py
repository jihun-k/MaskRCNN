import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from models.backbone.res_fpn import Res50FPN
from models.proposal.rpn import RPN
from models.roi_head.roi_head import RoIHead


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
                images  ([b, c, w, h])
                targets (List(boxes))
        '''
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, features, targets)
        # detections, detector_losses = self.roi_heads(features, proposals, targets)

        losses = {}
        # losses.update(detector_losses)
        losses.update(proposal_losses)
    
        return proposals, losses


class MaskRCNN(GenerallizedRCNN):
    ''' mask r cnn '''
    def __init__(self, cfg):
        transform = transforms.Compose(
            transforms.Resize(1024, interpolation=Image.BILINEAR)
            
        )
        backbone = Res50FPN(imagenet_pretrained=True)
        # backbone = torchvision.models.resnet50(pretrained=True)
        # backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        backbone.eval()

        rpn = RPN(cfg)
        rpn.train()
        roi_heads = RoIHead()

        super(MaskRCNN, self).__init__(backbone, rpn, roi_heads, transform)
