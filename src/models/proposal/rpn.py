import torch
from torch import nn
from torch.nn.modules import padding


class AnchorGenerator(nn.Module):
    '''
        region proposal network 
        Args:
            sizes:
            aspect_ratios: 
    '''
    def __init__(self, 
                 sizes=((128, 256, 512),),
                 aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()
        return

    def forward(self, features):
        grid_sizes = [feature.shape[-2:] for feature in features]
        # anchors = 
        return

class RPNHead(nn.Module):
    '''
        rpn head for classifying object/nonobject

        input is list of [b,c,w,h] features, each element corresponding to f2~f6

        output is list of k-channels classification [b, k, w, h] and
            4k-channels bbox prediction [b, 4k, w, h]

        Args:
            in_channels : # of input feature channels
            num_anchors : # of anchors (size * aspect)
    '''
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x : List of [b,c,w,h]
        logits = []
        bbox = []
        for feature in x:
            t = self.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox.append(self.bbox_pred(t))
        return logits, bbox


class RPN(nn.Module):
    ''' region proposal network '''
    def __init__(self):
        super(RPN, self).__init__()
        self.head = RPNHead(in_channels=256, num_anchors=3)
        self.anchor_generator = AnchorGenerator()
        return

    def forward(self, images, features, targets):
        '''
            Args:
                images : batch images
                features : List of features from backbone
                targets : ground truth boxes
        '''
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        # todo
        proposals = []
        losses = []
        return proposals, losses
