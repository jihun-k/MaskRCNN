import numpy as np
import torch.nn as nn
from torch import ceil, floor


class RoIAlign(nn.Module):
    '''
        roi align. [N, c, w, h], [N, [x1, y1, x2, y2]] -> [N, c, 7, 7]
    '''
    def __init__(self, output_size):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, feature, bbox):
        out = np.zeros((feature.shape[0], feature.shape[1], self.output_size, self.output_size))
        t = np.zeros((feature.shape[0], feature.shape[1], 2*self.output_size, 2*self.output_size))
        x_steps = (bbox[:,2] - bbox[:,0]) / (2 * self.output_size)
        y_steps = (bbox[:,3] - bbox[:,1]) / (2 * self.output_size)
        x_coords = np.zeros((feature.shape[0], 2*self.output_size))
        y_coords = np.zeros((feature.shape[0], 2*self.output_size))
        for i in range(feature[0]):
            x_coords = range(bbox[i,0], bbox[i,2], x_steps)
            y_coords = range(bbox[i,1], bbox[i,3], y_steps)

        for i in range(self.output_size):
            for j in range(self.output_size):
                #    a    b
                # c |   |   |
                # d |   |   |
                x = x_coords[:,i]
                y = y_coords[:,j]
                top_left = feature[:,:, floor(i), floor(j)]
                top_right = feature[:,:, ceil(i), floor(j)]
                bottom_left = feature[:,:, floor(i), ceil(j)]
                bottom_right = feature[:,:, ceil(i), ceil(j)]
                a = i - floor(i)
                c = ceil(j) - j
                out[:,:,i,j] = a * c * top_left + (1-a) * c * top_right + a * (1-c) * bottom_left + (1-a) * (1-c) * bottom_right

        return self.maxpool(out)

class RoIHead(nn.Module):
    ''' roi head '''
    def __init__(self):
        super(RoIHead, self).__init__()
        self.roi_align = RoIAlign(7)
        self.box_head = nn.Sequential(
                            nn.Linear(7*7*256, 1024),
                            nn.Linear(1024, 1024)
        )
        self.cls_score = nn.Linear(1024, 80)
        self.bbox_pred = nn.Linear(1024, 80*4)

    def forward(self, features, proposals):
        x = self.roi_align(features, proposals)
        x = self.box_head(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        detections = {}
        detections["boxes"] = bbox_pred
        detections["labels"] = cls_score

        # todo: calculate losses
        losses = 0

        return detections, losses
