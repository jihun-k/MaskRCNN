from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
from utils import box_util


class RoIHead(nn.Module):
    ''' roi head '''
    def __init__(self):
        super(RoIHead, self).__init__()
        self.box_head = nn.Sequential(
            nn.Linear(12544, 1024),
            nn.Linear(1024, 1024),
        )
        self.cls_score = nn.Linear(1024, 2)
        self.bbox_pred = nn.Linear(1024, 4)

        # TODO initialize weight

    def roi_align(
        self,
        features: List[torch.Tensor],
        proposals: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        '''
            roi align

            Args:
                features:
                    for levels
                        feature map
                proposals:
                    for level features
                        for images
                            proposal
            Returns:
                roi:
                    for features
                        aligned roi

        '''
        output_size = 7
        roi = []
        for feature_level, proposal_level in zip(features, proposals):
            spatial_scale = feature_level.shape[-2] / 1024
            roi_level = torchvision.ops.roi_align(feature_level, [x for x in proposal_level], output_size, spatial_scale)
            roi.append(roi_level)
        return roi

    def label_proposals(
        self,
        proposals: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        images
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Args:
                proposals (Tensor):
                    proposals for single image [N, 4]
                    for images
                        proposals
                gt_boxes (List[Tensor]):
                    the ground-truth instances for each image. [num_gt_boxes, 4] (xywh)
                    for images
                        gt boxes
            Returns:
                list[Tensor]:
                    gt_labels [b, num_total_anchors]
                    List of #img tensors. i-th element is a vector of labels whose length is
                    the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                    Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                    class; 1 = positive class.
                list[Tensor]:
                    matched_gt_boxes [b, num_total_anchors, 4] (xyxy)
                    i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                    anchor. Values are undefined for those anchors not labeled as 1.
        '''
        
        pos_threshold = 0.7
        neg_threshold = 0.3

        gt_labels = []
        matched_gt_boxes = []
        for proposal_i, gt_boxes_i in zip(proposals, gt_boxes): # i th image in batch
            if torch.numel(gt_boxes_i) == 0:
                # all background
                gt_labels_i = torch.zeros(proposal_i.shape[0], device=proposal_i.device)
                matched_gt_boxes_i = torch.zeros(proposal_i.shape, device=proposal_i.device)
            else:
                gt_boxes_i = box_util.xywh_to_xyxy(gt_boxes_i)

                # (num_proposal) * (num_gt_box) match quality matrix 
                iou_matrix = box_util.get_iou(proposal_i, gt_boxes_i)

                matched_gt_score, matched_gt_idx = torch.max(iou_matrix, dim=1)
                matched_gt_boxes_i = gt_boxes_i[matched_gt_idx]

                # negative sample: max objectness of anchor is below threshold
                neg_idx = torch.where(matched_gt_score < neg_threshold)[0]

                # positive sample  
                pos_idx_a = torch.where(matched_gt_score > pos_threshold)[0] # iou > pos_threshold
                pos_idx_b = torch.max(iou_matrix, dim=0)[1] # maximum iou for each gt box
                pos_idx = torch.cat((pos_idx_a, pos_idx_b)).unique()
                
                gt_labels_i = torch.ones(proposal_i.shape[0], device=proposal_i.device) * -1
                gt_labels_i[neg_idx] = 0
                gt_labels_i[pos_idx] = 1

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def forward(
        self,
        features,
        proposals,
        targets
    ):
        roi = self.roi_align(features, proposals)
        batch_cls_score = []
        batch_bbox_pred = []
        for roi_i in roi:
            # for image
            x = torch.flatten(roi_i, start_dim=1)
            x = self.box_head(x)
            cls_score = self.cls_score(x)
            bbox_pred = self.bbox_pred(x)
            batch_cls_score.append(cls_score)
            batch_bbox_pred.append(bbox_pred)

        # TODO detection formatting
        losses = {}
        detections = {}
        detections["boxes"] = bbox_pred
        detections["labels"] = cls_score

        if self.training:
            # TODO label and sample proposals

            # TODO calculate loss
            losses = 0

        return detections, losses
