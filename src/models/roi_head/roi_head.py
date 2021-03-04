from typing import Dict, List, Tuple

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import box_util


class RoIHead(nn.Module):
    ''' roi head '''
    def __init__(
        self,
        cfg : config.Config,
    ):
        self.cfg = cfg
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
        proposals: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''
            Args:
                proposals (Tensor):
                    proposals for single image [N, 4]
                gt_boxes (List[Tensor]):
                    the ground-truth instances for each image. [num_gt_boxes, 4] (xywh)
            Returns:
                list[Tensor]:
                    gt_labels [b, num_total_anchors]
                list[Tensor]:
                    matched_gt_boxes [b, num_total_anchors, 4] (xyxy)
        '''
        
        threshold = 0.5

        matched_gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i, gt_labels_i in zip(gt_boxes, gt_labels): # i th image in batch
            matched_gt_labels_i = torch.zeros(proposals.shape[0], device=proposals.device)
            matched_gt_boxes_i = torch.zeros(proposals.shape, device=proposals.device)

            # (num_proposal) * (num_gt_box) match quality matrix 
            iou_matrix = box_util.get_iou(proposals, gt_boxes_i)

            matched_gt_score, matched_gt_idx = torch.max(iou_matrix, dim=1)
            neg_idx = torch.where(matched_gt_score < threshold)[0]
            matched_gt_idx = matched_gt_idx[matched_gt_idx]

            matched_gt_boxes_i = gt_boxes_i[matched_gt_idx]
            matched_gt_labels_i = gt_labels_i[matched_gt_idx]

            matched_gt_labels_i[neg_idx] = 0

            matched_gt_labels.append(matched_gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return matched_gt_labels, matched_gt_boxes

    def sample_anchors(self, gt_labels):
        for gt_label in gt_labels:
            
            positive = torch.where(gt_label >= 1)[0]
            negative = torch.where(gt_label == 0)[0]

            # protect against not enough positive examples
            num_pos = min(positive.numel(), 128)
            num_neg = 256 - num_pos
            # # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx = positive[perm1]
            neg_idx = negative[perm2]

            # self.writer.add_image_with_boxes("Image/pos_anchors_" + str(time.time()), images[0], box_util.cwh_to_xyxy(anchors[pos_idx]))

            gt_label.fill_(-1)
            gt_label.scatter_(0, pos_idx, 1)
            gt_label.scatter_(0, neg_idx, 0)
            
        return gt_labels

    def compute_loss(self, proposals, cls_score, pred_bbox, gt_labels, gt_boxes):
        '''
            Args:
                anchors                 list([b, A*w*h, 4]) (cx cy w h)
                pred_objectness_logits  list([b, A*w*h, A])
                pred_bbox_deltas        list([b, 4*A*w*h, 4*A]) (dx dy dw dh)
                gt_labels               list([A_sampled])
                gt_boxes                list([A_sampled]) (x1 y1 x2 y2)
            Returns:
                dict[loss name -> loss value]
        '''
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)

        pos_mask = gt_labels >= 1
        num_pos_samples = pos_mask.sum().item()
        num_neg_samples = (gt_labels == 0).sum().item()
        print("head # of pos/neg proposals: ", num_pos_samples, num_neg_samples)

        # box regression loss
        target_deltas = [box_util.get_deltas(proposals, box_util.xywh_to_cxy(gt_boxes_i)) for gt_boxes_i in gt_boxes]

        target_deltas = torch.stack(target_deltas)  # (N_images, R, 4) (dx dy dw dh) 

        pred_bbox = pred_bbox.unsqueeze(0) # TODO batch

        loss_box_reg = F.smooth_l1_loss(pred_bbox[pos_mask], target_deltas[pos_mask], reduction="sum")
        
        # classification loss
        valid_mask = gt_labels >= 0
        cls_score = cls_score # TODO batch
        loss_cls = F.cross_entropy(
            cls_score, gt_labels[0], reduction="sum")
        
        mini_batch_size = 256
        normalizer = mini_batch_size * num_images

        losses = {"loss_head_box":loss_box_reg / normalizer, "loss_head_cls":loss_cls / normalizer}

        return losses
        
    def forward(
        self,
        features: List[torch.Tensor],
        proposals: List[List[torch.Tensor]],
        gt_boxes: List[torch.Tensor] = None,
        gt_labels: List[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        '''
            Args:
                features (List[Tensor])
                proposals (List[List[Tensor]]):
                    (level, image)
                gt_boxes (List[Tensor])
                gt_labels (List[Tensor])
            
            Returns:
                detections (List[Dict[str, Tensor]])
                losses (Dict[str, Tensor])

        '''
        losses = {}
        detections = {}

        # suppose batch_size == 1
        # TODO support batch_size > 1
        roi = self.roi_align(features, proposals)
        roi = torch.cat(roi, dim=0)

        x = torch.flatten(roi, start_dim=1)
        x = self.box_head(x)
        cls_score = self.cls_score(x)
        cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)


        proposals_i = torch.cat([x[0] for x in proposals])
        proposals_i = box_util.apply_deltas(
            box_util.xyxy_to_cxy(proposals_i),
            bbox_pred
        )

        if self.training:
            # TODO label and sample proposals
            matched_gt_labels, matched_gt_boxes = self.label_proposals(
                box_util.cwh_to_xyxy(proposals_i), [box_util.xywh_to_xyxy(x) for x in gt_boxes], gt_labels
            )
            matched_gt_labels = self.sample_anchors(matched_gt_labels)

            losses = self.compute_loss(proposals_i, cls_score, bbox_pred, matched_gt_labels, matched_gt_boxes)

        # TODO detection delta
        detections["boxes"] = bbox_pred
        detections["labels"] = cls_score

        return detections, losses
