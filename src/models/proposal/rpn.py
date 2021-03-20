import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from config import Config
from torch import device, nn
from utils import box_util


class AnchorGenerator(nn.Module):
    def __init__(
        self, 
        sizes: List[int] = [32, 64, 128, 256, 512],
        aspect_ratios: List[float] = [0.5, 1.0, 2.0],
    ) -> None:
        '''
            Anchor generator

            Args:
                sizes (List[int]): anchor size in original image
                aspect_ratios (List[float]): height / width
        '''
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        return

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor
    ) -> List[torch.Tensor]:
        '''
            Generate anchors for all given features

            Args:
                images (torch.Tensor): [b, c, w, h]
                features List(torch.Tensor): [b, c, w, h]
                
            Return:
                anchors (List[torch.Tensor]): [num_anchor_type * w * h, 4]
        '''
        anchors = []

        for feature in features:
            grid_size = feature.shape[-2]
            feature_scale = images.shape[-2] / grid_size
            
            anchors_per_feature = []

            for s in self.sizes:
                for r in self.aspect_ratios:
                    # cx, cy, w, h
                    r = math.sqrt(r)
                    anchor_w = s * r
                    anchor_h = s / r

                    steps = torch.arange(0, grid_size, dtype=torch.float32, device=feature.device) * feature_scale
                    x, y = torch.meshgrid(steps, steps)
                    anchor_w = torch.ones((grid_size, grid_size), device=feature.device) * anchor_w
                    anchor_h = torch.ones((grid_size, grid_size), device=feature.device) * anchor_h
                    anchors_per_single_template = torch.stack((x, y, anchor_w, anchor_h)).permute(1,2,0).reshape(-1,4)
                    anchors_per_feature.append(anchors_per_single_template)

            anchors_per_feature = torch.cat(anchors_per_feature, dim=0)
            anchors.append(anchors_per_feature)

        return anchors

class RPNHead(nn.Module):
    '''
        rpn head for classifying object/nonobject. input is list of [b,c,w,h] features, each element corresponding to f2~f6
        output is list of classification [b, k, w, h] and bbox prediction [b, 4k, w, h]

        Args:
            in_channels : # of input feature channels
            num_anchors : # of anchors (size * aspect)
    '''
    def __init__(
        self,
        in_channels: int,
        num_anchors: int
    ):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits = []
        bbox = []
        for x in features:
            t = self.conv(x)
            t = self.relu(t)
            logits.append(self.cls_logits(t))
            bbox.append(self.bbox_pred(t))
        return logits, bbox


class RPN(nn.Module):
    ''' region proposal network '''
    def __init__(
        self,
        cfg: Config
    ):
        super(RPN, self).__init__()
        self.sizes=[32, 64, 128, 256, 512]
        self.aspect_ratios=[0.5, 1.0, 2.0]
        self.num_anchors = len(self.sizes) * len(self.aspect_ratios)
        self.rpn_head = RPNHead(in_channels=256, num_anchors=self.num_anchors)
        self.anchor_generator = AnchorGenerator(self.sizes, self.aspect_ratios)
        self.cfg = cfg

    @torch.no_grad()
    def label_anchors(
        self,
        anchors: torch.Tensor,
        gt_boxes: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Args:
                anchors [Tensor]: anchors from all feature levels. [sum(num_anchors_in_level), 4] (xyxy)
                gt_boxes (List[Tensor]): the ground-truth instances for each image. [b, num_gt_boxes, 4] (xywh)
                images for debug
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
        for gt_boxes_i in gt_boxes: # i th image in batch
            if torch.numel(gt_boxes_i) == 0:
                # all negative
                gt_labels_i = torch.ones(anchors.shape[0], device=anchors.device) * -1
                matched_gt_boxes_i = torch.zeros(anchors.shape, device=anchors.device)
            else:
                # (num_total_anchor) * (num_gt_box) match quality matrix 
                iou_matrix = box_util.get_iou(anchors, gt_boxes_i)

                matched_gt_score, matched_gt_idx = torch.max(iou_matrix, dim=1)
                matched_gt_boxes_i = gt_boxes_i[matched_gt_idx]

                # negative sample: max objectness of anchor is below threshold
                neg_idx = torch.where(matched_gt_score < neg_threshold)[0]

                # positive sample  
                pos_idx_a = torch.where(matched_gt_score > pos_threshold)[0] # iou > pos_threshold
                pos_idx_b = torch.max(iou_matrix, dim=0)[1] # maximum iou for each gt box
                pos_idx = torch.cat((pos_idx_a, pos_idx_b)).unique()
                
                gt_labels_i = torch.ones(anchors.shape[0], device=anchors.device) * -1
                gt_labels_i[neg_idx] = 0
                gt_labels_i[pos_idx] = 1

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    @torch.no_grad()
    def sample_anchors(self, gt_labels):
        for gt_label in gt_labels:
            
            positive = torch.where(gt_label == 1)[0]
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

    def compute_loss(self, anchors, pred_objectness_logits, pred_bbox_deltas, gt_labels, gt_boxes):
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

        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        print("# of pos/neg anchors: ", num_pos_anchors, num_neg_anchors)

        # box regression loss
        target_deltas = [box_util.get_deltas(anchors, box_util.xyxy_to_cxy(gt_boxes_i)) for gt_boxes_i in gt_boxes]

        target_deltas = torch.stack(target_deltas)  # (N_images, R, 4) (dx dy dw dh)

        loss_box_reg = F.smooth_l1_loss(pred_bbox_deltas[pos_mask], target_deltas[pos_mask], reduction="sum")
        
        # classification loss
        valid_mask = gt_labels >= 0
        # pred_objectness_logits = torch.cat(pred_objectness_logits, dim=1)
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_objectness_logits[valid_mask], gt_labels[valid_mask], reduction="sum")
        
        mini_batch_size = 256
        normalizer = mini_batch_size * num_images

        losses = {"loss_rpn_box":loss_box_reg / normalizer, "loss_rpn_cls":loss_cls / normalizer}

        return losses

    def get_top_proposals(
        self,
        proposals: List[torch.Tensor],
        scores: List[torch.Tensor],
        nms_threshold: float = 0.7,
        pre_nms_topk: int = 12000,
        post_nms_topk: int = 1000,
    ) -> List[List[torch.Tensor]]:

        nms_proposals = []
        # topk proposals from each feature level
        level_scores = []
        level_proposals = []
        for proposals_lvl, scores_lvl in zip(proposals, scores):
            # topk 12000
            num_proposals_lvl = min(pre_nms_topk, scores_lvl.shape[-1])

            # topk scores
            scores_lvl, idx = torch.sort(scores_lvl, dim=1, descending=True)
            topk_scores_lvl = torch.narrow(scores_lvl, 1, 0, num_proposals_lvl)
            idx = torch.narrow(idx, 1, 0, num_proposals_lvl)

            # topk proposals (per image)
            topk_proposals_lvl = torch.zeros([idx.shape[0], idx.shape[1], 4], device=idx.device)
            for i, topk_idx_image in enumerate(idx):
                topk_proposals_lvl[i] = proposals_lvl[i][topk_idx_image]

            level_proposals.append(topk_proposals_lvl)
            level_scores.append(topk_scores_lvl)

        # concat topk anchors from all features
        level_proposals = torch.cat(level_proposals, dim=1)
        level_scores = torch.cat(level_scores, dim=1)

        # post nms topk
        for scores_i, proposals_i in zip(level_scores, level_proposals):
            # anchor outside image
            valid_idx = box_util.inside_box(proposals_i, self.cfg.image_size)
            proposals_i = proposals_i[valid_idx]
            scores_i = scores_i[valid_idx]

            # nms
            keep_idx = torchvision.ops.nms(proposals_i, scores_i, nms_threshold)
            proposals_i = proposals_i[keep_idx]

            # topk 1000
            proposals_i = proposals_i[:post_nms_topk]
            nms_proposals.append(proposals_i)

        return nms_proposals

    def forward(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        gt_boxes: List[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        '''
            rpn forward

            Args:
                images (Tensor): [b, c, w, h]
                features (List[Tensor]): [b, c, w, h]
                targets (Tensor): [b, N, 4] (xywh)
                annotations (List[Dict[str, torch.Tensor]]): "boxes", "labels"
            Returns:
                proposals (List[Tensor]): list of detections [b, 4] (x1y1x2y2)
                losses (Dict[str: Tensor]): losses for model during training
                    "loss_rpn_box", "loss_rpn_cls"

        '''
        losses = {}

        # generate anchors (List[Tensor])
        anchors = self.anchor_generator(images, features)

        # rpn forward (List[Tensor], List[Tensor])
        pred_objectness_logits, pred_bbox_deltas = self.rpn_head(features) # list of [b,A,w,h], [b,4A,w,h] for features (delta)
        
        pred_objectness_logits = [ # reshape [b, A, w, h] -> [b, Awh]
            cls.permute(0, 2, 3, 1).reshape((cls.shape[0], -1)) for cls in pred_objectness_logits]
        pred_bbox_deltas = [ #  reshape [b, 4A, w, h] -> [b, Awh, 4] (delta)
            box.permute(0, 2, 3, 1).reshape((box.shape[0], -1, 4)) for box in pred_bbox_deltas]

        if self.training:
            # list of tensors for features -> single tensor
            anchors_from_all_level = torch.cat(anchors)
            cls_from_all_level = torch.cat(pred_objectness_logits, dim=1)
            bbox_from_all_level = torch.cat(pred_bbox_deltas, dim=1)

            # exclude out-of-image anchors in training time
            # for i, (lvl_anchor, lvl_cls, lvl_box) in enumerate(zip(anchors, pred_objectness_logits, pred_bbox_deltas)):
            idx_inside = box_util.inside_box(box_util.cwh_to_xyxy(anchors_from_all_level), self.cfg.image_size)
            anchors_from_all_level = anchors_from_all_level[idx_inside]
            cls_from_all_level = cls_from_all_level[:, idx_inside]
            bbox_from_all_level = bbox_from_all_level[:, idx_inside]

            # label and sample anchors [N], [N, 4] (xyxy)
            gt_labels, gt_boxes = self.label_anchors(
                box_util.cwh_to_xyxy(anchors_from_all_level),
                [box_util.xywh_to_xyxy(x) for x in gt_boxes]
            )
            gt_labels = self.sample_anchors(gt_labels)

            # loss
            losses = self.compute_loss(
                anchors_from_all_level, cls_from_all_level, bbox_from_all_level, gt_labels, gt_boxes)

        # decode bbox proposals
        proposals = [] # List(Tensor)
        for anchor_level, bbox_delta_level in zip(anchors, pred_bbox_deltas):
            proposals_level = box_util.apply_deltas(anchor_level, bbox_delta_level)
            proposals.append(box_util.cwh_to_xyxy(proposals_level))

        # apply nms, get top 1000 proposals
        pred = self.get_top_proposals(proposals, pred_objectness_logits, pre_nms_topk=12000, post_nms_topk=1000)

        return pred, losses
