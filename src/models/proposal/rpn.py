import math

import torch
import torch.nn.functional as F
from torch import nn
from utils import box_util


class AnchorGenerator(nn.Module):
    '''
        region proposal network 
        Args:
            sizes:
            aspect_ratios: 
    '''
    def __init__(self, 
                 sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
    ):
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        return

    def forward(self, images, features):
        '''
            anchor generator for all given features
            Args:
                images [b, c, w, h]
                features List([b, c, w, h])
            Return:
                anchors: List([A*w*h, 4])
        '''
        anchors = []

        for feature in features:
            grid_size = feature.shape[-1]
            stride = images.shape[-1] / grid_size
            
            anchors_per_feature = torch.tensor([], device=feature.device)

            for s in self.sizes:
                for r in self.aspect_ratios:
                    # cx, cy, w, h
                    r = math.sqrt(r)
                    anchor_w = s * r
                    anchor_h = s / r

                    steps = torch.arange(0, grid_size, dtype=torch.float32, device=feature.device) * stride
                    x, y = torch.meshgrid(steps, steps)
                    anchor_w = torch.ones((grid_size, grid_size), device=feature.device) * anchor_w
                    anchor_h = torch.ones((grid_size, grid_size), device=feature.device) * anchor_h
                    anchors_per_single_template = torch.stack((x, y, anchor_w, anchor_h)).permute(1,2,0).reshape(-1,4)
                    anchors_per_feature = torch.cat((anchors_per_feature, anchors_per_single_template))

            anchors.append(anchors_per_feature)

        return anchors

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
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.cls_logits.weight.data.normal_(0, 0.01)
        self.cls_logits.bias.data.zero_()

        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        self.bbox_pred.weight.data.normal_(0, 0.01)
        self.bbox_pred.bias.data.zero_()

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

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
    def __init__(self, cfg):
        super(RPN, self).__init__()
        self.sizes=[32, 64, 128, 256, 512]
        self.aspect_ratios=[0.5, 1.0, 2.0]
        self.num_anchors = len(self.sizes) * len(self.aspect_ratios)
        self.head = RPNHead(in_channels=256, num_anchors=self.num_anchors)
        self.anchor_generator = AnchorGenerator(self.sizes, self.aspect_ratios)
        self.writer = cfg

    def label_anchors(self, anchors, gt_boxes, images):
        '''
            Args:
                anchors (list[(b, A*w*h, 4)]): anchors for each feature map. (cx, cy, w, h)
                gt_boxes [b, N, 4]: the ground-truth instances for each image. (x, y, w, h)
                images for debug
            Returns:
                list[Tensor]:
                    gt_labels
                    List of #img tensors. i-th element is a vector of labels whose length is
                    the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                    Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                    class; 1 = positive class.
                list[Tensor]:
                    matched_gt_boxes (xyxy)
                    i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                    anchor. Values are undefined for those anchors not labeled as 1.
        '''
        
        pos_threshold = 0.7
        neg_threshold = 0.3

        anchors = torch.cat(tuple(anchors))
        anchors = box_util.cwh_to_xyxy(anchors)


        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes: # i th image in batch
            if torch.numel(gt_boxes_i) == 0:
                # all negative
                gt_labels_i = torch.ones(anchors.shape[0], device=anchors.device) * -1
                matched_gt_boxes_i = torch.zeros(anchors.shape, device=anchors.device)
            else:
                gt_boxes_i = box_util.xywh_to_xyxy(gt_boxes_i)

                # N (anchor) * M (gt box) matrix 
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
                gt_labels_i[pos_idx] = 1
                gt_labels_i[neg_idx] = 0


            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def sample_anchors(self, anchors, gt_labels, images):
        # filter out-of-image anchors
        anchors = torch.cat(anchors)
        idx_inside = (
            (anchors[..., 0] >= 0)
            & (anchors[..., 1] >= 0)
            & (anchors[..., 2] < images.shape[-1])
            & (anchors[..., 3] < images.shape[-1])
        )
        for gt_label in gt_labels:
            gt_label[~idx_inside] = -1
            
            positive = torch.where((gt_label != -1) & (gt_label != 0))[0]
            negative = torch.where(gt_label == 0)[0]

            num_pos = 128
            # protect against not enough positive examples
            num_pos = min(positive.numel(), 128)
            num_neg = 256 - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx = positive[perm1]
            neg_idx = negative[perm2]

            gt_label.fill_(-1)
            gt_label.scatter_(0, pos_idx, 1)
            gt_label.scatter_(0, neg_idx, 0)
            
        return gt_labels

    def smooth_l1(self, input: torch.Tensor):
        input = torch.where(input <= -1, -input - 0.5, input)
        input = torch.where(input >= 1, input - 0.5, input)
        input = torch.where((input > -1) & (input < 1), input * input * 0.5, input)

        return input

    def cross_entropy(input, target):
        torch.softmax(input)
        return

    def compute_loss(self, anchors, pred_objectness_logits, pred_bbox_deltas, gt_labels, gt_boxes, writer):
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
        print("# pos and neg anchors", num_pos_anchors, num_neg_anchors)

        # box regression loss
        anchors = torch.cat(anchors)  # (R, 4)
        target_deltas = [box_util.get_deltas(anchors, box_util.xyxy_to_cxy(k)) for k in gt_boxes]
        target_deltas = torch.stack(target_deltas)  # (N_images, R, 4) (dx dy dw dh)

        pred_bbox_deltas = torch.cat(pred_bbox_deltas, dim=1) # (dx dy dw dh)

        # loss_box_reg = self.smooth_l1(target_deltas - pred_bbox_deltas).sum()
        loss_box_reg = F.smooth_l1_loss(pred_bbox_deltas[pos_mask], target_deltas[pos_mask], reduction="sum")
        
        # classification loss
        valid_mask = gt_labels >= 0
        pred_objectness_logits = torch.cat(pred_objectness_logits, dim=1)
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_objectness_logits[valid_mask], gt_labels[valid_mask], reduction="sum")
        
        mini_batch_size = 256
        normalizer = mini_batch_size * num_images

        losses = {"loss_rpn_box":loss_box_reg / normalizer, "loss_rpn_cls":loss_cls / normalizer}

        return losses

    def forward(self, images, features, gt_boxes):
        '''
            Args:
                images [b, c, w, h] : original images
                features List([b, c, w, h]): from backbone
                targets: GT labels [b, N, 4] (x y w h)
            Returns:
                detections: list of detections
                    bbox : List([b, 4*k, feat_x, feat_y])
                    objectness : List([b, 2*k, feat_x, feat_y])
                    k = # of anchors (sizes * aspect_ratios)
                    k : (obj, non-obj, cx, cy, w, h)
                losses:
                    losses for model during training
                    regression vector: (tx, ty, tw, th). init value=(0,0,0,0)
                    GT vector: (x/r-dx_pred, y/r-dy_pred, log(w/r/w_pred), log(w/r/h_pred))
                        r : downsample ratio
        '''

        proposals = []
        losses = []

        # generate anchors
        anchors = self.anchor_generator(images, features) # List([A*w*h, 4]) (cx, cy, w, h)

        # rpn forward
        pred_objectness_logits, pred_bbox_deltas = self.head(features)
        
        pred_objectness_logits = [ # List([b, A, w, h]) -> List([b, A*w*h])
            cls.permute(0, 2, 3, 1).reshape((cls.shape[0], -1)) for cls in pred_objectness_logits]
        pred_bbox_deltas = [ # List([b, 4*A, w, h]) -> List([b, 4*A*w*h, 4*A]) (dx dy dw dh)
            box.permute(0, 2, 3, 1).reshape((box.shape[0], -1, 4)) for box in pred_bbox_deltas]

        
        if self.training:
            # label and sample anchors ([N], [N, 4]) (x1 y1 x2 y2)
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_boxes, images)
            gt_labels = self.sample_anchors(anchors, gt_labels, images)

            # loss
            losses = self.compute_loss(
                anchors, pred_objectness_logits, pred_bbox_deltas, gt_labels, gt_boxes, self.writer)

        # decode bbox proposals
        proposals = [box_util.apply_deltas(lvl_anchors.unsqueeze(0), deltas) for lvl_anchors, deltas in zip(anchors, pred_bbox_deltas)]
        proposals = torch.cat(proposals, dim=1)
        proposals = box_util.cwh_to_xyxy(proposals)



        # '''
        pred_objectness_logits = torch.cat(pred_objectness_logits, dim=1)
        # pos_mask = pred_objectness_logits >= 0.7
        pos_mask = pred_objectness_logits.topk(10)[1]
        proposals = proposals[0][pos_mask]
        # pos_original_anchor = torch.cat(anchors)[pos_mask]
        # '''

        # TODO NMS

        return proposals, losses
