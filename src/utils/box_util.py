import torch


def xywh_to_xyxy(boxes):
    '''
        Args:
            List([x,y,w,h])
        Returns:
            List([x1,y1,x2,y2])
    '''
    orig_shape = boxes.shape
    boxes.reshape((-1,4))
    boxes[...,2] = boxes[...,0] + boxes[...,2]
    boxes[...,3] = boxes[...,1] + boxes[...,3]
    return boxes.reshape(orig_shape)

def cwh_to_xywh(boxes):
    '''
        Args:
            List([cx,cy,w,h])
        Returns:
            List([x,y,w,h])
    '''
    orig_shape = boxes.shape
    boxes.reshape((-1,4))
    boxes[..., 0] = boxes[...,0] - boxes[...,2] / 2
    boxes[..., 1] = boxes[...,1] - boxes[...,3] / 2
    return boxes.reshape(orig_shape)

def cwh_to_xyxy(boxes):
    '''
        Args:
            List([cx,cy,w,h])
        Returns:
            List([x1,y1,x2,y2])
    '''
    orig_shape = boxes.shape
    boxes = boxes.reshape((-1,4))
    x1 = boxes[...,0] - boxes[...,2] / 2
    y1 = boxes[...,1] - boxes[...,3] / 2
    x2 = boxes[...,0] + boxes[...,2] / 2
    y2 = boxes[...,1] + boxes[...,3] / 2
    return torch.stack((x1, y1, x2, y2)).permute(1,0).reshape(orig_shape)

def xyxy_to_cxy(boxes):
    '''
        Args:
            List([x1,y1,x2,y2])
        Returns:
            List([cx,cy,w,h])
    '''
    cx = (boxes[...,0] + boxes[...,2]) / 2
    cy = (boxes[...,1] + boxes[...,3]) / 2
    w = boxes[...,2] - boxes[...,0]
    h = boxes[...,3] - boxes[...,1]
    return torch.stack((cx,cy,w,h)).permute(1,0)

def intersection(boxes1, boxes2):
    '''
        return intersection (x1, y1, x2, y2) of box1 and box2
        Args:
            List([x1, y1, x2, y2])
    '''
    orig_shape = boxes1.shape
    boxes1, boxes2 = boxes1.reshape((-1,4)), boxes2.reshape((-1,4))
    x1 = torch.max(boxes1[...,0], boxes2[...,0])
    y1 = torch.max(boxes1[...,1], boxes2[...,1])
    x2 = torch.min(boxes1[...,2], boxes2[...,2])
    y2 = torch.min(boxes1[...,3], boxes2[...,3])
    return torch.stack((x1, y1, x2, y2)).permute(1,0).reshape(orig_shape)

def get_iou(anchors, targets):
    '''
        Args:
            anchors: [N, 4]
                (x1, y1, x2, y2)
            targets: [M, 4]
                (x1, y1, x2, y2)
        Returns:
            iou: List([N, M]). N*M matrix * batch
    '''
    area_anchors = (anchors[:,2] - anchors[:,0]) * (anchors[:,3] - anchors[:,1])
    area_targets = (targets[:,2] - targets[:,0]) * (targets[:,3] - targets[:,1])

    iou = torch.zeros((len(anchors), len(targets)), device=anchors.device)
    
    for m in range(len(targets)):
        inter = intersection(anchors, targets[m])

        width = (inter[:,2] - inter[:,0]).clamp(min=0)
        height = (inter[:,3] - inter[:,1]).clamp(min=0)

        area_intersection = width * height

        area_union = (area_anchors + area_targets[m].repeat(len(anchors)) - area_intersection)

        iou[:,m] = area_intersection / area_union

    return iou

def get_deltas(src_boxes, target_boxes):
    '''
        boxes [N, 4]
            (cx, cy, w, h)
    '''
    dx = (target_boxes[...,0] - src_boxes[...,0]) / src_boxes[...,2]
    dy = (target_boxes[...,1] - src_boxes[...,1]) / src_boxes[...,3]
    dw = torch.log(target_boxes[...,2] / src_boxes[...,2])
    dh = torch.log(target_boxes[...,3] / src_boxes[...,3])

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas

def apply_deltas(anchors, deltas):
    '''
        Args:
            boxes [N, 4]
                (cx, cy, w, h)
            deltas [N, 4]
        Returns:
            [N, 4]
                (cx, cy, w, h)
    '''
    x = anchors[...,0] + deltas[...,0] * anchors[...,2]
    y = anchors[...,1] + deltas[...,1] * anchors[...,3]
    w = torch.exp(deltas[...,2]) * anchors[...,2]
    h = torch.exp(deltas[...,3]) * anchors[...,3]
    return torch.stack((x, y, w, h), dim=1).permute(0, 2, 1)
