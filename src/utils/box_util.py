import torch


def xywh_to_xyxy(boxes):
    '''
        Args:
            List([x,y,w,h])
        Returns:
            List([x1,y1,x2,y2])
    '''
    xyxy = torch.zeros(boxes.shape, device=boxes.device)
    xyxy[...,0] = boxes[...,0]
    xyxy[...,1] = boxes[...,1]
    xyxy[...,2] = boxes[...,0] + boxes[...,2]
    xyxy[...,3] = boxes[...,1] + boxes[...,3]
    return xyxy

def cwh_to_xywh(boxes):
    '''
        Args:
            List([cx,cy,w,h])
        Returns:
            List([x,y,w,h])
    '''
    xywh = torch.zeros(boxes.shape, device=boxes.device)
    xywh[...,0] = boxes[...,0] - boxes[...,2] / 2
    xywh[...,1] = boxes[...,1] - boxes[...,3] / 2
    xywh[...,2] = boxes[...,2]
    xywh[...,3] = boxes[...,3]
    return xywh

def cwh_to_xyxy(boxes):
    '''
        Args:
            List([cx,cy,w,h])
        Returns:
            List([x1,y1,x2,y2])
    '''
    xyxy = torch.zeros(boxes.shape, device=boxes.device)
    xyxy[...,0] = boxes[...,0] - boxes[...,2] / 2
    xyxy[...,1] = boxes[...,1] - boxes[...,3] / 2
    xyxy[...,2] = boxes[...,0] + boxes[...,2] / 2
    xyxy[...,3] = boxes[...,1] + boxes[...,3] / 2
    return xyxy

def xyxy_to_cxy(boxes):
    '''
        Args:
            List([x1,y1,x2,y2])
        Returns:
            List([cx,cy,w,h])
    '''
    cxy = torch.zeros(boxes.shape, device=boxes.device)
    cxy[...,0] = (boxes[...,0] + boxes[...,2]) / 2
    cxy[...,1] = (boxes[...,1] + boxes[...,3]) / 2
    cxy[...,2] = boxes[...,2] - boxes[...,0]
    cxy[...,3] = boxes[...,3] - boxes[...,1]
    return cxy

def intersection(src, target):
    '''
        return intersection (x1, y1, x2, y2) of src and target
        Args:
            src [N, 4]
            target [4]
        Returns:
            intersection [N, 4]
    '''
    i = torch.zeros(src.shape, device=src.device)
    i[...,0] = torch.max(src[...,0], target[0])
    i[...,1] = torch.max(src[...,1], target[1])
    i[...,2] = torch.min(src[...,2], target[2])
    i[...,3] = torch.min(src[...,3], target[3])
    return i

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
    xywh = torch.zeros(deltas.shape, device=anchors.device)
    xywh[...,0] = anchors[...,0] + deltas[...,0] * anchors[...,2]
    xywh[...,1] = anchors[...,1] + deltas[...,1] * anchors[...,3]
    xywh[...,2] = torch.exp(deltas[...,2]) * anchors[...,2]
    xywh[...,3] = torch.exp(deltas[...,3]) * anchors[...,3]
    return xywh
