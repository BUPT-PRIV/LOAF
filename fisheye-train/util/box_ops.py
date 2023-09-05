# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, os
import math
from util.oriented_iou_loss import cal_giou, cal_iou
from torchvision.ops.boxes import box_area

def box_cxcywh_to_polar(cxcy, img_width=512.0):
    x, y, w, h = cxcy.unbind(-1)
    x = x-img_width
    y = y-img_width
    R = torch.pow(torch.pow(x,2)+torch.pow(y,2), 0.5)

    x_abs = torch.abs(x)
    y_abs = torch.abs(y)
    angle = torch.atan(y_abs/x_abs)*180/math.pi
    if x>0 and y>0:
        angle = angle
    elif x>0 and y<0:
        angle = 360-angle
    elif x<0 and y<0:
        angle  = angle+180
    elif x<0 and y>0:
        angle = 180-angle

    b = [R, angle, w, h]
    return torch.stack(b, dim=-1)

def box_polar_to_cxcywh(polar, width):
    R, angle, w, h = polar.unbind(-1)
    cosine = torch.cos(angle*math.pi/180)
    sine = torch.sin(angle*math.pi/180)
    x = cosine*R
    y = sine*R

    xyxy = [x+width/2,y+width/2,w,h]

    return torch.stack(xyxy, dim=-1) # N, 4

def box_polar_to_xyxy(polar, angle_norm=True):
    R, angle, w, h = polar.unbind(-1)
    if angle_norm:
        angle = angle*360
    cosine = torch.cos(angle*math.pi/180)
    sine = torch.sin(angle*math.pi/180)
    x = cosine*R
    y = sine*R

    left_x = x+w/2*sine
    left_y = y-w/2*cosine
    right_x = x-w/2*sine
    right_y = y+w/2*cosine

    gap_x = h/2*cosine
    gap_y = h/2*sine

    xyxy = [right_x+gap_x, right_y+gap_y,
            left_x+gap_x, left_y+gap_y,
            left_x-gap_x, left_y-gap_y,
            right_x-gap_x, right_y-gap_y]

    return torch.stack(xyxy, dim=-1) # N, 8

def rbox_cxcywh_to_xyxy(cxcy, offset=0.5):
    x_c, y_c, w, h = cxcy.unbind(-1)
    x_c = x_c-offset
    y_c = y_c-offset
    R = torch.pow(torch.pow(x_c,2)+torch.pow(y_c,2), 0.5)
    R[R==0]=1e-4
        
    cosine = x_c/R
    sine = y_c/R

    left_x = x_c+w/2*sine
    left_y = y_c-w/2*cosine
    right_x = x_c-w/2*sine
    right_y = y_c+w/2*cosine

    gap_x = h/2*cosine
    gap_y = h/2*sine

    xyxy = [right_x+gap_x, right_y+gap_y,
            left_x+gap_x, left_y+gap_y,
            left_x-gap_x, left_y-gap_y,
            right_x-gap_x, right_y-gap_y]

    return torch.stack(xyxy, dim=-1)+offset # N, 8


def rotated_box_iou(boxes1, boxes2):
    iou, corners1, corners2, u = cal_iou(boxes1, boxes2)
    
    return iou

def generalized_rotated_box_iou(boxes1, boxes2):
    giou, iou = cal_giou(boxes1, boxes2)
#     if iou.size(0)==iou.size(1):
#         print(torch.diag(iou))
    
    return giou


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # import ipdb; ipdb.set_trace()
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    # except:
    #     import ipdb; ipdb.set_trace()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)
    import ipdb; ipdb.set_trace()
