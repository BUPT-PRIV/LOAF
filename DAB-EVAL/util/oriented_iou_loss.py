import torch
import numpy as np
from util.box_intersection_2d import oriented_box_intersection_2d
from util.min_enclosing_box import smallest_bounding_box

def box_area(boxes):
    x0,y0, x1,y1, x2,y2, x3,y3 = boxes.unbind(-1)  # [N,]
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 *\
           ((x3 - x0) ** 2 + (y3 - y0) ** 2) ** 0.5  # [N,]

def cal_iou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box1.reshape(box1.shape[0],1,4,2).repeat(1,box2.shape[0],1,1)
    corners2 = box2.reshape(1,box2.shape[0],4,2).repeat(box1.shape[0],1,1,1)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box_area(corners1.flatten(2))
    area2 = box_area(corners2.flatten(2))
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u


def cal_giou(box1:torch.Tensor, box2:torch.Tensor, enclosing_type:str="smallest"):
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    area_c =  w*h
    giou =  iou - ( area_c - u )/area_c
    return giou, iou

def enclosing_box(corners1:torch.Tensor, corners2:torch.Tensor, enclosing_type:str="smallest"):
    if enclosing_type == "aligned":
        return enclosing_box_aligned(corners1, corners2)
    elif enclosing_type == "pca":
        return enclosing_box_pca(corners1, corners2)
    elif enclosing_type == "smallest":
        return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    else:
        ValueError("Unknow type enclosing. Supported: aligned, pca, smallest")

def enclosing_box_aligned(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the smallest enclosing box (axis-aligned)

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    w = x_max - x_min       # (B, N)
    h = y_max - y_min
    return w, h

def enclosing_box_pca(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the rotated smallest enclosing box using PCA

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    B = corners1.size()[0]
    c = torch.cat([corners1, corners2], dim=2)      # (B, N, 8, 2)
    c = c - torch.mean(c, dim=2, keepdim=True)      # normalization
    c = c.view([-1, 8, 2])                          # (B*N, 8, 2)
    ct = c.transpose(1, 2)                          # (B*N, 2, 8)
    ctc = torch.bmm(ct, c)                          # (B*N, 2, 2)
    # NOTE: the build in symeig is slow!
    # _, v = ctc.symeig(eigenvectors=True)
    # v1 = v[:, 0, :].unsqueeze(1)                   
    # v2 = v[:, 1, :].unsqueeze(1)
    v1, v2 = eigenvector_22(ctc)
    v1 = v1.unsqueeze(1)                            # (B*N, 1, 2), eigen value
    v2 = v2.unsqueeze(1)
    p1 = torch.sum(c * v1, dim=-1)                  # (B*N, 8), first principle component
    p2 = torch.sum(c * v2, dim=-1)                  # (B*N, 8), second principle component
    w = p1.max(dim=-1)[0] - p1.min(dim=-1)[0]       # (B*N, ),  width of rotated enclosing box
    h = p2.max(dim=-1)[0] - p2.min(dim=-1)[0]       # (B*N, ),  height of rotated enclosing box
    return w.view([B, -1]), h.view([B, -1])

def eigenvector_22(x:torch.Tensor):
    """return eigenvector of 2x2 symmetric matrix using closed form
    
    https://math.stackexchange.com/questions/8672/eigenvalues-and-eigenvectors-of-2-times-2-matrix
    
    The calculation is done by using double precision

    Args:
        x (torch.Tensor): (..., 2, 2), symmetric, semi-definite
    
    Return:
        v1 (torch.Tensor): (..., 2)
        v2 (torch.Tensor): (..., 2)
    """
    # NOTE: must use doule precision here! with float the back-prop is very unstable
    a = x[..., 0, 0].double()
    c = x[..., 0, 1].double()
    b = x[..., 1, 1].double()                                # (..., )
    delta = torch.sqrt(a*a + 4*c*c - 2*a*b + b*b)
    v1 = (a - b - delta) / 2. /c
    v1 = torch.stack([v1, torch.ones_like(v1, dtype=torch.double, device=v1.device)], dim=-1)    # (..., 2)
    v2 = (a - b + delta) / 2. /c
    v2 = torch.stack([v2, torch.ones_like(v2, dtype=torch.double, device=v2.device)], dim=-1)    # (..., 2)
    n1 = torch.sum(v1*v1, keepdim=True, dim=-1).sqrt()
    n2 = torch.sum(v2*v2, keepdim=True, dim=-1).sqrt()
    v1 = v1 / n1
    v2 = v2 / n2
    return v1.float(), v2.float()
