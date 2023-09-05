# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import torchvision
import datasets.transform_loaf as T
from util.box_ops import box_cxcywh_to_polar, box_polar_to_xyxy, box_polar_to_cxcywh
import random


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        #while True:
        #    boxes = [1 for obj in target]
        #    if len(boxes)!=0:
        #        break
        #    else:
        #        idx = random.randint(0, len(self.ids)-1)
        #        img, target = super(CocoDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #    img, target, image_equ, target_equ, angle = self._transforms[1](img, target)
        #    img, target, image_equ, target_equ, angle = self._transforms[2](img, target, image_equ, target_equ, angle)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        boxes = [torch.tensor(obj["rotated_box"][:-1]) for obj in anno]
        try:
            boxes = torch.stack(boxes, dim=0)
        except:
            return None ,None
            print(anno)
            # print(boxes)

        classes = [0 for obj in anno]
#         classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    #rotatenormalize = T.RotateCompose([
    #    T.RotateToTensor(),
    #    T.RotateNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomRotate(),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            T.RandomSelectMulti([
                T.AdjustBrightness(2),
                T.AdjustContrast(2),
            ]),   
        ]), 
    #    T.RotatePolar2CxCyHW(),
    #    rotatenormalize,
        ]

    if image_set in ['val', 'val-seen', 'val-unseen', 'test', 'test-seen', 'test-unseen']:
        return T.Compose([
#             T.RandomResize([800], max_size=1333),
#             T.Polar2CxCyHW(),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images/resolution_1k" / "train", root / "annotations/resolution_1k" / f'{mode}_train.json'),
        "val": (root / "images/resolution_1k" / "val", root / "annotations/resolution_1k" / f'{mode}_val.json'),
        "val-seen": (root / "images/resolution_1k" / "val", root / "annotations/resolution_1k" / f'{mode}_val-seen.json'),
        "val-unseen": (root / "images/resolution_1k" / "val", root / "annotations/resolution_1k" / f'{mode}_val-unseen.json'),
        "test": (root / "images/resolution_1k" / "test", root / "annotations/resolution_1k" / f'{mode}_test.json'),
        "test-seen": (root / "images/resolution_1k" / "test", root / "annotations/resolution_1k" / f'{mode}_test-seen.json'),
        "test-unseen": (root / "images/resolution_1k" / "test", root / "annotations/resolution_1k" / f'{mode}_test-unseen.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                           aux_target_hacks=None)
    return dataset
