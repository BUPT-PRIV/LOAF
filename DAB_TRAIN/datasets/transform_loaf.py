# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.misc import interpolate

from util import box_ops

def rotate(image, target, angle):
    rotated_image = F.rotate(image, 360-angle)

    target = target.copy()
    if "boxes" in target:
        rotated_boxes = target["boxes"]
        for ii in range(rotated_boxes.size(0)):
            rotated_boxes[ii,1] = (rotated_boxes[ii,1]+angle)%360    
        target["boxes"] = rotated_boxes

    return rotated_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    target = target.copy()
    if "boxes" in target:
        rotated_boxes = target["boxes"]
        for ii in range(rotated_boxes.size(0)):
            rotated_boxes[ii, 1] = (540-rotated_boxes[ii, 1])%360
        target["boxes"] = rotated_boxes

    return flipped_image, target


def resize(image, target, size):

    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, 1, ratio_width, ratio_width])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_width)
        target["area"] = scaled_area

    h, w = rescaled_image.size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomRotate(object):
    def __call__(self, img, target):
        angle = random.randrange(0, 180, 1)
        return rotate(img, target, angle)

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size)


class AdjustContrast:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _contrast_factor = ((random.random() + 1.0) / 2.0) * self.contrast_factor
        img = F.adjust_contrast(img, _contrast_factor)
        return img, target

class AdjustBrightness:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _brightness_factor = ((random.random() + 1.0) / 2.0) * self.brightness_factor
        img = F.adjust_brightness(img, _brightness_factor)
        return img, target

    
class Polar2CxCyHW(object):
    def __call__(self, img, target):
        h, w = img.size
        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = box_ops.box_polar_to_cxcywh(boxes, w)
            target["boxes"] = scaled_boxes
        return img, target
    
import copy
class RotatePolar2CxCyHW(object):
    def __call__(self, img, target):
        angle = random.randrange(0, 180, 1)
        img_equ, target_equ = rotate(img, copy.deepcopy(target), angle)
        h, w = img.size
        
        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = box_ops.box_polar_to_cxcywh(boxes, w)
            target["boxes"] = scaled_boxes
            
        target_equ = target_equ.copy()
        if "boxes" in target_equ:
            boxes = target_equ["boxes"]
            scaled_boxes = box_ops.box_polar_to_cxcywh(boxes, w)
            target_equ["boxes"] = scaled_boxes

        return img, target, img_equ, target_equ, angle
    
    
class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomSelectMulti(object):
    """
    Randomly selects between transforms1 and transforms2,
    """
    def __init__(self, transformslist, p=-1):
        self.transformslist = transformslist
        self.p = p
        assert p == -1

    def __call__(self, img, target):
        if self.p == -1:
            return random.choice(self.transformslist)(img, target)

            
class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, w, w, w], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target
    
class RotateToTensor(object):
    def __call__(self, img, target, img_equ, target_equ, angle):
        return F.to_tensor(img), target, F.to_tensor(img_equ), target_equ, angle

class RotateNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, img_equ, target_equ, angle):
        image = F.normalize(image, mean=self.mean, std=self.std)
        img_equ = F.normalize(img_equ, mean=self.mean, std=self.std)
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, w, w, w], dtype=torch.float32)
            target["boxes"] = boxes
        target_equ = target_equ.copy()
        h, w = img_equ.shape[-2:]
        if "boxes" in target_equ:
            boxes = target_equ["boxes"]
            boxes = boxes / torch.tensor([w, w, w, w], dtype=torch.float32)
            target_equ["boxes"] = boxes
        return image, target, img_equ, target_equ, angle


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class RotateCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, img_equ, target_equ, angle):
        for t in self.transforms:
            image, target, img_equ, target_equ, angle = t(image, target, img_equ, target_equ, angle)
        return image, target, img_equ, target_equ, angle

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string