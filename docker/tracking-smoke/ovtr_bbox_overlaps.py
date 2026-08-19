"""Torch-only replacement for the one mmdetection operator used by OVTR."""

import torch


def bbox_overlaps(boxes1, boxes2, mode="iou"):
    if mode != "iou":
        raise ValueError("OVTR only uses IoU overlap mode")
    left_top = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    right_bottom = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0).prod(dim=-1)
    union = area1[..., :, None] + area2[..., None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)
