import numpy as np

def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_miou(preds, gts):
    """
    Computes Mean IoU across multiple pairs.
    """
    ious = [compute_iou(p, g) for p, g in zip(preds, gts)]
    return np.mean(ious)
