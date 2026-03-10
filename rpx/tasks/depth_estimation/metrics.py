import numpy as np

def compute_depth_metrics(pred, gt, mask=None):
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    valid = gt > 0
    pred = pred[valid]
    gt = gt[valid]
    if len(gt) == 0:
        return {}
    thresh = np.maximum((gt / pred), (pred / gt))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }
