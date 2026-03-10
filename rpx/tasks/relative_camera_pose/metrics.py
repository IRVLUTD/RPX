import numpy as np
from ...utils.pose_math import compute_rpe

def compute_metrics(pred_T, gt_T):
    trans_error, rot_error = compute_rpe(pred_T, gt_T)
    return {
        'trans_error': trans_error,
        'rot_error': rot_error
    }
