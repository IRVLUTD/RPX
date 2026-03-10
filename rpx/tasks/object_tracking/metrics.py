import numpy as np

def compute_mot_metrics(pred_tracks, gt_tracks):
    pred_counts = [len(v) for v in pred_tracks.values()]
    gt_counts = [len(v) for v in gt_tracks.values()]
    count_error = np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts)))
    return {
        'count_error': count_error,
        'mot_placeholder': 0.0
    }
