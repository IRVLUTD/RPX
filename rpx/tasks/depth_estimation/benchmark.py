import numpy as np
from ..base import BaseBenchmark
from .metrics import compute_depth_metrics

class DepthEstimationBenchmark(BaseBenchmark):
    """
    Benchmark for Evaluating Metric Depth Estimation.
    """
    def evaluate(self, model):
        results = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if 'depth' not in sample:
                continue
            pred_depth = model.predict_depth(sample['rgb'])
            gt_depth = sample['depth']
            if isinstance(gt_depth, (dict, np.lib.npyio.NpzFile)):
                gt_depth = gt_depth['depth']
            metrics = self.compute_metrics(pred_depth, gt_depth)
            if metrics:
                metrics.update({
                    'scene': sample['scene'],
                    'phase': sample['phase']
                })
                results.append(metrics)
        return self._aggregate_results(results)

    def compute_metrics(self, pred, gt):
        return compute_depth_metrics(pred, gt)

    def _aggregate_results(self, results):
        if not results:
            return {}
        metrics_keys = ['abs_rel', 'rmse', 'delta1', 'delta2', 'delta3']
        summary = {}
        for key in metrics_keys:
            summary[f'mean_{key}'] = np.mean([r[key] for r in results if key in r])
        return summary
