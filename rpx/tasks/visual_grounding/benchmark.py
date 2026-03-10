import numpy as np
from ..base import BaseBenchmark
from .metrics import compute_miou

class VisualGroundingBenchmark(BaseBenchmark):
    """
    Benchmark for Evaluating Visual Grounding (General & Spatial awareness).
    """
    def evaluate(self, model, task_type='general'):
        results = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if 'phrases' not in sample or len(sample['phrases']) == 0:
                continue
                
            pred_masks = model.predict_grounding(sample['rgb'], sample['phrases'], task_type=task_type)
            gt_masks = sample.get('masks', []) 
            
            if not gt_masks:
                continue
                
            iou = compute_miou(pred_masks, gt_masks)
            results.append({
                'scene': sample['scene'],
                'phase': sample['phase'],
                'iou': iou
            })
            
        return self._aggregate_results(results)

    def compute_metrics(self, pred, gt):
        return {'iou': compute_miou(pred, gt)}

    def _aggregate_results(self, results):
        if not results:
            return {}
        return {'mean_iou': np.mean([r['iou'] for r in results])}
