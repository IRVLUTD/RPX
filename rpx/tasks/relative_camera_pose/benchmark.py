import numpy as np
from ..base import BaseBenchmark
from ...utils.pose_math import quat_to_matrix, compute_relative_transform
from .metrics import compute_metrics

class RelativePoseBenchmark(BaseBenchmark):
    """
    Benchmark for Evaluating Relative Camera Pose Estimation.
    """
    def evaluate(self, model):
        results = []
        groups = {}
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            key = (sample['scene'], sample['phase'])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
            
        for key, indices in groups.items():
            for i in range(len(indices) - 1):
                idx1, idx2 = indices[i], indices[i+1]
                sample1 = self.dataset[idx1]
                sample2 = self.dataset[idx2]
                
                T1 = quat_to_matrix(sample1['pose']['position'], sample1['pose']['orientation'])
                T2 = quat_to_matrix(sample2['pose']['position'], sample2['pose']['orientation'])
                gt_rel_T = compute_relative_transform(T1, T2)
                pred_rel_T = model.predict_relative_pose(sample1, sample2)
                
                metrics = self.compute_metrics(pred_rel_T, gt_rel_T)
                metrics.update({
                    'scene': sample1['scene'],
                    'phase': sample1['phase'],
                    'frame_id_pair': (sample1['frame_id'], sample2['frame_id'])
                })
                results.append(metrics)
            
        return self._aggregate_results(results)

    def compute_metrics(self, pred_T, gt_T):
        return compute_metrics(pred_T, gt_T)

    def _aggregate_results(self, results):
        if not results:
            return {}
        phases = set(r['phase'] for r in results)
        summary = {
            'overall': {
                'mean_trans_error': np.mean([r['trans_error'] for r in results]),
                'mean_rot_error': np.mean([r['rot_error'] for r in results])
            }
        }
        for phase in phases:
            phase_results = [r for r in results if r['phase'] == phase]
            summary[phase] = {
                'mean_trans_error': np.mean([r['trans_error'] for r in phase_results]),
                'mean_rot_error': np.mean([r['rot_error'] for r in phase_results])
            }
        return summary
