from ..base import BaseBenchmark
from .metrics import compute_mot_metrics

class ObjectTrackingBenchmark(BaseBenchmark):
    """
    Benchmark for Evaluating Object Tracking.
    """
    def evaluate(self, model):
        results = []
        sequences = {}
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            key = (sample['scene'], sample['phase'])
            if key not in sequences:
                sequences[key] = []
            sequences[key].append(sample)
        for key, sequence in sequences.items():
            pred_tracks = model.track_objects(sequence)
            gt_tracks = self._get_gt_tracks(sequence)
            metrics = self.compute_metrics(pred_tracks, gt_tracks)
            results.append(metrics)
        return self._aggregate_results(results)

    def compute_metrics(self, pred, gt):
        return compute_mot_metrics(pred, gt)

    def _get_gt_tracks(self, sequence):
        gt_tracks = {}
        for sample in sequence:
            if 'bboxes' in sample:
                gt_tracks[sample['frame_id']] = {i: b for i, b in enumerate(sample['bboxes'])}
        return gt_tracks

    def _aggregate_results(self, results):
        if not results:
            return {}
        return {'mean_count_error': sum(r['count_error'] for r in results) / len(results)}
