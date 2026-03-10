from ..base import BaseBenchmark
from .metrics import compute_nvs_metrics

class NovelViewSynthesisBenchmark(BaseBenchmark):
    """
    Benchmark for Evaluating Novel View Synthesis.
    """
    def evaluate(self, model):
        results = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            pred_img = model.synthesize_view(sample['scene'], sample['phase'], sample['pose'])
            gt_img = sample['rgb']
            metrics = self.compute_metrics(pred_img, gt_img)
            results.append(metrics)
        return self._aggregate_results(results)

    def compute_metrics(self, pred, gt):
        return compute_nvs_metrics(pred, gt)

    def _aggregate_results(self, results):
        if not results:
            return {}
        return {
            'mean_psnr': sum(r['psnr'] for r in results) / len(results),
            'mean_mse': sum(r['mse'] for r in results) / len(results)
        }
