from typing import List, Dict
from .models.base_model import DepthModel
from .utils.dataset_loader import DatasetLoader
from .utils.evaluator import Evaluator
from .utils.profiler import Profiler
import numpy as np

class BenchmarkManager:
    """Orchestrates the benchmarking process for a list of models."""
    def __init__(
        self,
        models: List[DepthModel],
        dataset_loader: DatasetLoader,
        evaluator: Evaluator
    ):
        self.models = models
        self.dataset = dataset_loader
        self.evaluator = evaluator
        self.results: Dict = {}

    def run_single_model(self, model: DepthModel) -> None:
        """Runs the full benchmark for a single model."""
        model_name = model.model_name
        print(f"\n{'='*20}\nRunning benchmark for: {model_name}\n{'='*20}")
        self.results[model_name] = {"scores": [], "perf": []}

        for i, (image, ground_truth, name) in enumerate(self.dataset):
            print(f"  Processing image {i+1}/{len(self.dataset)}: {name}")
            
            with Profiler() as p:
                prediction = model.predict(image)

            scores = self.evaluator.evaluate(prediction, ground_truth)
            self.results[model_name]["scores"].append(scores)
            
            perf_metrics = {
                "runtime_sec": p.runtime_sec,
                "gpu_mem_mb": p.gpu_mem_mb
            }
            self.results[model_name]["perf"].append(perf_metrics)

    def run_all(self) -> None:
        """Runs the benchmark for all models provided during initialization."""
        if not self.models:
            print("No models to benchmark.")
            return
        for model in self.models:
            self.run_single_model(model)

    def report_results(self) -> None:
        """Prints a formatted summary report of the benchmark results."""
        print(f"\n{'='*30}\n🏆 Benchmark Report 🏆\n{'='*30}")
        if not self.results:
            print("No results to report. Please run the benchmark first.")
            return

        for model_name, result in self.results.items():
            avg_scores = {
                "rmse": np.mean([s['rmse'] for s in result['scores']]),
                "absrel": np.mean([s['absrel'] for s in result['scores']])
            }
            avg_perf = {
                "runtime": np.mean([p['runtime_sec'] for p in result['perf']]),
                "memory": np.mean([p['gpu_mem_mb'] for p in result['perf']])
            }

            print(f"\n--- Model: {model_name} ---")
            print("  Accuracy:")
            print(f"    - Avg. RMSE:   {avg_scores['rmse']:.4f}")
            print(f"    - Avg. AbsRel: {avg_scores['absrel']:.4f}")
            print("  Performance:")
            print(f"    - Avg. Runtime: {avg_perf['runtime']:.4f} sec")
            print(f"    - Avg. GPU Mem: {avg_perf['memory']:.2f} MB")
        print(f"\n{'='*30}")