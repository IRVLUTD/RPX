from abc import ABC, abstractmethod
from typing import Any, Optional

from ..utils.model_profiler import profile_model, ModelComplexityProfile
from ..utils.deployment_score import compute_deployment_score, DeploymentScore


class BaseBenchmark(ABC):
    """
    Abstract base class for all RPX benchmark tasks.

    Subclasses implement `evaluate()` and `compute_metrics()`.
    Model profiling and deployment score computation are available
    automatically via `run()`.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._complexity_profile: Optional[ModelComplexityProfile] = None
        self._deployment_score: Optional[DeploymentScore] = None

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def evaluate(self, model) -> dict:
        """Run benchmarking and return task results dict."""
        pass

    @abstractmethod
    def compute_metrics(self, predictions, ground_truth) -> dict:
        """Compute task-specific metrics for a single sample."""
        pass

    def _get_dummy_input(self, model):
        """
        Returns a dummy input for model profiling.
        Override in subclasses to provide task-appropriate inputs.
        Default: tries to build a (1, 3, 480, 640) RGB tensor.
        """
        try:
            import torch
            return torch.zeros(1, 3, 480, 640)
        except ImportError:
            return None

    # ── Full evaluation pipeline ─────────────────────────────────────────────

    def run(
        self,
        model,
        profile: bool = True,
        warmup_runs: int = 5,
        timing_runs: int = 50,
        deployment_targets: Optional[dict] = None,
        deployment_weights: Optional[dict] = None,
    ) -> dict:
        """
        Full evaluation pipeline:
          1. Profile model complexity (params, FLOPs, MACs, latency, …)
          2. Run task-specific benchmark
          3. Compute Deployment Readiness Score (DRS)

        Args:
            model: The model to evaluate (should implement task-specific interface).
            profile: If True, run model complexity profiling before evaluation.
            warmup_runs: Number of warmup passes for latency measurement.
            timing_runs: Number of timed passes for latency measurement.
            deployment_targets: Override default DRS reference thresholds.
            deployment_weights: Override default DRS sub-score weights.

        Returns:
            dict with keys:
              - 'task_results': output of evaluate()
              - 'complexity_profile': ModelComplexityProfile as dict
              - 'deployment_score': DeploymentScore as dict
        """
        # ── Step 1: Profile complexity ────────────────────────────────────────
        if profile:
            dummy = self._get_dummy_input(model)
            if dummy is not None:
                self._complexity_profile = profile_model(
                    model,
                    dummy_input=dummy,
                    warmup_runs=warmup_runs,
                    timing_runs=timing_runs,
                )
                print(self._complexity_profile.summary())

        # ── Step 2: Evaluate task ─────────────────────────────────────────────
        task_results = self.evaluate(model)

        # ── Step 3: Deployment score ──────────────────────────────────────────
        self._deployment_score = compute_deployment_score(
            task_results=task_results,
            complexity_profile=self._complexity_profile,
            targets=deployment_targets,
            weights=deployment_weights,
        )
        print(self._deployment_score.summary())

        return {
            'task_results': task_results,
            'complexity_profile': (
                self._complexity_profile.to_dict()
                if self._complexity_profile else {}
            ),
            'deployment_score': (
                self._deployment_score.to_dict()
                if self._deployment_score else {}
            ),
        }
