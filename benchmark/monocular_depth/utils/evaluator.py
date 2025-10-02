import numpy as np
from typing import Dict

class Evaluator:
    """Contains static methods to compute evaluation metrics for depth estimation."""
    @staticmethod
    def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
        """Computes the Root Mean Square Error."""
        return np.sqrt(np.mean((pred - gt) ** 2))

    @staticmethod
    def compute_absrel(pred: np.ndarray, gt: np.ndarray) -> float:
        """Computes the Absolute Relative Error."""
        # Add a small epsilon to avoid division by zero
        return np.mean(np.abs(pred - gt) / (gt + 1e-6))

    @staticmethod
    def evaluate(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Runs all evaluation metrics and returns a dictionary of scores."""
        return {
            "rmse": Evaluator.compute_rmse(pred, gt),
            "absrel": Evaluator.compute_absrel(pred, gt),
        }