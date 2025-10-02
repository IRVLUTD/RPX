# models/base_model.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class DepthModel(ABC):
    """
    Abstract Base Class for a monocular depth estimation model.
    It defines the standard interface for loading and running inference.
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model()
        self.model.to(self.device)

    @abstractmethod
    def load_model(self) -> Any:
        """Loads the model from a checkpoint or hub and returns it."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocesses a single image to the model's required format."""
        pass

    @abstractmethod
    def inference(self, preprocessed_data: Any) -> Any:
        """Runs the model on the preprocessed data."""
        pass

    @abstractmethod
    def postprocess(self, model_output: Any, original_size: tuple) -> np.ndarray:
        """Converts the model's raw output to a NumPy depth map."""
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        The main public method to get a depth map from an image.
        This method orchestrates the full pipeline.
        """
        original_size = (image.shape[0], image.shape[1])
        preprocessed_data = self.preprocess(image)
        model_output = self.inference(preprocessed_data)
        depth_map = self.postprocess(model_output, original_size)
        return depth_map