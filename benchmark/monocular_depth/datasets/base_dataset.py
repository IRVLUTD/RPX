from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Tuple, Any

class BenchmarkDataset(Dataset, ABC):
    """
    Abstract Base Class for a benchmark dataset.
    It defines the standard interface for any dataset to be "plugged in"
    to the benchmark manager.
    """

    @abstractmethod
    def __init__(self, path: str, **kwargs):
        """
        Loads and prepares the dataset.
        Args:
            path (str): The root path to the dataset files.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Fetches a single data sample.
        Args:
            index (int): The index of the item.
        Returns:
            A tuple containing (image, ground_truth_depth).
            The image should be in a format like a PIL Image or NumPy array.
            The ground_truth_depth should be a NumPy array.
        """
        pass
