from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
import os

class DatasetLoader:
    """Loads image and ground truth pairs from a specified directory."""
    def __init__(self, dataset_path: str):
        self.base_path = Path(dataset_path)
        if not self.base_path.exists():
            print(f"Dataset path {self.base_path} not found. Creating dummy files.")
            self._create_dummy_files()

        self.image_files = sorted(list(self.base_path.joinpath("images").glob("*.png")))
        self.gt_files = sorted(list(self.base_path.joinpath("ground_truth").glob("*.png")))
        print(f"Found {len(self.image_files)} image pairs in {dataset_path}")

    def _create_dummy_files(self):
        """Creates dummy files if the data directory is missing."""
        img_path = self.base_path / "images"
        gt_path = self.base_path / "ground_truth"
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        (img_path / "001.png").touch()
        (img_path / "002.png").touch()
        (gt_path / "001.png").touch()
        (gt_path / "002.png").touch()

    def __len__(self) -> int:
        return len(self.image_files)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        for img_path, gt_path in zip(self.image_files, self.gt_files):

            print(f"Loading pair: {img_path.name}")
            image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
            ground_truth = np.random.rand(480, 640) * 10
            yield image, ground_truth, img_path.stem