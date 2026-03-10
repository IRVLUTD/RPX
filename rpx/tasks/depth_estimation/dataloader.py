"""
Metric Depth Estimation — Task-specific dataset.

Requires: rgb, depth
Works on: both multi-obj and single-obj scenes
"""
from ...data.dataset import RPXDataset


class DepthEstimationDataset(RPXDataset):
    """
    Dataset for Metric Depth Estimation.

    Pre-configured modalities: rgb, depth
    Samples without a depth file are automatically excluded.

    Args:
        depth_scale: Scale factor to convert raw depth values to metres.
                     Common values: 1000.0 (mm → m), 1.0 (already metres).
    """

    def __init__(
        self,
        root_dir,
        scenes=None,
        phases=None,
        scene_type='auto',
        depth_scale=1000.0,
        transform=None,
    ):
        super().__init__(
            root_dir=root_dir,
            scenes=scenes,
            phases=phases,
            modalities=['rgb', 'depth'],
            scene_type=scene_type,
            transform=transform,
        )
        self.depth_scale = depth_scale
        # Only keep samples that actually have a depth file
        self.samples = [s for s in self.samples if s['depth_path'] is not None]

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # Apply depth scale if a depth map was loaded
        if 'depth' in data and data['depth'] is not None:
            import numpy as np
            raw = data['depth']
            # Handle npz: extract the first array key
            if hasattr(raw, 'files'):
                key = raw.files[0]
                raw = raw[key]
            data['depth'] = raw.astype('float32') / self.depth_scale
        return data
