"""
Visual Grounding — Task-specific dataset.

Requires: rgb, mask (which includes bboxes + phrases from verified annotations)
Works on: both multi-obj and single-obj scenes
Supports: general grounding and spatial-awareness grounding modes.
"""
from ...data.dataset import RPXDataset


class VisualGroundingDataset(RPXDataset):
    """
    Dataset for General and Spatial Visual Grounding.

    Pre-configured modalities: rgb, mask
    Samples with no associated phrases are automatically skipped.

    Args:
        grounding_type: 'general' | 'spatial' | 'both' (default: 'both')
            Placeholder for future filtering by grounding query type.
    """

    def __init__(
        self,
        root_dir,
        scenes=None,
        phases=None,
        scene_type='auto',
        grounding_type='both',
        transform=None,
    ):
        super().__init__(
            root_dir=root_dir,
            scenes=scenes,
            phases=phases,
            modalities=['rgb', 'mask'],
            scene_type=scene_type,
            transform=transform,
        )
        self.grounding_type = grounding_type
        # Filter to only samples that have annotation files (npz)
        self.samples = [
            s for s in self.samples
            if s['mask_path'] is not None and s['mask_path'].suffix == '.npz'
        ]

    def get_phrases(self, idx):
        """Return the text phrases for sample idx (without loading full sample)."""
        import numpy as np
        info = self.samples[idx]
        if info['mask_path'] and info['mask_path'].suffix == '.npz':
            d = np.load(info['mask_path'], allow_pickle=True)
            return list(d.get('phrases', []))
        return []
