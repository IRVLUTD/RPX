"""
Relative Camera Pose — Task-specific dataset.

Requires: rgb, pose
Works on: both multi-obj and single-obj scenes
Produces sequential frame pairs grouped by (scene, phase).
"""
from ...data.dataset import RPXDataset


class RelativePoseDataset(RPXDataset):
    """
    Dataset for Relative Camera Pose Estimation.

    Pre-configured modalities: rgb, pose
    Adds .pairs property for iterating sequential frame pairs.
    """

    def __init__(self, root_dir, scenes=None, phases=None, scene_type='auto', transform=None):
        super().__init__(
            root_dir=root_dir,
            scenes=scenes,
            phases=phases,
            modalities=['rgb', 'pose'],
            scene_type=scene_type,
            transform=transform,
        )
        self._pairs = None  # lazily computed

    @property
    def pairs(self):
        """
        Returns a list of (idx_a, idx_b) sequential frame pairs,
        grouped within the same (scene, phase) to avoid cross-boundary pairs.
        """
        if self._pairs is not None:
            return self._pairs

        groups = {}
        for i, s in enumerate(self.samples):
            key = (s['scene'], s['phase'])
            groups.setdefault(key, []).append(i)

        self._pairs = []
        for indices in groups.values():
            for a, b in zip(indices, indices[1:]):
                self._pairs.append((a, b))

        return self._pairs

    def get_pair(self, pair_idx):
        """Load a pair of samples by pair index."""
        a, b = self.pairs[pair_idx]
        return self[a], self[b]

    def num_pairs(self):
        return len(self.pairs)
