"""
Novel View Synthesis — Task-specific dataset.

Requires: rgb, pose, depth (depth optional for NeRF-style methods)
Works on: both multi-obj and single-obj scenes
Supports train/test split within each scene.
"""
from ...data.dataset import RPXDataset


class NovelViewSynthesisDataset(RPXDataset):
    """
    Dataset for Novel View Synthesis.

    Pre-configured modalities: rgb, pose, depth
    Provides train/test split helpers for hold-out view evaluation.

    Args:
        test_every: Holds out every Nth frame as a test view (default: 8).
                    Set to None to disable splitting.
    """

    def __init__(
        self,
        root_dir,
        scenes=None,
        phases=None,
        scene_type='auto',
        test_every=8,
        split='train',       # 'train' | 'test' | 'all'
        transform=None,
    ):
        super().__init__(
            root_dir=root_dir,
            scenes=scenes,
            phases=phases,
            modalities=['rgb', 'pose', 'depth'],
            scene_type=scene_type,
            transform=transform,
        )
        self.test_every = test_every
        self.split = split

        if test_every is not None and split != 'all':
            self.samples = self._apply_split(self.samples, split, test_every)

    @staticmethod
    def _apply_split(samples, split, test_every):
        """
        Hold out every Nth frame per (scene, phase) as the test set.
        """
        # Group by (scene, phase) to split within each sequence independently
        from collections import defaultdict
        groups = defaultdict(list)
        for i, s in enumerate(samples):
            groups[(s['scene'], s['phase'])].append(i)

        selected = []
        for indices in groups.values():
            for local_idx, global_idx in enumerate(indices):
                is_test = (local_idx % test_every == 0)
                if (split == 'test' and is_test) or (split == 'train' and not is_test):
                    selected.append(global_idx)

        return [samples[i] for i in sorted(selected)]

    def get_scene_views(self, scene, phase):
        """
        Returns all samples for a given (scene, phase) — useful for
        providing context views to a NeRF model before synthesising.
        """
        return [s for s in self.samples if s['scene'] == scene and s['phase'] == phase]
