"""
Object Tracking — Task-specific dataset.

Requires: rgb, mask (bboxes for GT tracks)
Best suited for: multi-object scenes (temporal sequences)
Provides sequence-level iteration.
"""
from ...data.dataset import RPXDataset


class ObjectTrackingDataset(RPXDataset):
    """
    Dataset for Object Tracking.

    Pre-configured modalities: rgb, mask
    Provides .sequences property for iterating scene-level temporal clips.

    Note: Object tracking is most meaningful on multi-object scenes
    (scene_type='multi'), though single-object sequences are supported.
    """

    def __init__(
        self,
        root_dir,
        scenes=None,
        phases=None,
        scene_type='multi',      # default: multi-obj for tracking
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
        self._sequences = None  # lazily computed

    @property
    def sequences(self):
        """
        Returns a dict mapping (scene, phase) → list of sample indices,
        representing a full temporal sequence for each scene-phase pair.
        """
        if self._sequences is not None:
            return self._sequences

        self._sequences = {}
        for i, s in enumerate(self.samples):
            key = (s['scene'], s['phase'])
            self._sequences.setdefault(key, []).append(i)

        return self._sequences

    def get_sequence(self, scene, phase):
        """
        Load and return all samples for a given (scene, phase) as an ordered list.
        """
        indices = self.sequences.get((scene, phase), [])
        return [self[i] for i in indices]

    def iter_sequences(self):
        """Iterate over all sequences, yielding (scene, phase, [samples])."""
        for (scene, phase), indices in self.sequences.items():
            yield scene, phase, [self[i] for i in indices]
