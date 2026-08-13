"""Official model adapters for the RPX D3 tracking protocol."""

from .cutie_tracker import CutieTracker
from .edgetam_tracker import EdgeTAMTracker
from .sam2_tracker import SAM2Tracker
from .sam2long_tracker import SAM2LongTracker

TRACKER_CLASSES = {
    "cutie": CutieTracker,
    "edgetam": EdgeTAMTracker,
    "sam2": SAM2Tracker,
    "sam2long": SAM2LongTracker,
}

__all__ = [
    "CutieTracker",
    "EdgeTAMTracker",
    "SAM2LongTracker",
    "SAM2Tracker",
    "TRACKER_CLASSES",
]
