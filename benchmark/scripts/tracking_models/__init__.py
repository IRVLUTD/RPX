"""Official model adapters for the RPX D3 tracking protocol."""

from .cutie_tracker import CutieTracker
from .edgetam_tracker import EdgeTAMTracker
from .sam2_tracker import SAM2Tracker

TRACKER_CLASSES = {
    "cutie": CutieTracker,
    "edgetam": EdgeTAMTracker,
    "sam2": SAM2Tracker,
}

__all__ = ["CutieTracker", "EdgeTAMTracker", "SAM2Tracker", "TRACKER_CLASSES"]
