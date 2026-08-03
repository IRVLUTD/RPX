"""Official model adapters for the RPX D3 tracking protocol."""

from .edgetam_tracker import EdgeTAMTracker
from .sam2_tracker import SAM2Tracker

TRACKER_CLASSES = {
    "edgetam": EdgeTAMTracker,
    "sam2": SAM2Tracker,
}

__all__ = ["EdgeTAMTracker", "SAM2Tracker", "TRACKER_CLASSES"]
