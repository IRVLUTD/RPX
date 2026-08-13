"""Official model adapters for the RPX D3 tracking protocol."""

from .cutie_tracker import CutieTracker
from .edgetam_tracker import EdgeTAMTracker
from .masa_tracker import MASATracker
from .motip_tracker import MOTIPTracker
from .sam2_plus_tracker import SAM2PlusTracker
from .sam2_tracker import SAM2Tracker
from .sam2long_tracker import SAM2LongTracker

TRACKER_CLASSES = {
    "cutie": CutieTracker,
    "edgetam": EdgeTAMTracker,
    "masa": MASATracker,
    "motip": MOTIPTracker,
    "sam2": SAM2Tracker,
    "sam2-plus": SAM2PlusTracker,
    "sam2long": SAM2LongTracker,
}

__all__ = [
    "CutieTracker",
    "EdgeTAMTracker",
    "MASATracker",
    "MOTIPTracker",
    "SAM2LongTracker",
    "SAM2PlusTracker",
    "SAM2Tracker",
    "TRACKER_CLASSES",
]
