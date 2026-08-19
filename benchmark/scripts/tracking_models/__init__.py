"""Official model adapters for the RPX D3 tracking protocol."""

from .cutie_tracker import CutieTracker
from .edgetam_tracker import EdgeTAMTracker
from .mits_tracker import MITSTracker
from .ovtr_tracker import OVTRTracker
from .sam2_plus_tracker import SAM2PlusTracker
from .sam2_tracker import SAM2Tracker
from .sam2long_tracker import SAM2LongTracker
from .xmem_tracker import XMemTracker

TRACKER_CLASSES = {
    "cutie": CutieTracker,
    "edgetam": EdgeTAMTracker,
    "mits": MITSTracker,
    "ovtr": OVTRTracker,
    "sam2": SAM2Tracker,
    "sam2-plus": SAM2PlusTracker,
    "sam2long": SAM2LongTracker,
    "xmem": XMemTracker,
}

__all__ = [
    "CutieTracker",
    "EdgeTAMTracker",
    "MITSTracker",
    "OVTRTracker",
    "SAM2LongTracker",
    "SAM2PlusTracker",
    "SAM2Tracker",
    "XMemTracker",
    "TRACKER_CLASSES",
]
