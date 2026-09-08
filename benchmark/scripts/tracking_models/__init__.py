"""Official model adapters for the RPX D3 tracking protocol."""

from .cutie_tracker import CutieTracker
from .dam4sam_tracker import DAM4SAMTracker
from .edgetam_tracker import EdgeTAMTracker
from .grounded_sam2_tracker import GroundedSAM2Tracker
from .mits_tracker import MITSTracker
from .ovtr_tracker import OVTRTracker
from .sam2_plus_tracker import SAM2PlusTracker
from .sam2_tracker import SAM2Tracker
from .sam2long_tracker import SAM2LongTracker
from .sam3_1_tracker import SAM31Tracker
from .xmem_tracker import XMemTracker

TRACKER_CLASSES = {
    "cutie": CutieTracker,
    "dam4sam": DAM4SAMTracker,
    "edgetam": EdgeTAMTracker,
    "mits": MITSTracker,
    "ovtr": OVTRTracker,
    "sam2": SAM2Tracker,
    "sam2-plus": SAM2PlusTracker,
    "sam2long": SAM2LongTracker,
    "xmem": XMemTracker,
    "grounded-sam2": GroundedSAM2Tracker,
    "sam3.1": SAM31Tracker,
}

__all__ = [
    "CutieTracker",
    "DAM4SAMTracker",
    "EdgeTAMTracker",
    "MITSTracker",
    "OVTRTracker",
    "SAM2LongTracker",
    "SAM2PlusTracker",
    "SAM2Tracker",
    "XMemTracker",
    "GroundedSAM2Tracker",
    "SAM31Tracker",
    "TRACKER_CLASSES",
]
