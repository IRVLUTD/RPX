#----------------------------------------------------------------------------------------------------
# Please check the licenses of the respective works utilized here before using this script.
#----------------------------------------------------------------------------------------------------
import logging
import matplotlib
import matplotlib.colors as mcolors
from absl import logging as absl_logging

# Configure Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sam2_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress absl logging verbosity
absl_logging.set_verbosity(absl_logging.ERROR)

# Set interactive Matplotlib backend
try:
    matplotlib.use('TkAgg')
except ImportError:
    logger.warning("TkAgg backend not available, falling back to default interactive backend")

# Supervision-inspired 25-color palette (RGB, 0-255 range)
SUPERVISION_INSPIRED_COLORS = [
    (237, 27, 27),    # Vivid Red (#ED1B1B)
    (27, 237, 27),    # Bright Green (#1BED1B)
    (27, 27, 237),    # Deep Blue (#1B1BED)
    (255, 159, 26),   # Tangerine (#FF9F1A)
    (186, 109, 211),  # Lavender Purple (#BA6DD3)
    (255, 209, 102),  # Golden Yellow (#FFD166)
    (76, 201, 240),   # Sky Blue (#4CC9F0)
    (237, 76, 103),   # Rose Pink (#ED4C67)
    (88, 177, 159),   # Teal (#58B19F)
    (255, 130, 171),  # Bubblegum Pink (#FF82AB)
    (136, 136, 255),  # Periwinkle (#8888FF)
    (180, 180, 180),  # Neutral Gray (#B4B4B4)
    (255, 94, 87),    # Salmon (#FF5E57)
    (0, 171, 85),     # Forest Green (#00AB55)
    (90, 200, 250),   # Light Cyan (#5AC8FA)
    (255, 185, 0),    # Amber (#FFB900)
    (199, 146, 234),  # Lilac (#C792EA)
    (255, 245, 105),  # Lemon (#FFF569)
    (0, 128, 255),    # Electric Blue (#0080FF)
    (233, 30, 99),    # Magenta (#E91E63)
    (72, 191, 145),   # Mint (#48B191)
    (255, 111, 145),  # Flamingo (#FF6F91)
    (121, 134, 203),  # Slate Blue (#7986CB)
    (204, 204, 0),    # Olive (#CCCC00)
    (158, 158, 158),  # Medium Gray (#9E9E9E)
]

# Normalize to [0, 1] and add alpha channel
NORMALIZED_COLORS = [(r/255, g/255, b/255, 1.0) for r, g, b in SUPERVISION_INSPIRED_COLORS]

# Create ListedColormap
SUPERVISION_COLORMAP = mcolors.ListedColormap(NORMALIZED_COLORS, name='supervision_inspired')

# Constants
OUTPUT_DIRS = [
    "masks",
    "rgb_and_mask",
    "palette",
    "bbox_overlay",
    "contour_gt_masks"
]