import sys
from pathlib import Path

_VQA_GT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VQA_GT))               # vqa_gt/ itself (gt_spatial, lib, sos_catalog, ...)
sys.path.insert(0, str(_VQA_GT.parent))         # visual_grounding_gt/ (generate_spatial_gt.py)
