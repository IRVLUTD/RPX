# `reference.adapters.seg_hf`

Reference HuggingFace segmentation adapter. Works with any checkpoint
whose image processor exposes one of:

- `post_process_instance_segmentation` — Mask2Former, OneFormer, MaskFormer
- `post_process_panoptic_segmentation` — same families configured for panoptic
- `post_process_semantic_segmentation` — SegFormer / DPT-for-Semantic

The adapter introspects the processor at setup time and picks the
correct post-process method.

## Install

```bash
pip install 'rpx-benchmark[depth-hf]'   # transformers + torch
```

## Usage

```python
from rpx_benchmark.reference.adapters.seg_hf import make_hf_instance_seg_model

bm = make_hf_instance_seg_model(
    "facebook/mask2former-swin-tiny-coco-instance",
    device="cuda",
    threshold=0.5,
)
```

Output contract: a single `(H, W) int32` mask whose pixel values are
consistent instance IDs. Panoptic and semantic checkpoints are
flattened to the same shape so downstream metrics don't branch on
task variant.

::: rpx_benchmark.reference.adapters.seg_hf
    options:
      show_root_toc_entry: false
      members_order: source
