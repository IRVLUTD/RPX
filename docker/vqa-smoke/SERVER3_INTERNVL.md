# Server 3: InternVL 3.5 validation

This deployment replaces the two Molmo roster entries with the fully trained,
unsuffixed OpenGVLab checkpoints:

- `internvl3.5-1b`: `OpenGVLab/InternVL3_5-1B`, BF16 on one GPU.
- `internvl3.5-14b`: `OpenGVLab/InternVL3_5-14B`, BF16 across both Server 3 GPUs.

Both Hugging Face revisions are immutable and recorded in
`benchmark/scripts/vqa_models/internvl_backend.py`. The 14B checkpoint is not
quantized. Because it needs both 24 GB A5000s, it must not run concurrently
with the 1B checkpoint on this server.

## Scored inference contract

Each row makes one deterministic model call with all required images. Normal
rows contain the target image. In-context rows are passed in the documented
order `Image-1` (reference), then `Image-2` (target), together with InternVL's
`num_patches_list` boundary metadata.

Images use OpenGVLab's 448-pixel dynamic tiling and ImageNet normalization.
The prompt is the exact RefCOCO-style native grounding request, including the
required `<ref>...</ref>` referring-expression tags. Matching OpenGVLab's
official evaluator, the strict parser accepts exactly one native box in either
`label[[x0,y0,x1,y1]]` or `[x0,y0,x1,y1]` form on InternVL's 0--1000 grid,
rejects ambiguous or malformed boxes, and scales it relative to the target
image. No ground-truth label, ground-truth box, box repair, or second
localization call is used.

## Required order on Server 3

1. Build and verify the pinned RPX image.
2. Run the 1B smoke gate and 104-row acceptance gate on GPU 0.
3. Stop only the 1B resident container.
4. Run the 14B smoke gate and acceptance gate with GPUs `0,1`.
5. Inspect both reports and galleries before starting full benchmarks.
6. Run the full benchmarks sequentially in the same GPU configuration.

All source, caches, results, and logs remain under `/mnt/ssd1tb/naren/rpx`.
Do not install or upgrade host Python, CUDA, drivers, or system packages.
For a relayed source archive without Git metadata, export the full immutable
commit as `RPX_GIT_SHA`; the benchmark wrapper uses it for the output path.
