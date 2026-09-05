# VQA smoke images

The frozen roster is in `model-matrix.json`. Build environments cumulatively
by dependency family, not by model size:

1. `grounding`: GroundingDINO + Florence-2-large
2. `transformers-modern`: PaliGemma 2 + Qwen2.5-VL (both sizes share one image)
3. `internvl`: InternVL 2.5
4. `molmo`: Molmo 2
5. `robopoint`: RoboPoint
6. `cogvlm2`: CogVLM2

Each image must implement the same JSONL request/response contract documented
in `benchmark/data/vqa_smoke/v1/README.md`. Lockfiles, model revision pins, and
GPU-tested Dockerfiles are added one family at a time; the shared benchmark
contract must not acquire model-specific scoring behavior.
