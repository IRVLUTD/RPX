# Benchmark runners

Install the package from [the benchmark README](../README.md), then run scripts
from `benchmark/`: `python scripts/<script>.py --help`.

| Purpose | Entry points |
|---|---|
| Image depth | `run_depth.py`, `run_depth_paper.py` |
| Video depth | `run_video_depth.py` |
| Relative pose | `run_relative_pose.py`, `run_relative_pose_gate.py` |
| Tracking | `run_tracking.py`, `run_tracking_paper.py`, `run_tracking_gate.py` |
| VQA | `build_vqa_benchmark_plan.py`, `run_vqa_benchmark.py`, `extract_vqa_benchmark_metrics.py` |
| Depth environment and inference gates | `setup_depth_smoke_env.py`, `run_depth_smoke_gate.py`, `run_depth_smoke_matrix.py` |
| Dataset and pair preparation | `generate_pair_task_manifests.py`, `generate_pose_pairs_v2.py`, `generate_keypoint_pairs.py`, `generate_sparse_depth.py` |
| Result analysis | `analyze_depth_paper.py`, `analyze_tracking_paper.py`, `analyze_experiment.py` |
| Visualization | `visualize_rerun.py`, `visualize_modalities.py`, `render_tracking_predictions.py` |
| Optional result upload | `sync_results_to_box.py` |

Model implementations live in `depth_models/`, `video_depth_models/`,
`pose_models/`, `tracking_models/` and `vqa_models/`. Use the task-specific
Docker environments linked from the benchmark README. Custom models can use
the package's public callable APIs without editing these registries.

The original pose-pair generator is retained for existing manifest compatibility;
`generate_pose_pairs_v2.py` implements the newer stratified protocol. These
produce different sampling protocols and should not be interchanged silently.
