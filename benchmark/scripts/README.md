# Benchmark scripts

These are the scripts behind the paper's runs, gates and analysis. Install the package as described in the
[benchmark README](../README.md), then run them from `benchmark/`; each one documents itself with `--help`.

| Task | Scripts |
|---|---|
| Image depth | `run_depth.py`, `run_depth_paper.py` |
| Video depth | `run_video_depth.py` |
| Relative pose | `run_relative_pose.py`, `run_relative_pose_gate.py` |
| Tracking | `run_tracking.py`, `run_tracking_paper.py`, `run_tracking_gate.py` |
| VQA | `build_vqa_benchmark_plan.py`, `run_vqa_benchmark.py`, `extract_vqa_benchmark_metrics.py` |
| Depth environments and inference gates | `setup_depth_smoke_env.py`, `run_depth_smoke_gate.py`, `run_depth_smoke_matrix.py` |
| Manifests and pairs | `generate_pair_task_manifests.py`, `generate_pose_pairs_v2.py`, `generate_keypoint_pairs.py`, `generate_sparse_depth.py` |
| Analysis | `analyze_depth_paper.py`, `analyze_tracking_paper.py`, `analyze_experiment.py` |
| Visualization | `visualize_rerun.py`, `visualize_modalities.py`, `render_tracking_predictions.py` |
| Result upload (optional) | `sync_results_to_box.py` |

The paper's model wrappers are in `depth_models/`, `video_depth_models/`, `pose_models/`, `tracking_models/` and
`vqa_models/`, and each family has a Docker image listed in the benchmark README. You don't need to touch these
registries to evaluate your own model; the package's adapters take any Python function.

`generate_pose_pairs.py` is the original pose-pair sampler, kept so existing manifests can be regenerated.
`generate_pose_pairs_v2.py` uses the newer stratified protocol. The two sample differently, so record which one
produced a manifest and don't mix their outputs.
