# Benchmark scripts

The scripts behind the paper's runs, gates and analysis. Install the package as described in the
[benchmark README](../README.md), then run them from `benchmark/`; each one documents itself with
`--help`.

| Area | Scripts |
|:--|:--|
| Image depth | `run_depth.py`<br>`run_depth_paper.py` |
| Video depth | `run_video_depth.py` |
| Relative pose | `run_relative_pose.py`<br>`run_relative_pose_gate.py` |
| Tracking | `run_tracking.py`<br>`run_tracking_paper.py`<br>`run_tracking_gate.py` |
| VQA | `build_vqa_benchmark_plan.py`<br>`run_vqa_benchmark.py`<br>`extract_vqa_benchmark_metrics.py` |
| Depth environments and gates | `setup_depth_smoke_env.py`<br>`run_depth_smoke_gate.py`<br>`run_depth_smoke_matrix.py` |
| Manifests and pairs | `generate_pair_task_manifests.py`<br>`generate_pose_pairs_v2.py`<br>`generate_keypoint_pairs.py`<br>`generate_sparse_depth.py` |
| Analysis | `analyze_depth_paper.py`<br>`analyze_tracking_paper.py`<br>`analyze_experiment.py` |
| Visualization | `visualize_rerun.py`<br>`visualize_modalities.py`<br>`render_tracking_predictions.py` |
| Result upload (optional) | `sync_results_to_box.py` |

The paper's model wrappers are in `depth_models/`, `video_depth_models/`, `pose_models/`,
`tracking_models/` and `vqa_models/`, and each family has a Docker image listed in the benchmark
README. You don't need to touch these registries to evaluate your own model; the package's adapters
take any Python function.

`generate_pose_pairs.py` is the original pose-pair sampler, kept so existing manifests can be
regenerated. `generate_pose_pairs_v2.py` uses the newer stratified protocol. The two sample
differently, so record which one produced a manifest and don't mix their outputs.

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>
