# RPX Benchmark Toolkit

> **Bring your model. We bring the dataset, the splits, the metrics, and the tables.**
>
> **RPX enables you to choose and rank perception models for robot learning** — on real-world RGB-D data, under embodied deployment conditions, with ESD-stratified difficulty splits and deployment-readiness scoring.

`rpx-benchmark` is the reference toolkit for **RPX — Robot Perception
X**, a unified real-world RGB-D benchmark for evaluating the
perception models actually deployed inside robot learning stacks
(not generic perception leaderboards). It is built so a researcher
can run an off-the-shelf HuggingFace model on an RPX difficulty
split in **one command**, compare results across the slate of
robot-learning backbones, and a team can add a whole new task or
metric in **one file**.

```bash
pip install 'rpx-benchmark[depth]'

rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard
```

That line downloads only the RGB + depth files the Hard split
references (via HuggingFace), loads the model, runs inference with a
live progress bar, prints an ESD-weighted phase-score table, measures
FLOPs and median latency, and writes `result.json` + `summary.md`.

**No Python code required.**

---

## Five-minute tour

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quickstart](getting-started/quickstart.md)**

    ---

    Install, run your first benchmark, and read the output tables.

-   :material-plug: **[Bring Your Own Model](getting-started/bring-your-own-model.md)**

    ---

    Three paths: zero-code HF checkpoint, numpy callable, or custom
    adapter stack.

-   :material-puzzle: **[Architecture](architecture/overview.md)**

    ---

    How the adapter framework, metric registry, and task registry
    fit together.

-   :material-cog: **[Extension Guides](guides/adding-a-task.md)**

    ---

    Add a new task, metric, or model adapter in one file.

</div>

---

## What RPX is

- **75,000 frames** across 100 indoor scenes, captured with an Intel
  RealSense D435 (RGB-D) + T265 (6-DoF VIO) rig.
- **Three-phase capture protocol**: each scene is recorded under
  *Clutter* → *Interaction* (human grasps/moves objects) → *Clean*.
  This isolates scene reconfiguration from scene identity so
  performance deltas mean something.
- **Effort-Stratified Difficulty (ESD) splits** per `(scene, phase)` —
  Easy / Medium / Hard derived from real annotation effort.
- **Ten benchmark tasks** on identical scenes:

    | Task | Primary metric | Numpy fast path | HF extra |
    |---|---|---|---|
    | Monocular depth | RMSE | `make_numpy_depth_model` | `[depth-hf]` |
    | Object segmentation | mIoU | `make_numpy_mask_model` | `[depth-hf]` |
    | Object detection | mAP | `make_numpy_detection_model` | — |
    | Open-vocab detection | mAP | `make_numpy_detection_model` | — |
    | Object tracking | MOTA / IDF1 | `make_numpy_tracking_model` | — |
    | Visual grounding | grounding acc. | `make_numpy_grounding_model` | — |
    | Relative camera pose | rot. / trans. err. | `make_numpy_pose_model` | — |
    | Sparse depth | RMSE | `make_numpy_sparse_depth_model` | — |
    | Novel view synthesis | PSNR / SSIM | `make_numpy_nvs_model` | — |
    | Keypoint matching | acc. @ 3 px | `make_numpy_keypoint_model` | — |
- **Scoped first-class around models used as backbones in robot
  learning**, not generic perception SOTA.

---

## Design principles

1. **The only variable should be the model.** Datasets, splits,
   metrics, reports, and deployment-readiness scoring are fixed.
2. **Adding a new task or metric should touch one file.** Plugin
   registries for models, metrics, and tasks make this a hard
   invariant.
3. **Errors should tell the user what to fix.** Every raised
   exception is a subclass of `rpx.exceptions.RPXError` and carries
   a `hint` line.
4. **Documentation is the docstrings.** This site is built from them
   automatically via mkdocstrings. No separate rewrite exists or will
   exist.
5. **CPU-first, CUDA-aware.** Every pipeline auto-falls-back to CPU
   with a clear warning when CUDA isn't available.

---

## Repository

<https://github.com/IRVLUTD/RPX>

## License

- Benchmark toolkit: **MIT**
- RPX dataset: **CC BY 4.0**
