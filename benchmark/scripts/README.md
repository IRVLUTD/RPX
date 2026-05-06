# Benchmark scripts

Utility scripts that run alongside the package. All are runnable directly with
`PYTHONPATH=. python scripts/<name>.py`, or via the `make` shortcuts in the
parent `Makefile`.

## Visualization

| Script | What it does | Make target |
|---|---|---|
| [`visualize_rerun.py`](visualize_rerun.py) | Interactive 6- or 18-panel Rerun viewer for any RPX (scene, phase) directly from the HuggingFace cache. Single phase or all-3-phases comparison; cividis-equalized depth; royal jewel-tone masks; T265 trajectory + Pinhole frustum; per-frame status bar. | `make visualize` / `make visualize-phases` / `make visualize-lite` / `make visualize-headless` / `make visualize-save` |
| [`visualize_modalities.py`](visualize_modalities.py) | Static 6-panel matplotlib figure of one frame across all modalities. Useful for paper figures and offline preview. | — |
| [`test_hub_visualize.py`](test_hub_visualize.py) | Smoke test: pulls a single RGB+mask sample from the HF dataset and renders it. Verifies the cache + decode path works end-to-end. | — |
| [`compare_depth_colormaps.py`](compare_depth_colormaps.py) | Side-by-side comparison of 8 colormaps (turbo, plasma, viridis, cividis, magma, inferno, gray, jet) on a single depth frame, all using the same per-frame percentile stretch. Helps pick the right colormap for a given paper or dashboard. | — |

The interactive viewer is the main entry point. See `python scripts/visualize_rerun.py --help` for full options.

### Resource expectations

| Mode | .rrd size | Build time | Peak RSS |
|---|---|---|---|
| Default (250 frames, single phase) | streamed (no .rrd) | ~13 s | ~310 MB |
| `--save` (250 frames, single phase) | ~87 MB | ~18 s | ~315 MB |
| `--lite` (stride 3, jpeg 70) | ~25 MB if `--save` | ~7 s | ~290 MB |
| `--all-phases` (3 × 250 frames) | streamed | ~80 s | ~600 MB |

Numbers from a workstation; SBC-class hardware (Pi 4, Jetson Nano) should add ~5–10× wall time and stay roughly within the same RAM ceiling. The Rerun *viewer* itself benefits from a GPU; if you're on a headless box, build with `--no-spawn --save` and open the `.rrd` on a workstation.

## Other scripts

- `build_hf_shards.py` — pack a capture tree into the per-modality tar shards the dataset hub expects.
- `upload_to_hf.py` — push a packed tree to a HuggingFace dataset repo.
- `generate_keypoint_pairs.py`, `generate_pair_task_manifests.py`, `generate_sparse_depth.py` — various data-prep helpers, see top of each file for details.
