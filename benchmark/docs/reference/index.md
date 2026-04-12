# Reference implementations

!!! info "These are examples, not framework API"
    Everything under `rpx_benchmark.reference` is an *opinionated,
    working reference implementation* for a specific model family.
    Use it as-is when it fits, or copy the pattern into your own code
    when it doesn't. The framework surface (adapter protocols, the
    `BenchmarkableModel` container, the runner, the metric registry)
    never depends on anything in `reference/`.

The split keeps the "bring your model, we bring the harness" story
crisp:

| Layer | Module | Depends on | Install extra |
|---|---|---|---|
| **Framework** (stable API) | `rpx_benchmark.{api,adapters.base,loader,metrics,tasks,runner,profiler,data}` | numpy, Pillow (required); `pydantic` / `datasets` (optional) | default / `[schemas]` / `[hf-datasets]` |
| **Reference adapters** | `rpx_benchmark.reference.adapters.*` | torch, transformers, model-specific libs | `[depth-hf]`, `[depth-unidepth]`, `[depth-metric3d]` |
| **Reference models** | `rpx_benchmark.reference.models.*` | same as the adapter family they wrap | same |

## What ships under `reference/`

### Monocular depth

| Module | Entry point | Upstream |
|---|---|---|
| [`reference.adapters.depth_hf`](adapters-depth-hf.md) | `make_hf_depth_model(checkpoint)` | `transformers.AutoModelForDepthEstimation` (Depth Anything V2, Depth Pro, ZoeDepth, …) |
| [`reference.adapters.depth_unidepth`](adapters-depth-unidepth.md) | `make_unidepth_v2_model(checkpoint)` | [UniDepth V2](https://github.com/lpiccinelli-eth/UniDepth) — self-prompted intrinsics |
| [`reference.adapters.depth_metric3d`](adapters-depth-metric3d.md) | `make_metric3d_v2_model(entry=...)` | [Metric3D V2](https://github.com/YvanYin/Metric3D) — canonical-focal letterbox |

Five ready factories register against the monocular-depth slate:
`depth_anything_v2_metric_indoor_{small,base,large}`, `depth_pro`,
`zoedepth_nyu`, `unidepth_v2_{vitb,vitl}`,
`metric3d_v2_vit_{small,large,giant2}`.

### Object segmentation

| Module | Entry point | Upstream |
|---|---|---|
| [`reference.adapters.seg_hf`](adapters-seg-hf.md) | `make_hf_instance_seg_model(checkpoint)` | Any `AutoModelForUniversalSegmentation` (Mask2Former, OneFormer, …) |

## Copy-from-template pattern

When your model family isn't covered, **copy the closest reference
adapter, rename it, and edit**. Here's the recommended workflow:

1. Pick a reference adapter whose shape matches your model. Use
   `depth_hf` for HF checkpoints, `depth_unidepth` for custom
   classes exposing `.infer(...)`, `depth_metric3d` for
   letterbox-style preprocessing.
2. Copy the module outside `rpx_benchmark.reference` — into your own
   project or into a new module under `reference/adapters/` if you
   plan to upstream.
3. Update the `from ...api import ...` / `from ...adapters.base
   import ...` imports.
4. Implement `InputAdapter.prepare` and `OutputAdapter.finalize` for
   your model's pre-/post-processing.
5. Write a `make_*` factory that loads the checkpoint and wraps the
   adapters into a `BenchmarkableModel`.
6. Register a user-facing name with
   `rpx_benchmark.models.registry.register("my_model", "path.to.my.module", "my_factory")`.

Nothing in step 2–6 touches framework code.

## Deprecated import paths

For one release, the old import paths still resolve:

- `rpx_benchmark.adapters.{depth_hf,depth_unidepth,depth_metric3d,seg_hf}`
  → shim forwarding to `rpx_benchmark.reference.adapters.*` with a
  `DeprecationWarning`.

Update imports to the new path; the shims will be removed in the
next minor release.
