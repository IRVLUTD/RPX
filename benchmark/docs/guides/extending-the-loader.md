# Extending the Loader

The [`RPXDataset`][rpx_benchmark.loader.RPXDataset] class is the only
entry point for turning a manifest JSON file into a stream of
[`Sample`][rpx_benchmark.api.Sample] batches. It dispatches to
per-task `_load_*` methods based on the manifest's `task` field. To
add support for a brand-new modality or task, you either:

1. **Add new fields to an existing task's sample entries** — nothing
   to change in the loader if the new fields are optional and your
   adapter picks them out of `sample.metadata`.
2. **Add a new task type** — write a new `_load_<task>` method and
   add a dispatch branch in `_load_ground_truth`.

## Option 1: passthrough via metadata

The loader already attaches arbitrary extras to `sample.metadata`.
For example, pair tasks (keypoint matching, relative pose) use this
channel to ship a second RGB frame:

```python
# rpx_benchmark/loader.py excerpt
if "rgb_b" in entry:
    metadata["rgb_b"] = self._load_rgb(entry["rgb_b"])
```

Any custom field in your manifest that the loader doesn't recognise
will be left alone in `entry` — you can reach it from your
InputAdapter via `sample.metadata` (if you stash it there) or via
`sample.ground_truth` (if it's part of a task-specific dataclass).

## Option 2: a whole new task

1. **Add the enum value** to `TaskType` in
   `rpx_benchmark/api.py`.
2. **Add the ground-truth dataclass** next to the existing ones in
   `api.py`. Keep it simple — numpy arrays, enums, strings.
3. **Write `_load_<your_task>`** in
   `rpx_benchmark/loader.py`:

    ```python
    def _load_my_task(self, entry: Dict[str, Any]) -> MyTaskGroundTruth:
        return MyTaskGroundTruth(
            foo=self._load_array_or_inline(entry["foo"]),
            bar=entry.get("bar", "default"),
        )
    ```

4. **Dispatch from `_load_ground_truth`**:

    ```python
    if task == TaskType.MY_TASK:
        return self._load_my_task(entry)
    ```

5. **Raise [`ManifestError`][rpx_benchmark.exceptions.ManifestError]**
   from the new loader with a `hint` whenever the manifest field
   shape is wrong. Every built-in `_load_*` method does this.
6. **Add at least one test** under
   `tests/test_loader_gt_paths.py` that builds a tiny synthetic
   fixture and verifies the returned dataclass shape.

## File-level helpers to reuse

- `self._load_rgb(path)` → `uint8 H×W×3`
- `self._load_depth(path)` → `float32 H×W` in metres (16-bit PNG mm input)
- `self._load_mask(path)` → `int32 H×W` instance IDs
- `self._load_pose(path)` → `float64 4×4` SE(3) from a T265 NPZ
- `self._load_gray(path)` → `uint8 H×W` (fisheye-style)
- `self._load_boxes(path)` → `(N×4 float32, list[str])`
- `self._load_array_or_inline(value)` → accepts a path (`.npy`,
  `.json`) **or** an inline list/tuple

All of them resolve paths via `self._resolve(relative)` which
joins the dataset root with the (possibly absolute) input.
