# New-Task Adapter Playbook

When a team member starts wiring D2 (Detection), D3 (Tracking), D4 (Scene QA), D5 (Spatial QA), D6 (Relative Pose), or D7 (NVS), follow this playbook so we don't redo the discovery work that landed for D1.

The full pattern was developed in PRs #69–#78 for D1. This document distills it.

---

## Phase 1 — Build the canonical roster

**Goal**: produce a `dict[str, ModelCard]` listing every model the paper Table X for your task names, with paper_ref + install_hint + output kind, before touching upstream code.

**Where it lives**: `rpx_benchmark/adapters/depth_scaffold.py:DEPTH_MODEL_CARDS` is the reference. For a new task, create the analogue, e.g.
`rpx_benchmark/adapters/detection_scaffold.py:DETECTION_MODEL_CARDS`.

**Each card carries**:

```python
@dataclass(frozen=True)
class ModelCard:
    name: str               # Display name matching the paper table EXACTLY
    task: TaskType
    install_hint: str       # First-pass guess; refined in Phase 2
    paper_ref: str          # The bib key in root.bib (must exist)
    # Task-specific extras (e.g. depth_output_kind for depth)
```

**Critical**: every entry's `paper_ref` MUST exist in `paper-submission/overleaf/root.bib`. Grep before committing:

```bash
for key in $(grep -oP "paper_ref=\"\K[^\"]+" rpx_benchmark/adapters/<your>_scaffold.py); do
    grep -q "@.*{$key" paper-submission/overleaf/root.bib || echo "MISSING: $key"
done
```

---

## Phase 2 — WebFetch verify each model's upstream release

This is the step that prevents shipping wrong-weights adapters. For every model in the roster:

1. **WebFetch the candidate HF model card**: `huggingface.co/<owner>/<model>`. Look for:
   - paper link
   - GitHub repo link
   - quick-start code snippet
   - file listing
   - author handle / lab name

2. **Classify the result into one of four buckets**:

| Outcome | Action |
|---|---|
| HF model card + paper link + GitHub + quick-start code, all consistent | **VERIFIED-OFFICIAL**: ship a real adapter |
| Verified weights but no transformers/diffusers config (custom Python class on GitHub) | **VERIFIED-OFFICIAL with github-clone**: ship adapter that imports the upstream class |
| Candidate HF repo exists under an author/community handle with no README or no lineage | **UNVERIFIED**: ship adapter behind the safety rail |
| Nothing on HF, no GitHub release found | **NOT-YET-RELEASED**: skeleton only; document the search you did |

3. **Capture findings in `adapter_status.md`** (live table) and in the model card's install_hint.

**Tip**: WebFetch supports parallel calls. Batch 4–6 at a time in one assistant message.

---

## Phase 3 — Adapter shape

Two architectural patterns are available. Pick by use case:

### Pattern A — Subclass `BenchmarkModel` directly (D1's Image Depth pre-existing path)

Best for adapters with model-specific preprocessing or batched dispatch logic.
The 11 existing `scripts/depth_models/*.py` adapters use this pattern. Each is a plain callable wrapped by `BatchedDepthBenchmarkModel`.

### Pattern B — `BenchmarkableModel(InputAdapter, model, OutputAdapter)` composition

Best when the I/O preprocessing is generic (numpy in, numpy out). All 9 per-task `make_numpy_<task>_model` factories in `rpx_benchmark/adapters/base.py` use this pattern.

### Pattern C — Task-specific base class (D1-V's choice)

Best when the task's input/output shape needs enforcement. D1-V uses
`scripts/video_depth_models/_video_adapter_base.py:VideoDepthAdapterBase`:

```python
class VideoDepthAdapterBase(BenchmarkModel):
    task = TaskType.VIDEO_DEPTH

    DISPLAY_NAME: str = ""    # set by subclass
    OUTPUT_KIND: str = "metric"

    def __init_subclass__(cls, **kwargs):
        # Enforce that DISPLAY_NAME is set; propagate to .name / .depth_output_kind
        ...

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        # Override in subclass — returns (T, H, W) float32
        raise NotImplementedError

    def predict(self, batch):
        # Generic loop + shape enforcement
        ...
```

For D2–D7, you'll likely want **Pattern C** — a `DetectionAdapterBase`, `TrackingAdapterBase`, etc. — because the task's prediction shape (boxes vs masks vs tracklets vs accuracy scalars) is task-specific and worth enforcing in one place.

---

## Phase 4 — Bridge the canonical kebab-case names to the legacy snake_case registry

D1 today has TWO naming conventions in active use:

- **Canonical roster** (kebab-case, matches the paper table): `da-v2-large`, `depth-pro`, …
- **Legacy registry** (snake_case): `da_v2_metric_indoor`, `depth_pro`, …

The bridge in `scripts/depth_models/__init__.py:CANONICAL_TO_LEGACY` + `resolve_model_key()` lets `--model da-v2-large` (paper name) resolve to the same adapter as `--model da_v2_metric_indoor` (legacy name).

When you add a new task with its own model zoo under `scripts/<task>_models/`:

1. Create `MODEL_REGISTRY` (snake_case keys → builder)
2. Create `CANONICAL_TO_LEGACY` (kebab-case canonical → snake_case registry; `None` for unimplemented)
3. Create `resolve_model_key()` that accepts both, raises `SystemExit` with the full list of valid names + install hints

Copy the D1 implementation (`scripts/depth_models/__init__.py`) — it's ~30 lines and handles every edge case (canonical → legacy, legacy passthrough, unknown, missing-with-hint).

---

## Phase 5 — Safety rail for unverified upstream weights

When Phase 2 classifies a model as **UNVERIFIED**, do NOT skip it from the roster. Instead, ship the adapter behind the safety rail.

The exception is in the shared library:
`rpx_benchmark/exceptions.py:UnverifiedAdapterError`.

The mixin for Pattern-C base classes is in:
`scripts/video_depth_models/_unverified.py:UnverifiedAdapterMixin`.

A safety-railed adapter:

```python
class FooBarAdapter(VideoDepthAdapterBase, UnverifiedAdapterMixin):
    DISPLAY_NAME = "FooBar"
    OUTPUT_KIND = "metric"
    UNVERIFIED = True
    UNVERIFIED_NOTE = "Candidate weights at huggingface.co/<owner>/<model> are ..."
    UNVERIFIED_CANDIDATE_HF = "<owner>/<model>"

    def __init__(self, device="cuda", *, acknowledge_unverified=False):
        super().__init__(device=device)
        self._acknowledged = bool(acknowledge_unverified)

    def setup(self):
        if self._loaded:
            return
        self._assert_acknowledged()      # ← raises unless flag set
        # ... try to load; if no upstream code, raise NotImplementedError
```

For Pattern-A adapters (the inline guard from `scripts/depth_models/fe2e.py`):

```python
class FooBarAdapter:
    UNVERIFIED = True
    UNVERIFIED_NOTE = "..."
    UNVERIFIED_CANDIDATE_HF = "..."

    def __init__(self, device="cuda", batch_size=1, *,
                 acknowledge_unverified=False):
        if not acknowledge_unverified:
            raise UnverifiedAdapterError(...)
        # ... try to load; raise NotImplementedError if no upstream code
```

**CLI flag**: both `scripts/run_depth.py` and `scripts/run_video_depth.py` already accept `--acknowledge-unverified` and thread it through. For new tasks, copy the same pattern from those scripts.

**Discipline**: never publish benchmark numbers from a safety-railed adapter without confirming with the paper authors first. The runtime error and the docstring both spell this out.

---

## Phase 6 — Tests

Every task's adapter package needs three test files:

### `test_<task>_canonical_bridge.py`

- Every roster entry has a CANONICAL_TO_LEGACY mapping (no orphans, no missing)
- Non-None bridge targets are real keys in MODEL_REGISTRY
- `resolve_model_key()` round-trips canonical → legacy
- Legacy keys pass through unchanged
- Unknown names raise SystemExit with both naming conventions listed

### `test_<task>_adapter_shape.py`

For each real adapter, verify the contract without loading weights:

```python
@pytest.mark.parametrize("key,expected_name,expected_kind", REAL_ADAPTERS)
def test_real_adapter_shape(key, expected_name, expected_kind):
    mod = importlib.import_module(f"<task>_models.{key.replace('-', '_')}")
    adapter = mod.build(device="cpu")
    assert adapter.task is TaskType.<YOUR_TASK>
    assert adapter.depth_output_kind == expected_kind  # if relevant
    assert adapter.name == expected_name
```

For unverified adapters, also verify the safety rail fires:

```python
@pytest.mark.parametrize(...)
def test_unverified_adapter_safety_rail(key, ...):
    adapter = mod.build(device="cpu")
    with pytest.raises(UnverifiedAdapterError):
        adapter.setup()
    # With acknowledgement, the rail clears (then NotImplementedError until code wired)
    acked = mod.build(device="cpu", acknowledge_unverified=True)
    try:
        acked.setup()
    except NotImplementedError:
        pass
```

### `test_make_numpy_<task>_model.py`

If you add a `make_numpy_<task>_model` factory (Pattern B), test the wrapping contract: shape passthrough, output-shape validation, edge cases (empty batch, wrong dtype, etc.). See `tests/test_make_numpy_video_depth_model.py` for the reference.

---

## Phase 7 — Smoke protocol

Before launching the full sweep, run two smokes per task:

### Phase 7a — Adapter construction (no weights, no GPU)

```python
for adapter_name in all_adapters:
    mod = importlib.import_module(f"<task>_models.{adapter_name}")
    adapter = mod.build(device="cpu")  # NOT setup
    assert adapter.task is TaskType.<YOUR_TASK>
    # Verifies the adapter class is constructable; catches typos.
```

### Phase 7b — Real-GPU `setup()` sweep

```python
for adapter_name in all_adapters:
    mod = importlib.import_module(f"<task>_models.{adapter_name}")
    adapter = mod.build(device="cuda")
    try:
        adapter.setup()
        print(f"  {adapter_name}: OK")
    except ImportError as e:
        print(f"  {adapter_name}: needs install — {e}")
    except Exception as e:
        print(f"  {adapter_name}: {type(e).__name__}: {e}")
```

Real failures expected for upstream-package-required adapters; **what should not happen**: cryptic transformers/diffusers errors like "Unrecognized model" or "module X has no attribute Y". When you see those:

- It means the bare `AutoModel.from_pretrained` / `DiffusionPipeline.from_pretrained` path doesn't work for that release
- Fix: switch the adapter's setup() to the upstream-github-clone pattern (see Phase 3 Pattern C)
- D1 hit this with Video DA, VGGT-Ω, DepthCrafter, RollingDepth, ChronoDepth — all four were fixed by importing the upstream module directly instead of relying on transformers/diffusers introspection

### Phase 7c — One-clip / one-image smoke per adapter on lab GPU

For each adapter that passed setup, run on ONE cached scene:

```bash
PYTHONPATH=. python scripts/run_<task>.py \
    --model <canonical-key> --split easy --max-samples 1
```

Verify the cells.parquet has the right columns and the metric values are sane. Document any per-model schema fixes in the adapter's docstring "Verification status" section.

---

## Phase 8 — Document in `adapter_status.md`

Update [`adapter_status.md`](./adapter_status.md) with your task's table:

```
| # | Roster key | Display name | Status | Upstream | Install hint |
```

Categories: `✓ READY`, `✓ READY (needs pkg)`, `⚠ UNVERIFIED (safety rail)`, `○ SKELETON`.

---

## Anti-patterns to avoid

These all came up during D1; the rules are now baked in.

1. **No hallucinated model_ids.** If you haven't WebFetched the model card to confirm the HF ID exists, don't put it in `setup()`. Either verify it or mark the adapter as unverified.

2. **No "best-effort guessing the load incantation."** The D1 first attempts at DepthCrafter, RollingDepth, ChronoDepth, and Video DA all used the bare `from_pretrained` path because that's what the model card *seemed* to say. They all failed in `setup()` with cryptic errors. The fix in every case was to clone the upstream github repo and import their custom class. Default to this pattern when the model uses a custom architecture.

3. **No silent fallbacks for missing weights.** If the candidate HF repo has an empty README, that's a signal — don't pretend the weights are official. Use the safety rail.

4. **No publishing safety-railed numbers** in Table 3/4/etc. until the lineage is confirmed. The `--acknowledge-unverified` flag exists for sanity-checking, not for headlines.

5. **No skipping the canonical bridge.** Even if a model's adapter exists under the legacy snake_case name, add the canonical kebab-case bridge entry. The paper table uses the canonical names; the CLI should too.

6. **No spec drift between the paper roster and the code roster.** When the paper authors revise their model list (a row appears/disappears in Table X), update `<TASK>_MODEL_CARDS` in the same PR. The tests `test_every_card_has_a_skeleton_class` and `test_no_orphan_bridge_entries` fail loudly when this drifts.

---

## Reference: D1's per-PR breakdown

For the full record of what shipped during D1's setup, see `git log`:

- PR #69: D1-V runner + first real adapter + team handoff doc
- PR #74: D1VDataset `root`-field warning (caught by real-data smoke)
- PR #75: D1-F / D1-V → Image Depth / Video Depth rename
- PR #76: Canonical bridge + 9 video skeletons
- PR #77: 8 real video adapters + `make_numpy_video_depth_model`
- PR #78: All 20 adapters covered (ViGeo real, FE2E/D4RT/GemDepth safety-railed)

Each PR's commit message captures the design call. When in doubt about a D2–D7 design choice, the corresponding D1 PR's message is the precedent.
