# Manifest schemas (`rpx_benchmark.schemas`)

Strict, typed Pydantic v2 models for the JSON manifests that describe a
slice of the RPX dataset. This layer is **additive and opt-in** — the
default [`RPXDataset`][rpx_benchmark.loader.RPXDataset] loader continues
to work without Pydantic installed. Reach for this module when you
want:

- Field-level validation errors (`samples.3.depth: Field required`)
  instead of the loader's coarse checks.
- JSON Schema export for third-party tooling, TypeScript bindings, or
  doc generators.
- Pre-upload validation in release automation so malformed shards are
  caught before they reach the HuggingFace repo.

## Install

```bash
pip install 'rpx-benchmark[schemas]'
```

## Usage

```python
from rpx_benchmark import schemas
from rpx_benchmark.api import TaskType

# Validate a manifest dict or path; returns a typed Manifest object.
m = schemas.validate_manifest("manifests/monocular_depth/hard.json")

# Emit JSON Schema for external tooling.
generic = schemas.dump_json_schema()                   # top-level Manifest
depth   = schemas.dump_json_schema(TaskType.MONOCULAR_DEPTH)  # per-task sample
```

Opt into strict validation from the loader:

```python
from rpx_benchmark.loader import RPXDataset

ds = RPXDataset.from_manifest("manifest.json", validate=True)
```

When validation fails, a [`ManifestError`][rpx_benchmark.exceptions.ManifestError]
is raised. The structured Pydantic errors are surfaced in
`error.details["pydantic_errors"]`, each with a `loc` tuple pointing at
the offending sample index and field (e.g. `("samples", 3, "depth")`).

::: rpx_benchmark.schemas
    options:
      show_root_toc_entry: false
      members_order: source
