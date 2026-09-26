# RPX hardware profiling

The T1 profile measures real pretrained-model inference for every monocular
depth model in the paper roster. It runs all ten models serially on one selected
GPU so every row uses the same hardware and no concurrent workload affects the
measurement.

Each model receives one excluded warm-up call followed by 1,000 measured
frames at batch size 1. Timing covers the synchronized adapter call, including
its preprocessing and postprocessing. It excludes model loading, dataset I/O,
metric evaluation, and prediction persistence. The profile also records peak
CUDA memory, loaded parameter count, counted PyTorch operations, image digest,
software versions, and the host/GPU inventory.

From the repository root on the GPU server, use an existing Hugging Face cache
that contains the pinned RPX dataset revision and model weights:

```bash
RPX_HF_CACHE=/data/narendhiran_rpx/hf-cache \
python3 benchmark/scripts/run_t1_hardware.py run \
  --gpu 0 --output-root /data/narendhiran_rpx/hardware/T1
```

The command pulls the three immutable Docker images, checks the selected GPU is
idle, verifies each image's model-specific Python executable, and runs the ten
models in order. Re-running the identical command resumes the suite: completed
profiles with matching code, configuration, and sample budget are skipped.

Inspect progress without starting work:

```bash
python3 benchmark/scripts/run_t1_hardware.py status \
  --output-root /data/narendhiran_rpx/hardware/T1
```

Use `--retry-failed` after correcting a failed model. Use `--no-pull` only when
the digest-pinned images are already present. The individual raw call records
are in each run's `hardware_calls.jsonl`; `T1_hardware_metrics.csv` and
`T1_hardware_metrics.json` are regenerated from all completed profiles.

The images are currently published in `narendhiranv04/rpx-depth-smoke`. The
digest pins prevent mutable tags from changing the experiment. Once identical
manifests are copied to the lab namespace, only the repository prefix in
`t1-image-depth.json` needs to change; the digests remain the identity check.
