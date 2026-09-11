# Server 3: Molmo validation

This deployment is intentionally isolated. Do not run host `apt`, `pip`,
Conda, CUDA, driver, or package-upgrade commands. Docker image layers remain
in the existing Docker root. All RPX source, model downloads, benchmark data,
outputs, and logs live below `/mnt/ssd1tb/naren/rpx`.

## Models and GPUs

- GPU 0: `molmo-7b-d` (`allenai/Molmo-7B-D-0924`)
- GPU 1: `molmoe-1b` (`allenai/MolmoE-1B-0924`)

The model revisions are frozen in `benchmark/scripts/vqa_models/molmo_backend.py`.
MolmoE has 1.5B active but 7.2B total parameters; it is not a dense 1B-memory
model.

## Scoring contract

Each scored row makes exactly one model generation with the original question
and its one or two required images. The requested result is a normalized
0–1000 XYXY JSON bounding box. Native Molmo `<point>` markup is retained in
`raw_output` but is not converted into a box, because a point contains no
object-extent information and cannot support a valid bbox IoU score.

Run the 14-row smoke gate and 104-row acceptance gate before scheduling the
full benchmark. Infrastructure completion proves reproducibility and coverage;
the acceptance report's `parse_rate` shows whether the checkpoint actually
obeyed the bbox protocol.

## Persistent paths

```text
/mnt/ssd1tb/naren/rpx/
├── src/RPX-vqa-molmo/
└── runtime/
    ├── hf-cache/
    ├── cache/
    ├── outputs/
    └── logs/
```

The scripts use `RPX_VQA_RUNTIME`, `RPX_VQA_IMAGE`, and `RPX_VQA_TAG`; set
those variables in every new shell or tmux command that launches work.
