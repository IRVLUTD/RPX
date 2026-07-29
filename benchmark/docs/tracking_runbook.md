# RPX D3 tracking runbook

## Protocol fixed by the paper

- The evaluation unit is a complete `(scene, phase)` clip (250 frames).
- Trackers are initialized with the released ground-truth instance masks on
  frame 0. Text prompts are not used for mask-initialized D3 models. Frame 0
  is preserved in the outputs but excluded from metrics because it is supplied
  to the model rather than predicted.
- Positive mask values are persistent instance IDs within the clip.
- Predictions are evaluated as tight boxes derived from the predicted and GT
  masks using the official TrackEval implementation:
  - MOTA
  - IDF1
  - HOTA (mean over thresholds 0.05 through 0.95)
  - ID switches
- CLEAR and Identity association use the MOTChallenge IoU threshold 0.5.
- No depth F-score is part of D3.

YOLOE is in the paper's D2 detection roster. The historical
`run_yoloe_tracking_smoke.py` is retained only as an engineering check for
YOLOE plus ByteTrack; its synthetic bus video and text prompts are not an RPX
D3 result.

## Released ground truth

The pinned `IRVLUTD/RPX` release contains RGB, temporally consistent instance
masks and `sam2/mask_to_object.json`. It does not contain the historical
`tracklets/v1.json` referenced by older manifests. The D3 runner therefore
uses the masks as authoritative GT and downloads `sam2_meta` for ID/name
provenance.

Expected production inventory:

| Split | Frames | Scene-phase clips |
|---|---:|---:|
| Easy | 24,750 | 99 |
| Medium | 24,750 | 99 |
| Hard | 25,500 | 102 |
| Total | 75,000 | 300 |

## Resume contract

Predictions are written atomically to:

`predictions/<scene>/<phase>/<frame>.npz`

Each file contains one non-negative integer `mask` of shape `480×640`. A
scene-phase `_complete.json` marker is written only after all 250 predictions
are valid. A completed clip is skipped on rerun with zero model propagation.
If interruption occurs inside a clip, that clip is recomputed from frame 0:
SAM 2's temporal memory state cannot be reconstructed from isolated output
masks, so pretending to resume at an arbitrary frame would be invalid.

## Docker layout

`docker/tracking-smoke/Dockerfile` is cumulative:

1. `tracking_base`
2. `tracking_all_yoloe` (historical engineering smoke)
3. `tracking_all_sam2` (first paper-valid D3 model)

The SAM 2 target retains preceding environments, while Docker shares their
immutable layers. Do not run `docker system prune -a` while benchmark
containers or required images exist.

## Outputs

Each split writes `cells.csv`, `cells.parquet`, `result.json`,
`run_metadata.json`, and validated prediction masks. After all splits:

- `combined_cells.parquet` (exactly 300 cells)
- `paper_analysis.json`
- `paper_analysis.md`
- `paper_table.csv`

The analysis reports repeated-measures Φ/Wilks statistics, paired
Hotelling/Holm transitions, per-tier effects, phase×difficulty, and phase
means. JEDI emits `bounds_missing` until an approved versioned D3 bounds file
is supplied; it never guesses bounds.
