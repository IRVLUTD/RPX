# Dataset QA

Visual and structural checks for a downloaded or staged RPX dataset. Every script takes
`--repo-root`, the dataset directory (not this source repository), and writes its outputs to
`rerun_check/` under the current directory unless you pass `--out-dir`.

<p align="center">
  <a href="#setup">Setup</a> ·
  <a href="#quick-check">Quick check</a> ·
  <a href="#release-check">Release check</a> ·
  <a href="#viewing-results">Viewing results</a>
</p>

## Setup

```bash
python -m venv /tmp/rpx-qa
/tmp/rpx-qa/bin/pip install -r tools/dataset/qa/requirements.txt "rpx-benchmark[hub]"
```

To fetch data to check, `download_rgbd_segmentation.py` pulls the RGB-D segmentation shards for all
three tiers (`--splits` to choose), and `download_ego.py --local-dir DIR` pulls the egocentric shards.

## Quick check

`make_rerun_check.py` inspects the middle frame of every scene phase listed in
`preview/data_studio_preview.csv`, 300 in all (100 scenes × 3 phases). It confirms that the RGB,
depth and mask shards exist with matching frame counts and image sizes, that depth has nonzero
values, and that masks contain object instances.

```bash
/tmp/rpx-qa/bin/python tools/dataset/qa/make_rerun_check.py \
    --repo-root /data/rpx --limit 10
```

Drop `--limit` to check everything. It writes `rerun_check/out/rpx_visual_check.rrd` (a Rerun
recording), `rpx_visual_contact_sheet.jpg` and `visual_check_report.json`. This is a fast preview
check, not the release check.

## Release check

`check_all_modalities.py` checks the whole dataset contract for the 300 MOS scene phases and 70 SOS
objects:

```bash
/tmp/rpx-qa/bin/python tools/dataset/qa/check_all_modalities.py --repo-root /data/rpx
```

By default it decodes the first, middle and last frame of each stream; `--sample-policy all` decodes
every frame and takes much longer. Outputs go to `rerun_check/out_all/`: a Rerun recording, a contact
sheet, a JSON report and a Markdown summary.

<details>
<summary>What the release check covers</summary>

- RGB, depth, mask, fisheye, camera pose, `masks_aux` and SAM2 metadata shards, with aligned frame
  counts and consecutive file names
- decoded samples from every image stream
- camera pose fields, finite values and unit quaternions
- MOS mask-to-object joins against `manifest/mos_mask_object_map_v1.csv`
- SOS object metadata and questionnaires
- split consistency, Data Studio preview images, release assets and Parquet row counts

</details>

`check_depth_benchmark.py` audits depth shards on their own. Depth PNGs are uint16 millimetres; the
script converts them to metres, treats 0 and 65535 as holes, and runs the benchmark's depth metrics
on sampled ground-truth frames. Pass `--fail-on-warnings` to use it as a gate.

## Viewing results

```bash
/tmp/rpx-qa/bin/rerun rerun_check/out/rpx_visual_check.rrd
/tmp/rpx-qa/bin/rerun rerun_check/out_all/rpx_all_modalities_visual_check.rrd
```

---

<sub>Part of <a href="../../../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>
