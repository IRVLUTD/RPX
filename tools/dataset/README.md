# Dataset preparation and QA

These tools operate on a separately staged RPX dataset. Install the benchmark
package and the tool's optional visualization dependencies first.

```bash
python tools/dataset/prepare_ego_release.py --help
python tools/dataset/qa/check_all_modalities.py --help
python tools/dataset/qa/check_depth_benchmark.py --help
```

`prepare_ego_release.py` packs egocentric captures and updates dataset metadata.
It requires an explicit `--repo-root` and `--ego-source-root`; inspect its
preview before choosing `--apply`. The QA tools also require `--repo-root`
pointing at the dataset, not this source repository. See [QA usage](qa/README.md).
Generated recordings, contact sheets and reports belong outside the Git tree.
