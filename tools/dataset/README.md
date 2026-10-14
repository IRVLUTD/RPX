# Dataset tools

Scripts for preparing and checking a staged copy of the RPX dataset. They operate on the dataset
itself, so every script takes `--repo-root` pointing at the dataset directory rather than at this
repository.

| Path | Purpose |
|:--|:--|
| `prepare_ego_release.py` | packs the egocentric captures and updates the dataset metadata |
| [`qa/`](qa/README.md) | visual and structural checks run before a release |
| [`hf_card/`](hf_card/README.md) | source of the Hugging Face dataset card |

`prepare_ego_release.py` also needs `--ego-source-root`. Without `--apply` it only prints what it
would change, so review that first. Run any script with `--help` for its options. Keep generated
recordings, contact sheets and reports outside the Git tree.

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>
