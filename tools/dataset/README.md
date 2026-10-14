# Dataset tools

Scripts for preparing and checking a staged copy of the RPX dataset. They operate on the dataset itself, so every
script takes `--repo-root` pointing at the dataset directory rather than at this repository.

- `prepare_ego_release.py` packs the egocentric captures and updates the dataset metadata. It also needs
  `--ego-source-root`. Without `--apply` it only prints what it would change, so review that first.
- [`qa/`](qa/README.md) holds the visual and structural checks run before a release.
- [`hf_card/`](hf_card/README.md) is the source of the Hugging Face dataset card.

Run any script with `--help` for its options. Keep generated recordings, contact sheets and reports outside the Git
tree.
