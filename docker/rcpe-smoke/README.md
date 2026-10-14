# Relative camera pose images (T4)

Pull `irvlutd/rpx:rcpe-monst3r-rpx-latest` (see the [image table](../README.md#which-image-to-pull)). The overlays are
built cumulatively, so this image holds all ten pose models, each in its own environment: `vggt-omega`, `da3`, `cut3r`,
`dust3r`, `mast3r`, `must3r`, `reloc3r`, `pi3x`, `fast3r` and `monst3r`. Choose a model with `--model` and its
interpreter with `--python /opt/rpx-envs/<env>/bin/python`, where `<env>` is the model name (`vggt` for `vggt-omega`).

## Building an overlay

To build an overlay yourself, run `build_<model>_rpx.sh`; it selects the matching Dockerfile and pinned upstream
source, and `--help` lists its options.

## Running the gates

The gate launcher needs Docker with the NVIDIA Container Toolkit, a GPU, access to the checkpoints and a pinned dataset
revision. From the repository root:

```bash
export RCPE_GPU=0
export HF_CACHE="$HOME/.cache/huggingface/hub"
export RCPE_OUTPUT="$PWD/rpx_results/relative_pose"
export RPX_REVISION=2e2a387f7f93e98c177b2e039c141eacda94e5fc
# or an image you built
export RCPE_IMAGE=irvlutd/rpx:rcpe-monst3r-rpx-latest
docker/rcpe-smoke/run_rcpe_gates.sh --model dust3r --image "$RCPE_IMAGE" --gate all
```

## Full runs

The launcher runs smoke, micro and acceptance in order and keeps their logs and reports. For a full run, see
`benchmark/scripts/run_relative_pose.py --help` inside the container, and use either `--pairs-source on_the_fly` or a
prepared local manifest. When comparing models, keep the split, dataset revision, pose convention, pair protocol and
translation alignment the same.

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="../README.md">Model environments</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a></sub>
