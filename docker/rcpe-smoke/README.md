# Relative camera pose environments

Source-pinned Docker overlays support `vggt-omega`, `da3`, `cut3r`, `dust3r`,
`mast3r`, `must3r`, `reloc3r`, `pi3x`, `fast3r` and `monst3r`. Each
`build_<model>_rpx.sh` selects its Dockerfile and pinned upstream runtime.
Run a build script with `--help` to inspect its image options.

The gate launcher needs Docker, NVIDIA Container Toolkit, a compatible GPU,
model-checkpoint access and a pinned RPX dataset revision. From the repository
root:

```bash
export RCPE_GPU=0
export HF_CACHE="$HOME/.cache/huggingface/hub"
export RCPE_OUTPUT="$PWD/rpx_results/relative_pose"
export RPX_REVISION=2e2a387f7f93e98c177b2e039c141eacda94e5fc
# Set RCPE_IMAGE to the model image you built or pulled.
docker/rcpe-smoke/run_rcpe_gates.sh --model dust3r --image "$RCPE_IMAGE" --gate all
```

The launcher runs smoke, micro and acceptance in order and preserves logs and
reports. `--python` selects an isolated interpreter when required by the image.
For a full run, execute `benchmark/scripts/run_relative_pose.py --help` inside
that runtime and choose `--pairs-source on_the_fly` or a prepared local
manifest. Keep the split, dataset revision, pose convention, pair protocol and
translation alignment identical when comparing models.
