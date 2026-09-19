# RPX NVS model slate

The paper slate contains exactly ten research systems split by the information
available to the model. `identity_passthrough` remains a diagnostic and is not
counted as a paper model.

| Track | Models | RPX input |
|---|---|---|
| RGB-D | DN-Splatter, SplaTAM, RTG-SLAM, GauS-SLAM | RGB, D435 depth, recorded poses |
| RGB | DepthSplat, MVSplat, pixelSplat, NoPoSplat | RGB and model-appropriate pose input |
| Compact | LightGaussian, CompGS | common scene Gaussian representation |

## Adapter boundary

DepthSplat is an in-process adapter. Every other upstream is isolated behind a
Docker-side executable named `/usr/local/bin/rpx-nvs-<model>-bridge` (or the
command supplied through `RPX_NVS_<MODEL>_BRIDGE`). The executable receives:

```text
--input INPUT.npz --output OUTPUT.npz
```

The input contains `context_rgbs`, `context_depths`, `context_poses`,
`target_pose`, and `device`. The output contains an `rgb` image and optionally a
`depth` map. Shape and modality checks are performed by the evaluator.

RGB-D bridges must consume the supplied RPX depth and poses. They must not
silently substitute monocular depth or estimated SLAM trajectories. Compact
bridges must include representation construction and compression in their
reported wall time.

## Gates

Every image is run through the same deterministic extrapolation gates:

| Gate | Samples |
|---|---:|
| smoke | 1 |
| micro | 5 |
| acceptance | 25 |

Use `docker/nvs-smoke/run_nvs_gates.sh`. A gate passes only when the requested
number of rendered PNGs exists and `result.json` contains PSNR and SSIM.

Example:

```bash
export NVS_GPU=0
export RPX_REVISION=2e2a387f7f93e98c177b2e039c141eacda94e5fc
export HF_CACHE=/data/narendhiran_rpx/docker-smoke/tracking-caches/rpx-shared/huggingface
export TORCH_CACHE=/data/narendhiran_rpx/docker-smoke/nvs-caches/torch
export NVS_OUTPUT=/data/narendhiran_rpx/docker-smoke/nvs-paper-outputs

docker/nvs-smoke/run_nvs_gates.sh \
  --model depthsplat \
  --image vndhiran123/rpx-nvs-smoke:depthsplat-rpx-latest \
  --gate all
```

## Integration status

Only DepthSplat currently has a validated upstream image and native adapter.
The other nine registry entries have a strict bridge contract but require their
model-specific cumulative Docker stage and bridge implementation before their
real RPX smoke gate can pass. Registry presence alone is not considered model
completion.
