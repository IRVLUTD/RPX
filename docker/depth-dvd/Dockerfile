# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG RPX_GIT_SHA
ARG DVD_SHA=62f52d16faa2cac10e31eb7815ece6e61d5449ec

LABEL org.opencontainers.image.title="RPX DVD v1.1 video-depth runtime" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}" \
      org.opencontainers.image.dvd.revision="${DVD_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA} \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

RUN test -n "${RPX_GIT_SHA}" && \
    git init /opt/rpx-envs/sources/dvd && \
    git -C /opt/rpx-envs/sources/dvd remote add origin \
      https://github.com/EnVision-Research/DVD.git && \
    git -C /opt/rpx-envs/sources/dvd fetch \
      --depth 1 origin "${DVD_SHA}" && \
    git -C /opt/rpx-envs/sources/dvd checkout --detach FETCH_HEAD && \
    git config --system --add safe.directory /opt/rpx-envs/sources/dvd

COPY docker/depth-dvd/requirements-dvd.lock /tmp/requirements-dvd.lock

# Keep DVD isolated from the already accepted model environments.  The
# cumulative base supplies the CUDA-enabled PyTorch stack; DVD's exact
# user-space dependencies live only in this venv.
RUN python -m venv --system-site-packages /opt/rpx-envs/dvd && \
    /opt/rpx-envs/dvd/bin/python -m pip install \
      --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1 && \
    /opt/rpx-envs/dvd/bin/python -m pip install \
      -r /tmp/requirements-dvd.lock && \
    /opt/rpx-envs/dvd/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'dvd-source.pth').write_text('/opt/rpx-envs/sources/dvd\\n')"

COPY benchmark /opt/rpx/benchmark

RUN /opt/rpx-envs/dvd/bin/python -m pip install \
      --no-deps --force-reinstall /opt/rpx/benchmark && \
    /opt/rpx-envs/dvd/bin/python -c \
      "import accelerate,diffusers,einops,huggingface_hub,numpy,omegaconf,peft,safetensors,torch,transformers; from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule; from test_script.test_single_video import generate_depth_sliced; assert torch.cuda.is_available() is False or torch.version.cuda is not None; print('DVD imports ready:', torch.__version__)" && \
    /opt/rpx-envs/dvd/bin/python -m pip freeze \
      > /opt/rpx-envs/dvd/pip-freeze.txt

RUN /opt/rpx-envs/dvd/bin/python - <<'PY'
import json
from pathlib import Path

Path("/opt/rpx-envs/dvd/rpx-environment.json").write_text(
    json.dumps(
        {
            "container": True,
            "family": "dvd",
            "models": ["dvd"],
            "source": "/opt/rpx-envs/sources/dvd",
            "upstream_revision": "62f52d16faa2cac10e31eb7815ece6e61d5449ec",
            "checkpoint_repo": "FayeHongfeiZhang/DVD",
            "checkpoint_revision": "769534dec2d77f8da3667719568fbe57fe69cd9b",
            "checkpoint_filename": "dvd_1.1.safetensors",
            "checkpoint_sha256": "63079861484dfd6397a04f6327a0e20bbcee95ff1440cb7e61a83be2ecc6d854",
            "wan_repo": "Wan-AI/Wan2.1-T2V-1.3B",
            "wan_revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
            "alignment": "ls_disparity",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

WORKDIR /opt/rpx/benchmark
ENTRYPOINT []
