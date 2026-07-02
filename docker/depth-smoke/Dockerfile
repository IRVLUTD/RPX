# syntax=docker/dockerfile:1.7

# Pinned linux/amd64 manifest for Python 3.11.15 slim-bookworm, published
# 2026-06-24. CUDA user-space libraries come from the pinned PyTorch cu128
# wheels; the host driver is injected by NVIDIA Container Toolkit at runtime.
FROM python:3.11-slim-bookworm@sha256:961f57b66f6aee85ae11272cca9219123a860ab20e2a09a1238f57e0df495825

ARG RPX_GIT_SHA
ARG UNIDEPTH_SHA=8d8cfe4c7ee15297099983607febf0d4f32eb3d6
ARG MOGE_SHA=07444410f1e33f402353b99d6ccd26bd31e469e8

LABEL org.opencontainers.image.title="RPX four-model depth smoke runtime" \
      org.opencontainers.image.description="GPU-only RPX smoke runtime for DA-V2 Large, Depth Pro, UniDepth V2 and MoGe-2 ViT-L" \
      org.opencontainers.image.source="https://github.com/IRVLUTD/RPX" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    RPX_SMOKE_ROOT=/outputs \
    RPX_ENV_ROOT=/opt/rpx-envs \
    RPX_GIT_SHA=${RPX_GIT_SHA} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN test -n "${RPX_GIT_SHA}" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rpx

COPY docker/depth-smoke/requirements.lock /tmp/requirements.lock

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip==25.3 && \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.10.0+cu128 \
      torchvision==0.25.0+cu128 && \
    python -m pip install -r /tmp/requirements.lock

RUN mkdir -p /opt/rpx-envs/sources && \
    git init /opt/rpx-envs/sources/unidepth2 && \
    git -C /opt/rpx-envs/sources/unidepth2 remote add origin \
      https://github.com/lpiccinelli-eth/UniDepth.git && \
    git -C /opt/rpx-envs/sources/unidepth2 fetch --depth 1 origin "${UNIDEPTH_SHA}" && \
    git -C /opt/rpx-envs/sources/unidepth2 checkout --detach FETCH_HEAD && \
    git init /opt/rpx-envs/sources/moge2 && \
    git -C /opt/rpx-envs/sources/moge2 remote add origin \
      https://github.com/microsoft/MoGe.git && \
    git -C /opt/rpx-envs/sources/moge2 fetch --depth 1 origin "${MOGE_SHA}" && \
    git -C /opt/rpx-envs/sources/moge2 checkout --detach FETCH_HEAD

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps /opt/rpx-envs/sources/unidepth2 && \
    python -m pip install /opt/rpx-envs/sources/moge2

COPY benchmark /opt/rpx/benchmark
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install '/opt/rpx/benchmark[hub]' psutil==7.2.2 && \
    python -c "import json, pathlib, sys; root=pathlib.Path('/opt/rpx-envs'); payload={'container': True, 'python': sys.executable, 'rpx_git_sha': '${RPX_GIT_SHA}', 'torch': {'request': '2.10.0+cu128'}, 'families': ['transformers-image', 'unidepth2', 'moge2']}; [(root/f/'bin').mkdir(parents=True, exist_ok=True) for f in payload['families']]; [(root/f/'bin'/'python').symlink_to(sys.executable) for f in payload['families']]; [(root/f/'rpx-environment.json').write_text(json.dumps(dict(payload, family=f), indent=2)+'\n') for f in payload['families']]" && \
    python -c "import moge, torch, transformers; from unidepth.models import UniDepthV2; assert torch.__version__ == '2.10.0+cu128'; print('four-model imports ready')"

COPY docker/depth-smoke/entrypoint.sh /usr/local/bin/rpx-depth-smoke
RUN chmod 0755 /usr/local/bin/rpx-depth-smoke

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/outputs"]
ENTRYPOINT ["/usr/local/bin/rpx-depth-smoke"]
CMD ["help"]
