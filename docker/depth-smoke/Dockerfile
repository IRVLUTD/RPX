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
      git \
      libglib2.0-0 \
      libgomp1 \
      procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rpx

COPY docker/depth-smoke/requirements.lock /tmp/requirements.lock
COPY --chmod=0755 docker/depth-smoke/install_pinned.sh /usr/local/bin/rpx-pip-install

# Keep the large CUDA wheels in separate layers. Besides improving layer reuse,
# this prevents a single multi-gigabyte registry upload from exceeding Docker
# Hub's upload-session window on slower links. The package set is exactly the
# dependency closure declared by torch 2.10.0+cu128.
RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install --upgrade pip==25.3

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install \
      filelock==3.29.0 \
      typing-extensions==4.15.0 \
      sympy==1.14.0 \
      networkx==3.6.1 \
      jinja2==3.1.6 \
      fsspec==2026.4.0 \
      cuda-bindings==12.9.4 \
      cuda-pathfinder==1.2.2

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install \
      nvidia-cuda-nvrtc-cu12==12.8.93 \
      nvidia-cuda-runtime-cu12==12.8.90 \
      nvidia-cuda-cupti-cu12==12.8.90 \
      nvidia-nvtx-cu12==12.8.90 \
      nvidia-nvjitlink-cu12==12.8.93 \
      nvidia-cufile-cu12==1.13.1.3

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install nvidia-cublas-cu12==12.8.4.1

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install nvidia-cudnn-cu12==9.10.2.21

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install \
      nvidia-cufft-cu12==11.3.3.83 \
      nvidia-curand-cu12==10.3.9.90

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install nvidia-cusparse-cu12==12.5.8.93

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install nvidia-cusparselt-cu12==0.7.1

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install nvidia-cusolver-cu12==11.7.3.90

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install \
      nvidia-nccl-cu12==2.27.5 \
      nvidia-nvshmem-cu12==3.4.5

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install triton==3.6.0

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install --no-deps torch==2.10.0+cu128

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    rpx-pip-install --no-deps torchvision==0.25.0+cu128

RUN --mount=type=bind,from=torch_wheelhouse,target=/wheelhouse,readonly \
    python -m pip install --find-links /wheelhouse -r /tmp/requirements.lock

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
    python -m pip install --no-deps /opt/rpx-envs/sources/moge2

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


# Extend the exact four-model image that passed the GPU acceptance matrix.
FROM vndhiran123/rpx-depth-smoke@sha256:c539f021c26ce95136e72422b9ec0a8a187b531f383b6371af49dd4bc39e4d6f AS depth_all_da3

ARG RPX_GIT_SHA
ARG DA3_SHA=41736238f5bced4debf3f2a12375d2466874866d

LABEL org.opencontainers.image.title="RPX isolated depth runtimes" \
      org.opencontainers.image.description="Accepted four-model runtime plus isolated Depth Anything 3 family" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA}

COPY docker/depth-smoke/requirements-da3.lock /tmp/requirements-da3.lock

RUN mkdir -p /opt/rpx-envs/sources && \
    git init /opt/rpx-envs/sources/da3 && \
    git -C /opt/rpx-envs/sources/da3 remote add origin \
      https://github.com/ByteDance-Seed/depth-anything-3.git && \
    git -C /opt/rpx-envs/sources/da3 fetch --depth 1 origin "${DA3_SHA}" && \
    git -C /opt/rpx-envs/sources/da3 checkout --detach FETCH_HEAD

# Share the large CUDA/PyTorch installation, while allowing DA3-specific
# packages such as NumPy 1.26 to override the four-model environment.
RUN python -m venv --system-site-packages /opt/rpx-envs/da3 && \
    /opt/rpx-envs/da3/bin/python -m pip install --upgrade pip==25.3 && \
    /opt/rpx-envs/da3/bin/python -m pip install -r /tmp/requirements-da3.lock && \
    /opt/rpx-envs/da3/bin/python -m pip install \
      --no-deps -e /opt/rpx-envs/sources/da3

COPY benchmark /opt/rpx/benchmark

RUN python -m pip install --no-deps --force-reinstall /opt/rpx/benchmark && \
    /opt/rpx-envs/da3/bin/python -c \
      "import numpy, torch, xformers; from depth_anything_3.api import DepthAnything3; assert numpy.__version__ == '1.26.4'; assert torch.__version__ == '2.10.0+cu128'; print('DA3 imports ready')" && \
    /opt/rpx-envs/da3/bin/python -m pip freeze \
      > /opt/rpx-envs/da3/pip-freeze.txt && \
    /opt/rpx-envs/da3/bin/python -c \
      "import json,pathlib,sys; p=pathlib.Path('/opt/rpx-envs/da3/rpx-environment.json'); p.write_text(json.dumps({'container': True, 'family': 'da3', 'models': ['da3-metric-l','da3-video'], 'python': sys.executable, 'rpx_git_sha': '${RPX_GIT_SHA}', 'upstream_revision': '${DA3_SHA}'}, indent=2)+'\n')"

WORKDIR /opt/rpx/benchmark


# Lotus-2 isolated runtime, extending the accepted six-model DA3 stage.
FROM depth_all_da3 AS depth_all_lotus2

ARG RPX_GIT_SHA
ARG LOTUS2_SHA=2d5e4522f7213611184fd31992d0fac17ec36035

LABEL org.opencontainers.image.title="RPX isolated depth runtimes with Lotus-2" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA}

COPY docker/depth-smoke/requirements-lotus2.lock /tmp/requirements-lotus2.lock

RUN mkdir -p /opt/rpx-envs/sources && \
    git init /opt/rpx-envs/sources/lotus2 && \
    git -C /opt/rpx-envs/sources/lotus2 remote add origin \
      https://github.com/EnVision-Research/Lotus-2.git && \
    git -C /opt/rpx-envs/sources/lotus2 fetch \
      --depth 1 origin "${LOTUS2_SHA}" && \
    git -C /opt/rpx-envs/sources/lotus2 checkout --detach FETCH_HEAD

RUN python -m venv --system-site-packages /opt/rpx-envs/lotus2 && \
    /opt/rpx-envs/lotus2/bin/python -m pip install --upgrade pip==25.3 && \
    /opt/rpx-envs/lotus2/bin/python -m pip install \
      -r /tmp/requirements-lotus2.lock && \
    /opt/rpx-envs/lotus2/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'rpx-lotus2-source.pth').write_text('/opt/rpx-envs/sources/lotus2\n')"

COPY benchmark /opt/rpx/benchmark

RUN python -m pip install --no-deps --force-reinstall /opt/rpx/benchmark && \
    /opt/rpx-envs/lotus2/bin/python -c \
      "import diffusers,transformers,peft,infer,pipeline,torch,numpy; assert transformers.__version__ == '4.46.3'; assert diffusers.__version__ == '0.32.2'; assert numpy.__version__ == '1.26.4'; assert torch.__version__ == '2.10.0+cu128'; print('Lotus-2 imports ready')" && \
    /opt/rpx-envs/lotus2/bin/python -m pip freeze \
      > /opt/rpx-envs/lotus2/pip-freeze.txt && \
    /opt/rpx-envs/lotus2/bin/python -c \
      "import json,pathlib,sys; pathlib.Path('/opt/rpx-envs/lotus2/rpx-environment.json').write_text(json.dumps({'container': True, 'family': 'lotus2', 'models': ['lotus-2'], 'python': sys.executable, 'rpx_git_sha': '${RPX_GIT_SHA}', 'upstream_revision': '${LOTUS2_SHA}'}, indent=2)+'\n')"

WORKDIR /opt/rpx/benchmark


# Metric3D V2 isolated runtime, extending the accepted Lotus-2 stage.
FROM depth_all_lotus2 AS depth_all_metric3d

ARG RPX_GIT_SHA
ARG METRIC3D_SHA=eb5b6fac0dc155e4e52f576e304fbf11655ff339

LABEL org.opencontainers.image.title="RPX isolated depth runtimes with Metric3D V2" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA} \
    RPX_METRIC3D_SOURCE=/opt/rpx-envs/sources/metric3d \
    TORCH_HOME=/cache/torch

COPY docker/depth-smoke/requirements-metric3d.lock /tmp/requirements-metric3d.lock

RUN mkdir -p /opt/rpx-envs/sources && \
    git init /opt/rpx-envs/sources/metric3d && \
    git -C /opt/rpx-envs/sources/metric3d remote add origin \
      https://github.com/YvanYin/Metric3D.git && \
    git -C /opt/rpx-envs/sources/metric3d fetch \
      --depth 1 origin "${METRIC3D_SHA}" && \
    git -C /opt/rpx-envs/sources/metric3d checkout --detach FETCH_HEAD

RUN python -m venv --system-site-packages /opt/rpx-envs/metric3d && \
    /opt/rpx-envs/metric3d/bin/python -m pip install \
      --upgrade pip==25.3 && \
    /opt/rpx-envs/metric3d/bin/python -m pip install \
      --no-deps -r /tmp/requirements-metric3d.lock && \
    /opt/rpx-envs/metric3d/bin/python -m pip install \
      --no-deps mmengine==0.10.7 && \
    /opt/rpx-envs/metric3d/bin/python -m pip install \
      --no-build-isolation --no-deps mmcv==1.7.2

COPY benchmark /opt/rpx/benchmark

RUN python -m pip install \
      --no-deps --force-reinstall /opt/rpx/benchmark && \
    PYTHONPATH=/opt/rpx-envs/sources/metric3d \
      /opt/rpx-envs/metric3d/bin/python -c \
      "import runpy,torch,numpy,cv2,mmcv,mmengine; ns=runpy.run_path('/opt/rpx-envs/sources/metric3d/hubconf.py'); assert 'metric3d_vit_giant2' in ns; assert torch.__version__ == '2.10.0+cu128'; assert numpy.__version__ == '1.26.4'; print('Metric3D imports ready')" && \
    /opt/rpx-envs/metric3d/bin/python -m pip freeze \
      > /opt/rpx-envs/metric3d/pip-freeze.txt && \
    /opt/rpx-envs/metric3d/bin/python -c \
      "import json,pathlib,sys; pathlib.Path('/opt/rpx-envs/metric3d/rpx-environment.json').write_text(json.dumps({'container': True, 'family': 'metric3d', 'models': ['metric3d-v2'], 'python': sys.executable, 'rpx_git_sha': '${RPX_GIT_SHA}', 'upstream_revision': '${METRIC3D_SHA}'}, indent=2)+'\n')"

WORKDIR /opt/rpx/benchmark


# Shared geometry-video environment for VGGT-Ω and ViGeo.
FROM depth_all_metric3d AS depth_all_geometry_video

ARG RPX_GIT_SHA
ARG VGGT_SHA=a288dd0f14786c93483e45524328726ab7b1b4ce
ARG VIGEO_SHA=c2cf70302be459ca7be758185fa71b89949ee4b5

LABEL org.opencontainers.image.title="RPX geometry-video depth runtime" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA}

COPY docker/depth-smoke/requirements-geometry-video.lock \
     /tmp/requirements-geometry-video.lock

RUN mkdir -p /opt/rpx-envs/sources && \
    git init /opt/rpx-envs/sources/vggt && \
    git -C /opt/rpx-envs/sources/vggt remote add origin \
      https://github.com/facebookresearch/vggt.git && \
    git -C /opt/rpx-envs/sources/vggt fetch \
      --depth 1 origin "${VGGT_SHA}" && \
    git -C /opt/rpx-envs/sources/vggt checkout --detach FETCH_HEAD && \
    git init /opt/rpx-envs/sources/vigeo && \
    git -C /opt/rpx-envs/sources/vigeo remote add origin \
      https://github.com/aigc3d/ViGeo.git && \
    git -C /opt/rpx-envs/sources/vigeo fetch \
      --depth 1 origin "${VIGEO_SHA}" && \
    git -C /opt/rpx-envs/sources/vigeo checkout --detach FETCH_HEAD

RUN python -m venv \
      --system-site-packages /opt/rpx-envs/geometry-video && \
    /opt/rpx-envs/geometry-video/bin/python -m pip install \
      --upgrade pip==25.3 && \
    /opt/rpx-envs/geometry-video/bin/python -m pip install \
      --no-deps -r /tmp/requirements-geometry-video.lock && \
    /opt/rpx-envs/geometry-video/bin/python -m pip install \
      --no-deps \
      /opt/rpx-envs/sources/vggt \
      /opt/rpx-envs/sources/vigeo && \
    ln -s /opt/rpx-envs/geometry-video /opt/rpx-envs/vggt && \
    ln -s /opt/rpx-envs/geometry-video /opt/rpx-envs/vigeo

COPY benchmark /opt/rpx/benchmark

RUN python -m pip install \
      --no-deps --force-reinstall /opt/rpx/benchmark && \
    /opt/rpx-envs/geometry-video/bin/python -c \
      "import numpy,torch,einops,safetensors,vggt,vigeo; from vggt.models.vggt import VGGT; from vigeo import ViGeo; assert numpy.__version__ == '1.26.4'; assert torch.__version__ == '2.10.0+cu128'; print('VGGT and ViGeo imports ready')" && \
    /opt/rpx-envs/geometry-video/bin/python -m pip freeze \
      > /opt/rpx-envs/geometry-video/pip-freeze.txt && \
    /opt/rpx-envs/geometry-video/bin/python -c \
      "import json,pathlib,sys; pathlib.Path('/opt/rpx-envs/geometry-video/rpx-environment.json').write_text(json.dumps({'container': True, 'family': 'geometry-video', 'models': ['vggt-omega','vigeo'], 'python': sys.executable, 'rpx_git_sha': '${RPX_GIT_SHA}', 'upstreams': {'vggt': '${VGGT_SHA}', 'vigeo': '${VIGEO_SHA}'}}, indent=2)+'\n')"

WORKDIR /opt/rpx/benchmark

FROM depth_all_geometry_video AS depth_all_diffusion_video

ARG CHRONO_SHA=2580e891eb33a5c3a95eab27b745f64ac28b3514
ARG DEPTHCRAFTER_SHA=fc83d365f2b781ab05aeb94b13f7e97417df7d97

COPY docker/depth-smoke/requirements-chrono.lock /tmp/requirements-chrono.lock
COPY docker/depth-smoke/requirements-depthcrafter.lock /tmp/requirements-depthcrafter.lock

RUN git init /opt/rpx-envs/sources/chrono && \
    git -C /opt/rpx-envs/sources/chrono remote add origin \
      https://github.com/jiahao-shao1/ChronoDepth.git && \
    git -C /opt/rpx-envs/sources/chrono fetch --depth 1 origin "${CHRONO_SHA}" && \
    git -C /opt/rpx-envs/sources/chrono checkout --detach FETCH_HEAD

RUN git init /opt/rpx-envs/sources/depthcrafter && \
    git -C /opt/rpx-envs/sources/depthcrafter remote add origin \
      https://github.com/Tencent/DepthCrafter.git && \
    git -C /opt/rpx-envs/sources/depthcrafter fetch --depth 1 origin "${DEPTHCRAFTER_SHA}" && \
    git -C /opt/rpx-envs/sources/depthcrafter checkout --detach FETCH_HEAD

RUN python -m venv --system-site-packages /opt/rpx-envs/chrono && \
    /opt/rpx-envs/chrono/bin/python -m pip install \
      --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1 && \
    /opt/rpx-envs/chrono/bin/python -m pip install --no-deps -r /tmp/requirements-chrono.lock && \
    /opt/rpx-envs/chrono/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'rpx-chrono.pth').write_text('/opt/rpx-envs/sources/chrono\n')"

RUN python -m venv --system-site-packages /opt/rpx-envs/depthcrafter && \
    /opt/rpx-envs/depthcrafter/bin/python -m pip install \
      --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1 && \
    /opt/rpx-envs/depthcrafter/bin/python -m pip install --no-deps -r /tmp/requirements-depthcrafter.lock && \
    /opt/rpx-envs/depthcrafter/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'rpx-depthcrafter.pth').write_text('/opt/rpx-envs/sources/depthcrafter\n')"

COPY benchmark /opt/rpx/benchmark

RUN /opt/rpx-envs/chrono/bin/python -c \
      "import torch,diffusers,transformers,xformers; from chronodepth import ChronoDepthPipeline; print('ChronoDepth imports ready')" && \
    /opt/rpx-envs/depthcrafter/bin/python -c \
      "import torch,diffusers,transformers,xformers; from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline; print('DepthCrafter imports ready')"

WORKDIR /opt/rpx/benchmark

FROM depth_all_diffusion_video AS depth_all_rolling_vda_video

ARG ROLLINGDEPTH_SHA=c233765fb7adf9682442a9fbef3eb07212bb70fd
ARG VDA_SHA=4f5ae23172ba60fd7bc11ef671cca678842c7072

COPY docker/depth-smoke/requirements-rollingdepth.lock /tmp/requirements-rollingdepth.lock
COPY docker/depth-smoke/requirements-video-depth-anything.lock /tmp/requirements-video-depth-anything.lock

RUN git init /opt/rpx-envs/sources/rollingdepth && \
    git -C /opt/rpx-envs/sources/rollingdepth remote add origin \
      https://github.com/prs-eth/rollingdepth.git && \
    git -C /opt/rpx-envs/sources/rollingdepth fetch --depth 1 origin "${ROLLINGDEPTH_SHA}" && \
    git -C /opt/rpx-envs/sources/rollingdepth checkout --detach FETCH_HEAD

RUN git init /opt/rpx-envs/sources/video-depth-anything && \
    git -C /opt/rpx-envs/sources/video-depth-anything remote add origin \
      https://github.com/DepthAnything/Video-Depth-Anything.git && \
    git -C /opt/rpx-envs/sources/video-depth-anything fetch --depth 1 origin "${VDA_SHA}" && \
    git -C /opt/rpx-envs/sources/video-depth-anything checkout --detach FETCH_HEAD

RUN python -m venv --system-site-packages /opt/rpx-envs/rollingdepth && \
    /opt/rpx-envs/rollingdepth/bin/python -m pip install \
      --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1 && \
    /opt/rpx-envs/rollingdepth/bin/python -m pip install --no-deps -r /tmp/requirements-rollingdepth.lock && \
    /opt/rpx-envs/rollingdepth/bin/python -m pip install --no-deps -e /opt/rpx-envs/sources/rollingdepth/diffusers && \
    /opt/rpx-envs/rollingdepth/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'rpx-rollingdepth.pth').write_text('/opt/rpx-envs/sources/rollingdepth\\n')"

RUN python -m venv --system-site-packages /opt/rpx-envs/video-depth-anything && \
    /opt/rpx-envs/video-depth-anything/bin/python -m pip install \
      --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1 && \
    /opt/rpx-envs/video-depth-anything/bin/python -m pip install --no-deps -r /tmp/requirements-video-depth-anything.lock && \
    /opt/rpx-envs/video-depth-anything/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'rpx-video-depth-anything.pth').write_text('/opt/rpx-envs/sources/video-depth-anything\\n')"

RUN ln -sfn /opt/rpx-envs/rollingdepth /opt/rpx-envs/rolling && \
    ln -sfn /opt/rpx-envs/video-depth-anything /opt/rpx-envs/video_da

COPY benchmark /opt/rpx/benchmark

RUN /opt/rpx-envs/rollingdepth/bin/python -c \
      "import torch,transformers,xformers; from rollingdepth import RollingDepthPipeline; print('RollingDepth imports ready')" && \
    /opt/rpx-envs/video-depth-anything/bin/python -c \
      "import torch,cv2,imageio,decord; from video_depth_anything.video_depth import VideoDepthAnything; print('Video Depth Anything imports ready')"

WORKDIR /opt/rpx/benchmark
