ARG TRACKING_BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

FROM ${TRACKING_BASE_IMAGE} AS tracking_base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

ENV YOLO_CONFIG_DIR=/cache/ultralytics/config
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface
ENV XDG_CACHE_HOME=/cache/xdg

RUN apt-get update && \
    apt-get install -y \
      --no-install-recommends \
      ca-certificates \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
      /opt/rpx-envs \
      /opt/rpx/benchmark \
      /cache/ultralytics/models \
      /cache/ultralytics/config \
      /cache/huggingface \
      /cache/torch \
      /cache/xdg

WORKDIR /opt/rpx/benchmark

COPY benchmark /opt/rpx/benchmark


FROM tracking_base AS tracking_all_yoloe

ARG RPX_GIT_SHA=unknown

RUN python -m venv \
      --system-site-packages \
      /opt/rpx-envs/yoloe && \
    /opt/rpx-envs/yoloe/bin/python -m pip install \
      --upgrade \
      pip \
      setuptools \
      wheel

COPY docker/tracking-smoke/requirements-yoloe.lock \
     /tmp/requirements-yoloe.lock

RUN /opt/rpx-envs/yoloe/bin/python -m pip install \
      --no-deps \
      -r /tmp/requirements-yoloe.lock

RUN RPX_MANIFEST_GIT_SHA="${RPX_GIT_SHA}" \
    /opt/rpx-envs/yoloe/bin/python - <<'PY'
import json
import os
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLOE

manifest = {
    "container": True,
    "family": "yoloe",
    "models": ["yoloe"],
    "python": "/opt/rpx-envs/yoloe/bin/python",
    "rpx_git_sha": os.environ["RPX_MANIFEST_GIT_SHA"],
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "ultralytics": ultralytics.__version__,
    "opencv": cv2.__version__,
}

path = Path(
    "/opt/rpx-envs/yoloe/rpx-environment.json"
)

path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print("YOLOE class:", YOLOE.__name__)
print(json.dumps(manifest, indent=2))
print("Wrote:", path)
PY

WORKDIR /opt/rpx/benchmark


# Cumulative target: this image retains the YOLOE engineering environment
# while adding an isolated, paper-valid SAM 2 D3 environment. Docker shares
# all preceding layers, so tagging both targets does not duplicate their
# bytes on disk.
FROM tracking_all_yoloe AS tracking_all_sam2

ARG RPX_GIT_SHA=unknown
ARG SAM2_GIT_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4
ARG TRACKEVAL_GIT_SHA=12c8791b303e0a0b50f753af204249e622d0281a

RUN python -m venv \
      --system-site-packages \
      /opt/rpx-envs/sam2 && \
    /opt/rpx-envs/sam2/bin/python -m pip install \
      --upgrade \
      pip \
      setuptools \
      wheel

COPY docker/tracking-smoke/requirements-sam2.lock \
     /tmp/requirements-sam2.lock

RUN /opt/rpx-envs/sam2/bin/python -m pip install \
      -r /tmp/requirements-sam2.lock && \
    SAM2_BUILD_CUDA=0 \
    /opt/rpx-envs/sam2/bin/python -m pip install \
      --no-deps \
      "git+https://github.com/facebookresearch/sam2.git@${SAM2_GIT_SHA}" && \
    /opt/rpx-envs/sam2/bin/python -m pip install \
      --no-deps \
      "git+https://github.com/JonathonLuiten/TrackEval.git@${TRACKEVAL_GIT_SHA}" && \
    /opt/rpx-envs/sam2/bin/python -m pip install \
      --no-deps \
      --force-reinstall \
      /opt/rpx/benchmark

RUN RPX_MANIFEST_GIT_SHA="${RPX_GIT_SHA}" \
    RPX_SAM2_GIT_SHA="${SAM2_GIT_SHA}" \
    RPX_TRACKEVAL_GIT_SHA="${TRACKEVAL_GIT_SHA}" \
    /opt/rpx-envs/sam2/bin/python - <<'PY'
import json
import os
from pathlib import Path

import pyarrow
import torch
import trackeval
from sam2.sam2_video_predictor import SAM2VideoPredictor

manifest = {
    "container": True,
    "family": "tracking",
    "models": ["yoloe-engineering-smoke", "sam2"],
    "paper_valid_d3_models": ["sam2"],
    "python": "/opt/rpx-envs/sam2/bin/python",
    "rpx_git_sha": os.environ["RPX_MANIFEST_GIT_SHA"],
    "sam2_git_sha": os.environ["RPX_SAM2_GIT_SHA"],
    "trackeval_git_sha": os.environ["RPX_TRACKEVAL_GIT_SHA"],
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "pyarrow": pyarrow.__version__,
}
path = Path("/opt/rpx-envs/sam2/rpx-environment.json")
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("SAM2 predictor:", SAM2VideoPredictor.__name__)
print("TrackEval:", Path(trackeval.__file__).resolve())
print(json.dumps(manifest, indent=2))
PY

ENV RPX_TRACKING_PYTHON=/opt/rpx-envs/sam2/bin/python

WORKDIR /opt/rpx/benchmark
ENTRYPOINT []
