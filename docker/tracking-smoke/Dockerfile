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
      /opt/rpx/benchmark/scripts \
      /cache/ultralytics/models \
      /cache/ultralytics/config \
      /cache/huggingface \
      /cache/torch \
      /cache/xdg

WORKDIR /opt/rpx/benchmark


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

COPY docker/tracking-smoke/requirements-yoloe.in \
     /tmp/requirements-yoloe.in

RUN /opt/rpx-envs/yoloe/bin/python -m pip install \
      -r /tmp/requirements-yoloe.in

COPY benchmark/scripts/run_yoloe_tracking_smoke.py \
     /opt/rpx/benchmark/scripts/run_yoloe_tracking_smoke.py

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
