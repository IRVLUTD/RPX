# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm@sha256:961f57b66f6aee85ae11272cca9219123a860ab20e2a09a1238f57e0df495825 AS vqa_base

ARG RPX_GIT_SHA

LABEL org.opencontainers.image.title="RPX VQA smoke runtime" \
      org.opencontainers.image.description="Cumulative single-image VQA smoke environment" \
      org.opencontainers.image.source="https://github.com/IRVLUTD/RPX" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    RPX_VQA_CACHE=/cache/rpx-vqa \
    RPX_VQA_OUTPUTS=/outputs \
    RPX_GIT_SHA=${RPX_GIT_SHA} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN test -n "${RPX_GIT_SHA}" && \
    apt-get update && \
    apt-get install -y --no-install-recommends bash ca-certificates git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rpx


FROM vqa_base AS vqa_paligemma

COPY docker/vqa-smoke/requirements-paligemma.lock /tmp/requirements-paligemma.lock

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip==25.3 && \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.10.0+cu128 && \
    python -m pip install -r /tmp/requirements-paligemma.lock

COPY benchmark /opt/rpx/benchmark

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps /opt/rpx/benchmark && \
    python -c "import accelerate,torch,transformers; from transformers import PaliGemmaForConditionalGeneration; assert torch.__version__ == '2.10.0+cu128'; assert transformers.__version__ == '4.57.6'; print('PaliGemma 2 runtime imports ready')"

COPY --chmod=0755 docker/vqa-smoke/entrypoint.sh /usr/local/bin/rpx-vqa-smoke

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/cache/rpx-vqa", "/outputs"]
ENTRYPOINT ["/usr/local/bin/rpx-vqa-smoke"]
CMD ["help"]


FROM vqa_base AS vqa_gemma3

COPY docker/vqa-smoke/requirements-paligemma.lock /tmp/requirements-vqa.lock

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip==25.3 && \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.10.0+cu128 && \
    python -m pip install -r /tmp/requirements-vqa.lock pyarrow==21.0.0

COPY benchmark /opt/rpx/benchmark

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps /opt/rpx/benchmark && \
    python -c "import pyarrow,torch,transformers; from transformers import Gemma3ForConditionalGeneration; print('Gemma 3 runtime imports ready')"

COPY --chmod=0755 docker/vqa-smoke/entrypoint.sh /usr/local/bin/rpx-vqa-smoke

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/cache/rpx-vqa", "/outputs"]
ENTRYPOINT ["/usr/local/bin/rpx-vqa-smoke"]
CMD ["help"]
