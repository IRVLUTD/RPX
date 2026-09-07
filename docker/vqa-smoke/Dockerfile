# syntax=docker/dockerfile:1.7

FROM vllm/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635 AS vqa_vllm

ARG RPX_GIT_SHA

LABEL org.opencontainers.image.title="RPX VQA vLLM runtime" \
      org.opencontainers.image.description="vLLM-only bbox VQA runtime for the RPX ten-model roster" \
      org.opencontainers.image.source="https://github.com/IRVLUTD/RPX" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    RPX_VQA_CACHE=/cache/rpx-vqa \
    RPX_VQA_OUTPUTS=/outputs \
    RPX_GIT_SHA=${RPX_GIT_SHA} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN test -n "${RPX_GIT_SHA}" && \
    python -m pip install --no-cache-dir pyarrow==21.0.0

COPY benchmark /opt/rpx/benchmark

RUN python -m pip install --no-cache-dir --no-deps /opt/rpx/benchmark && \
    python -c "import pyarrow,vllm; assert vllm.__version__ == '0.28.0'; print('RPX vLLM runtime ready')"

COPY --chmod=0755 docker/vqa-smoke/entrypoint.sh /usr/local/bin/rpx-vqa-smoke

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/cache/rpx-vqa", "/outputs"]
ENTRYPOINT ["/usr/local/bin/rpx-vqa-smoke"]
CMD ["help"]
