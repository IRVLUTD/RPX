# syntax=docker/dockerfile:1.7

FROM vllm/vllm-openai@sha256:50509e700235cea487715cedeb501d20a1cd15fa6a54ce93688284bd0d96995d AS vqa_vllm

ARG RPX_GIT_SHA

LABEL org.opencontainers.image.title="RPX VQA runtime" \
      org.opencontainers.image.description="Pinned vLLM and native grounding runtimes" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    RPX_VQA_CACHE=/cache/rpx-vqa \
    RPX_VQA_OUTPUTS=/outputs \
    RPX_GIT_SHA=${RPX_GIT_SHA} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN test -n "${RPX_GIT_SHA}" && \
    python3 -m pip install --no-cache-dir \
      pyarrow==21.0.0 \
      sentencepiece==0.2.1 \
      einops==0.8.1 \
      timm==1.0.22

# Florence-2 and native PaliGemma 2 are validated with Transformers 4.49,
# while Qwen3-VL needs the vLLM image's newer Transformers runtime. Keep both
# dependency sets isolated and select the interpreter at entrypoint.
RUN python3 -m venv --system-site-packages /opt/rpx-envs/florence2 && \
    /opt/rpx-envs/florence2/bin/python -m pip install --no-cache-dir \
      transformers==4.49.0 \
      'huggingface-hub>=0.26,<1.0' \
      'tokenizers>=0.21,<0.22' && \
    /opt/rpx-envs/florence2/bin/python -c \
      "import torch,transformers; assert transformers.__version__ == '4.49.0'; print('Florence-2 runtime ready')"

# InternVL 3.5 grounding is run through its official Transformers chat and
# dynamic-tiling path, isolated from vLLM so native bbox serialization and
# multi-image ordering are preserved exactly.
RUN python3 -m venv --system-site-packages /opt/rpx-envs/internvl && \
    /opt/rpx-envs/internvl/bin/python -m pip install --no-cache-dir \
      transformers==4.57.1 \
      accelerate==1.10.1 \
      'huggingface-hub>=0.34,<1.0' \
      'tokenizers>=0.22,<0.23' && \
    /opt/rpx-envs/internvl/bin/python -c \
      "import accelerate,torch,torchvision,transformers; assert transformers.__version__ == '4.57.1'; print('InternVL runtime ready')"

COPY benchmark /opt/rpx/benchmark

RUN python3 -m pip install --no-cache-dir --no-deps /opt/rpx/benchmark && \
    /opt/rpx-envs/florence2/bin/python -m pip install --no-cache-dir --no-deps /opt/rpx/benchmark && \
    /opt/rpx-envs/internvl/bin/python -m pip install --no-cache-dir --no-deps /opt/rpx/benchmark && \
    python3 -c "import einops,pyarrow,sentencepiece,timm,transformers,vllm; assert vllm.__version__ == '0.28.0'; assert sentencepiece.__version__ == '0.2.1'; print('RPX VQA runtime ready')"

COPY --chmod=0755 docker/vqa-smoke/entrypoint.sh /usr/local/bin/rpx-vqa-smoke

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/cache/rpx-vqa", "/outputs"]
ENTRYPOINT ["/usr/local/bin/rpx-vqa-smoke"]
CMD ["help"]
