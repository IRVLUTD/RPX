# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9

ARG RPX_GIT_SHA
ARG GENCEPTION_SHA=a47e55120cc2b00027c19c9f1937831e77541056

LABEL org.opencontainers.image.title="RPX GenCeption runtime" \
      org.opencontainers.image.description="GenCeption 1.3B and 14B RPX benchmark runtime" \
      org.opencontainers.image.source="https://github.com/IRVLUTD/RPX" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    RPX_CACHE_DIR=/cache/rpx \
    RPX_GENCEPTION_ROOT=/models/genception \
    RPX_GENCEPTION_TEXT_DEVICE=cpu \
    RPX_GIT_SHA=${RPX_GIT_SHA} \
    PYTHONPATH=/opt/rpx/benchmark:/opt/rpx/benchmark/scripts \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN test -n "${RPX_GIT_SHA}" && \
    apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      'jax[cuda12]==0.11.2' \
      flax==0.12.10 \
      kauldron==1.4.4 \
      ml_dtypes==0.6.0 \
      mediapy==1.2.7 \
      safetensors==0.8.0 \
      typeguard==4.4.4 \
      transformers==4.57.6 \
      accelerate==1.15.0 \
      sentencepiece==0.2.1 \
      pyarrow==21.0.0 \
      pandas==2.3.3 \
      scipy==1.16.3 \
      'huggingface_hub[hf_xet]==0.36.0'

RUN git clone https://github.com/google-deepmind/representations4d.git \
      /opt/rpx-models/representations4d && \
    git -C /opt/rpx-models/representations4d checkout "${GENCEPTION_SHA}" && \
    python -m pip install --no-cache-dir --no-deps /opt/rpx-models/representations4d && \
    git clone https://github.com/JonathonLuiten/TrackEval.git /opt/rpx-models/TrackEval && \
    git -C /opt/rpx-models/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a && \
    python -m pip install --no-cache-dir --no-deps /opt/rpx-models/TrackEval

COPY benchmark /opt/rpx/benchmark

RUN python -m pip install --no-cache-dir --no-deps /opt/rpx/benchmark && \
    python -c "import jax,representations4d,torch,trackeval,transformers; assert jax.__version__ == '0.11.2'; print('RPX GenCeption runtime ready')"

COPY --chmod=0755 docker/genception/entrypoint.sh /usr/local/bin/rpx-genception

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/cache/rpx", "/models/genception", "/outputs"]
ENTRYPOINT ["/usr/local/bin/rpx-genception"]
CMD ["help"]
