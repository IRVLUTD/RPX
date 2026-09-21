# syntax=docker/dockerfile:1.7

# Immutable cumulative HyDen image published for docker/depth-all-hyden.
FROM rpx-depth-smoke@sha256:3b86de0e4e136587d6ea7d816a482af15cb115dc6a14f16ea8f7d9356ab220cc

ARG RPX_GIT_SHA

LABEL org.opencontainers.image.title="RPX DA-V2 Large D1-F paper runtime" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}" \
      org.opencontainers.image.base.digest="sha256:3b86de0e4e136587d6ea7d816a482af15cb115dc6a14f16ea8f7d9356ab220cc"

ENV RPX_GIT_SHA=${RPX_GIT_SHA} \
    RPX_DOCKER_DIGEST=sha256:3b86de0e4e136587d6ea7d816a482af15cb115dc6a14f16ea8f7d9356ab220cc \
    HF_HOME=/cache/huggingface \
    RPX_CACHE_DIR=/cache/huggingface/rpx-resolved \
    PYTHONUNBUFFERED=1

COPY benchmark /opt/rpx/benchmark

RUN test -n "${RPX_GIT_SHA}" && \
    python -m pip install --no-deps --force-reinstall /opt/rpx/benchmark && \
    python -c "import numpy, scipy, pyarrow; from rpx_benchmark.metrics.depth_paper import D1_CALIBRATION; assert (D1_CALIBRATION.width, D1_CALIBRATION.height) == (640, 480); print('paper imports ready', numpy.__version__, scipy.__version__, pyarrow.__version__)"

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/outputs"]
ENTRYPOINT []
CMD ["python", "scripts/run_depth_paper.py", "--help"]
