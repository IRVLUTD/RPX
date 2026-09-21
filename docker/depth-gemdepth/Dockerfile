# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG RPX_GIT_SHA
ARG GEMDEPTH_SHA=6e259b1aeba708dbaf4abca29879c81f77f41cbe

LABEL org.opencontainers.image.title="RPX GemDepth video-depth runtime" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}" \
      org.opencontainers.image.gemdepth.revision="${GEMDEPTH_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA} \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

RUN test -n "${RPX_GIT_SHA}" && \
    git init /opt/rpx-envs/sources/gemdepth && \
    git -C /opt/rpx-envs/sources/gemdepth remote add origin \
      https://github.com/Yuecheng919/GemDepth.git && \
    git -C /opt/rpx-envs/sources/gemdepth fetch \
      --depth 1 origin "${GEMDEPTH_SHA}" && \
    git -C /opt/rpx-envs/sources/gemdepth checkout --detach FETCH_HEAD

# GemDepth publishes a fully pinned environment. Keep it isolated so its
# torch 2.3.1/cu121 stack cannot alter previously accepted model runtimes.
RUN python -m venv /opt/rpx-envs/gemdepth && \
    /opt/rpx-envs/gemdepth/bin/python -m pip install \
      --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1 && \
    /opt/rpx-envs/gemdepth/bin/python -m pip install \
      -r /opt/rpx-envs/sources/gemdepth/requirements.txt

COPY benchmark /opt/rpx/benchmark

RUN /opt/rpx-envs/gemdepth/bin/python -m pip install \
      '/opt/rpx/benchmark[hub]' && \
    /opt/rpx-envs/gemdepth/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'gemdepth-source.pth').write_text('/opt/rpx-envs/sources/gemdepth\\n')" && \
    cd /opt/rpx-envs/sources/gemdepth && \
    /opt/rpx-envs/gemdepth/bin/python -c \
      "import torch; from model.gemdepth import GemDepth; assert torch.__version__.startswith('2.3.1'); print('GemDepth imports ready:', torch.__version__)" && \
    /opt/rpx-envs/gemdepth/bin/python -m pip freeze \
      > /opt/rpx-envs/gemdepth/pip-freeze.txt

RUN /opt/rpx-envs/gemdepth/bin/python - <<'PY'
import json
from pathlib import Path

Path("/opt/rpx-envs/gemdepth/rpx-environment.json").write_text(
    json.dumps(
        {
            "container": True,
            "family": "gemdepth",
            "models": ["gem-depth"],
            "source": "/opt/rpx-envs/sources/gemdepth",
            "upstream_revision": "6e259b1aeba708dbaf4abca29879c81f77f41cbe",
            "checkpoint_repo": "YuechengLiu/GemDepth",
            "checkpoint_revision": "c901a724f2fc9a6d8764a25bec8799602c6126d6",
            "checkpoint_sha256": "f7c3fb7791862cc82684ddb1804480fb0314fddba4cf0e3b706553d98292fcad",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

WORKDIR /opt/rpx/benchmark
ENTRYPOINT []
