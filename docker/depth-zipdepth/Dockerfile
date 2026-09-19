# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=vndhiran123/rpx-depth-smoke@sha256:3b86de0e4e136587d6ea7d816a482af15cb115dc6a14f16ea8f7d9356ab220cc
FROM ${BASE_IMAGE}

ARG RPX_GIT_SHA
ARG ZIPDEPTH_SHA=94da7527f7030a0e79d54f33b113bdce4065d735

LABEL org.opencontainers.image.title="RPX official ZipDepth runtime" \
      org.opencontainers.image.source="https://github.com/IRVLUTD/RPX" \
      org.opencontainers.image.revision="${RPX_GIT_SHA}" \
      org.opencontainers.image.zipdepth.revision="${ZIPDEPTH_SHA}"

ENV RPX_GIT_SHA=${RPX_GIT_SHA} \
    RPX_ZIPDEPTH_SOURCE=/opt/rpx-envs/sources/zipdepth \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY benchmark /opt/rpx/benchmark

RUN test -n "${RPX_GIT_SHA}" && \
    mkdir -p /opt/rpx-envs/sources && \
    git init "${RPX_ZIPDEPTH_SOURCE}" && \
    git -C "${RPX_ZIPDEPTH_SOURCE}" remote add origin \
      https://github.com/fabiotosi92/ZipDepth.git && \
    git -C "${RPX_ZIPDEPTH_SOURCE}" fetch --depth 1 origin "${ZIPDEPTH_SHA}" && \
    git -C "${RPX_ZIPDEPTH_SOURCE}" checkout --detach FETCH_HEAD && \
    test "$(git -C "${RPX_ZIPDEPTH_SOURCE}" rev-parse HEAD)" = "${ZIPDEPTH_SHA}" && \
    echo "a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65  ${RPX_ZIPDEPTH_SOURCE}/checkpoints/zipdepth_base.pth" | \
      sha256sum -c -

# ZipDepth needs only ordinary PyTorch/OpenCV operators. Reuse the cumulative
# image's CUDA-enabled PyTorch ABI and keep the official package isolated.
RUN python -m venv --system-site-packages /opt/rpx-envs/zipdepth && \
    /opt/rpx-envs/zipdepth/bin/python -m pip install --upgrade \
      pip==25.3 setuptools wheel && \
    /opt/rpx-envs/zipdepth/bin/python -m pip install --no-deps \
      "${RPX_ZIPDEPTH_SOURCE}" && \
    /opt/rpx-envs/zipdepth/bin/python -m pip install --no-deps --force-reinstall \
      /opt/rpx/benchmark && \
    /opt/rpx-envs/zipdepth/bin/python -c \
      "import pathlib,site; pathlib.Path(site.getsitepackages()[0], 'rpx-zipdepth-source.pth').write_text('/opt/rpx-envs/sources/zipdepth\n/opt/rpx/benchmark/scripts\n')"

RUN /opt/rpx-envs/zipdepth/bin/python -c \
      "import cv2,numpy,torch; from zipdepth.inference.predictor import DepthInference; from depth_models.zipdepth import ZipDepthAdapter; assert ZipDepthAdapter.UPSTREAM_REVISION == '${ZIPDEPTH_SHA}'; predictor=DepthInference(checkpoint_path='${RPX_ZIPDEPTH_SOURCE}/checkpoints/zipdepth_base.pth', device='cpu', warmup_iters=0); assert predictor.model is not None; print('official ZipDepth imports and checkpoint ready', torch.__version__)" && \
    /opt/rpx-envs/zipdepth/bin/python -m pip freeze \
      > /opt/rpx-envs/zipdepth/pip-freeze.txt && \
    /opt/rpx-envs/zipdepth/bin/python -c \
      "import json,pathlib,sys; pathlib.Path('/opt/rpx-envs/zipdepth/rpx-environment.json').write_text(json.dumps({'container': True, 'family': 'zipdepth', 'models': ['zipdepth'], 'python': sys.executable, 'rpx_git_sha': '${RPX_GIT_SHA}', 'upstream_revision': '${ZIPDEPTH_SHA}', 'checkpoint': 'checkpoints/zipdepth_base.pth', 'checkpoint_sha256': 'a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65'}, indent=2)+'\n')"

WORKDIR /opt/rpx/benchmark
VOLUME ["/cache/huggingface", "/outputs"]
ENTRYPOINT []
CMD ["/opt/rpx-envs/zipdepth/bin/python", "scripts/run_depth.py", "--model", "zipdepth", "--help"]
