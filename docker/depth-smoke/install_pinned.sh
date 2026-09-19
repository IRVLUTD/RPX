#!/bin/sh
set -eu

if find /wheelhouse -maxdepth 1 -name 'torch-2.10.0+cu128-*.whl' | grep -q .; then
  exec python -m pip install --no-index --find-links /wheelhouse "$@"
fi

exec python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "$@"
