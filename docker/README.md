# `docker/` — Reproducible RPX environment

> Top-level Dockerised environment that bundles the RealSense SDK,
> the benchmark toolkit, and the mask-generation pipeline.
> One image, three workflows.

Use this when you don't want to deal with the
[librealsense 2.47.0 pin](../data_capture/README.md#%EF%B8%8F-requirements)
on your host, or when you need a CUDA-ready box for
[`mask_pipeline/`](../mask_pipeline/README.md).

## Contents

- [Quick start](#quick-start)
- [Workflows](#workflows)
- [Helper scripts](#helper-scripts)

---

## Quick start

```bash
# 1. Build once (slow first time, cached after).
./build_docker_image.sh

# 2. Export ROS env vars if you'll run ROS inside the container.
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost

# 3. Start detached (default), then exec in:
./start_docker.sh
./enter_docker.sh

# Stop when done.
./stop_docker.sh
```

For an interactive (foreground) session use `./start_docker.sh -i`.

## Workflows

| Workflow | Command |
|---|---|
| **Run a benchmark** inside the container | exec in → `cd benchmark && python scripts/run_depth.py …` |
| **Run mask-pipeline** inside the container | exec in → `cd /workspace/mask_pipeline && python -m maskgen_pipeline.interactive_gsam2 …` |
| **Data capture** (host preferred) | The pinned librealsense often works better directly on the host. See [`../data_capture/`](../data_capture/README.md). |

## Helper scripts

| Script | What it does |
|---|---|
| `build_docker_image.sh` | One-time image build. |
| `start_docker.sh` | Launch detached (or `-i` for foreground). |
| `enter_docker.sh` | Exec into a running container. |
| `stop_docker.sh` | Stop + remove the running container. |
| `dockerfile` | Image definition (Ubuntu + ROS + CUDA + RealSense SDK). |
| `docker-compose.yml` | Compose variant for multi-container setups. |
