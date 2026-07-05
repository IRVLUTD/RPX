"""Regression tests for the CUDA smoke orchestration and upstream API wiring."""

from __future__ import annotations

import argparse
import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_depth_smoke_gate as gate
import run_depth_smoke_matrix as matrix
import setup_depth_smoke_env as setup_env
from depth_models.hyden import HyDen
from video_depth_models.chrono_depth import ChronoDepthAdapter
from video_depth_models.depth_crafter import DepthCrafterAdapter
from video_depth_models.monst3r import MonST3RAdapter
from video_depth_models.rolling_depth import RollingDepthAdapter
from video_depth_models.vggt_omega import VGGTOmegaAdapter
from video_depth_models.video_da import VideoDepthAnythingAdapter
from video_depth_models.vigeo import ViGeoAdapter


def _sample(t: int = 3, h: int = 16, w: int = 24) -> SimpleNamespace:
    rgb = np.arange(t * h * w * 3, dtype=np.uint8).reshape(t, h, w, 3)
    return SimpleNamespace(rgb_seq=rgb)


def test_setup_covers_every_runnable_canonical_model() -> None:
    runnable = (gate.IMAGE_MODELS | gate.VIDEO_MODELS) - gate.BLOCKED_MODELS
    assert set(setup_env.MODEL_FAMILY) == runnable
    assert setup_env.VIDEO_MODELS == gate.VIDEO_MODELS - gate.BLOCKED_MODELS
    assert set(setup_env.FAMILY_IMPORTS) == set(setup_env.FAMILY_PACKAGES)
    assert all(len(revision) == 40 for _, revision in setup_env.UPSTREAMS.values())
    assert setup_env.FAMILY_IMPORTS["unidepth2"] == ["unidepth.models"]
    assert {"matplotlib", "wandb"} <= set(setup_env.FAMILY_PACKAGES["unidepth2"])
    assert "matplotlib==3.10.0" in setup_env.FAMILY_PACKAGES["lotus2"]
    assert "transformers==4.46.3" in setup_env.FAMILY_PACKAGES["lotus2"]


def test_matrix_roster_covers_all_twenty_models() -> None:
    roster = matrix._canonical_models("all")
    assert len(roster) == 20
    assert {name for name, _task in roster} == gate.IMAGE_MODELS | gate.VIDEO_MODELS
    assert gate.BLOCKED_MODELS == {"fe2e", "d4rt", "gem-depth"}


def test_matrix_gate_parser_enforces_safe_order() -> None:
    assert matrix._parse_gates("acceptance,micro") == ["micro", "acceptance"]
    assert matrix._parse_gates("easy") == ["easy"]
    with pytest.raises(argparse.ArgumentTypeError, match="must run alone"):
        matrix._parse_gates("micro,easy")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown gate"):
        matrix._parse_gates("micro,unknown")


def test_matrix_model_filter_respects_task() -> None:
    assert matrix._select_models("image", ["depth-pro"]) == [("depth-pro", "image")]
    with pytest.raises(ValueError, match="outside --task image"):
        matrix._select_models("image", ["video-da"])


def test_matrix_resume_and_attempt_ceilings() -> None:
    assert (
        matrix._skip_reason(
            {"status": "passed", "attempts": 1},
            rerun_passed=False,
            retry_failed=False,
        )
        == "already passed; resume skip"
    )
    assert "terminal cuda_oom" in matrix._skip_reason(
        {"status": "failed", "attempts": 1, "failure_class": "cuda_oom"},
        rerun_passed=False,
        retry_failed=True,
    )
    assert "terminal access" in matrix._skip_reason(
        {"status": "failed", "attempts": 1, "failure_class": "access"},
        rerun_passed=False,
        retry_failed=True,
    )
    assert "two-attempt" in matrix._skip_reason(
        {"status": "failed", "attempts": 2, "failure_class": "dependency"},
        rerun_passed=False,
        retry_failed=True,
    )

    assert (
        matrix._setup_skip_reason(
            {"status": "failed", "attempts": 1},
            retry_failed=False,
        )
        == "prior setup failed; use --retry-failed after correcting the recipe"
    )
    assert (
        matrix._setup_skip_reason(
            {"status": "failed", "attempts": 1},
            retry_failed=True,
        )
        is None
    )
    assert "two-attempt" in matrix._setup_skip_reason(
        {"status": "failed", "attempts": 2},
        retry_failed=True,
    )


def test_gate_distinguishes_access_from_missing_weights() -> None:
    assert gate._classify_failure(RuntimeError("403 gated repo; not authorized")) == "access"
    assert gate._classify_failure(RuntimeError("404 repository not found")) == "weight_download"
    assert (
        matrix._skip_reason(
            {"status": "failed", "attempts": 1, "failure_class": "dependency"},
            rerun_passed=False,
            retry_failed=True,
        )
        is None
    )


def test_container_code_identity_uses_valid_embedded_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setenv("RPX_GIT_SHA", revision)
    monkeypatch.setattr(
        gate,
        "_run_text",
        lambda _args, cwd=None: (_ for _ in ()).throw(
            gate.subprocess.CalledProcessError(128, "git")
        ),
    )
    assert gate._code_identity(tmp_path) == (revision, False)


def test_container_code_identity_rejects_unpinned_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RPX_GIT_SHA", "latest")
    monkeypatch.setattr(
        gate,
        "_run_text",
        lambda _args, cwd=None: (_ for _ in ()).throw(
            gate.subprocess.CalledProcessError(128, "git")
        ),
    )
    with pytest.raises(gate.subprocess.CalledProcessError):
        gate._code_identity(tmp_path)


def test_matrix_environment_requires_completion_metadata(tmp_path: Path) -> None:
    env_root = tmp_path / "envs"
    python = env_root / "unidepth2" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    assert not matrix._environment_ready(env_root, "unidepth-v2")
    (env_root / "unidepth2" / "rpx-environment.json").write_text("{}")
    assert matrix._environment_ready(env_root, "unidepth-v2")


def test_matrix_state_is_atomic_and_code_specific(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    state = matrix._load_state(
        path,
        code_identity="abc",
        git_dirty=True,
        selected=[("depth-pro", "image")],
        gates=["micro"],
    )
    matrix._atomic_write_json(path, state)
    loaded = matrix._load_state(
        path,
        code_identity="abc",
        git_dirty=True,
        selected=[("depth-pro", "image")],
        gates=["micro"],
    )
    assert loaded["code_identity"] == "abc"
    assert loaded["setups"] == {}
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(RuntimeError, match="different code identity"):
        matrix._load_state(
            path,
            code_identity="different",
            git_dirty=True,
            selected=[],
            gates=["micro"],
        )


def test_setup_storage_preflight_refuses_low_space(tmp_path: Path, monkeypatch) -> None:
    usage = SimpleNamespace(total=100 * 1024**3, used=80 * 1024**3, free=20 * 1024**3)
    monkeypatch.setattr(setup_env.shutil, "disk_usage", lambda _path: usage)
    with pytest.raises(SystemExit, match="at least 30.0 GiB"):
        setup_env._storage_preflight(tmp_path, 30.0)


def test_setup_redirects_pip_cache_and_temporary_wheels(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(setup_env, "_storage_preflight", lambda _path, _minimum: 100.0)

    temporary, cache = setup_env._configure_install_storage(
        tmp_path / "envs",
        temp_dir=None,
        minimum_free_gb=30.0,
    )

    assert temporary == (tmp_path / "envs" / ".tmp").resolve()
    assert cache == (tmp_path / "envs" / ".pip-cache").resolve()
    assert setup_env.os.environ["TMPDIR"] == str(temporary)
    assert setup_env.os.environ["TEMP"] == str(temporary)
    assert setup_env.os.environ["TMP"] == str(temporary)
    assert setup_env.os.environ["PIP_CACHE_DIR"] == str(cache)


def test_setup_preflights_explicit_cache_filesystem(tmp_path: Path, monkeypatch) -> None:
    env_root = tmp_path / "envs"
    cache = tmp_path / "separate-cache"
    checked: list[Path] = []
    monkeypatch.setenv("PIP_CACHE_DIR", str(cache))
    monkeypatch.setattr(
        setup_env,
        "_storage_preflight",
        lambda path, _minimum: checked.append(path) or 100.0,
    )

    setup_env._configure_install_storage(
        env_root,
        temp_dir=tmp_path / "temporary",
        minimum_free_gb=30.0,
    )

    assert env_root.resolve() in checked
    assert cache.resolve() in checked
    assert (tmp_path / "temporary").resolve() in checked


def test_setup_verification_uses_final_json_marker() -> None:
    assert setup_env._last_json_line(
        'upstream banner\nwarning text\n{"cuda": "12.8"}\n'
    ) == {"cuda": "12.8"}
    with pytest.raises(RuntimeError, match="no JSON marker"):
        setup_env._last_json_line("")
    with pytest.raises(RuntimeError, match="without valid JSON"):
        setup_env._last_json_line("upstream banner only")

    completed = SimpleNamespace(returncode=1, stdout="banner", stderr="missing dep")
    with pytest.raises(RuntimeError, match="missing dep"):
        setup_env._verification_stdout(completed, "import check")


def test_setup_reuses_a_shared_pinned_torch_wheelhouse(tmp_path: Path) -> None:
    wheelhouse = tmp_path / ".wheelhouse"
    offline, download, install = setup_env._torch_install_commands(
        "/env/bin/python",
        wheelhouse,
        index=setup_env.TORCH_INDEX,
        torch_version=setup_env.TORCH_VERSION,
        torchvision_version=setup_env.TORCHVISION_VERSION,
    )

    assert download[:5] == [
        "/env/bin/python",
        "-m",
        "pip",
        "download",
        "--dest",
    ]
    assert str(wheelhouse) in download
    assert "--no-index" in offline
    assert offline[offline.index("--find-links") + 1] == str(wheelhouse)
    assert "--no-index" in install
    assert install[install.index("--find-links") + 1] == str(wheelhouse)
    assert f"torch=={setup_env.TORCH_VERSION}" in download
    assert f"torchvision=={setup_env.TORCHVISION_VERSION}" in install


def test_matrix_storage_preflight_accepts_one_family(tmp_path: Path, monkeypatch) -> None:
    usage = SimpleNamespace(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3)
    monkeypatch.setattr(matrix.shutil, "disk_usage", lambda _path: usage)
    matrix._check_storage(tmp_path, 30.0)


def test_gpu_occupancy_query_is_scoped_to_selected_gpu(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None):
        del cwd
        calls.append(args)
        if any("--query-gpu" in value for value in args):
            return "0, A, 100, 0, 1\n1, B, 200, 0, 1"
        return ""

    monkeypatch.setattr(gate, "_nvidia_smi", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(gate, "_run_text", fake_run)
    gpus, processes, _ = gate._gpu_state(1)
    assert len(gpus) == 2
    assert processes == []
    process_query = next(
        args for args in calls if "--query-compute-apps=pid,process_name,used_memory" in args
    )
    assert process_query[process_query.index("-i") + 1] == "1"


def test_gpu_occupancy_query_fails_closed(monkeypatch) -> None:
    def fake_run(args, *, cwd=None):
        del cwd
        if any("--query-gpu" in value for value in args):
            return "0, A, 100, 0, 1"
        raise gate.subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(gate, "_nvidia_smi", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(gate, "_run_text", fake_run)
    with pytest.raises(SystemExit, match="refusing to risk overlapping"):
        gate._gpu_state(0)


def test_depthcrafter_uses_normalized_official_pipeline_contract() -> None:
    sample = _sample()

    class Pipe:
        def __call__(self, video, **kwargs):
            assert video.dtype == np.float32
            assert 0.0 <= float(video.min()) <= float(video.max()) <= 1.0
            assert kwargs["output_type"] == "np"
            t, h, w = video.shape[:3]
            return SimpleNamespace(frames=np.ones((1, t, h, w, 3), np.float32))

    adapter = DepthCrafterAdapter(device="cpu")
    adapter._pipe = Pipe()
    depth = adapter.predict([sample])[0].depth_map_seq
    assert depth.shape == sample.rgb_seq.shape[:3]
    assert depth.dtype == np.float32


def test_chronodepth_uses_array_contract_and_restores_resolution() -> None:
    sample = _sample()

    class Pipe:
        def __call__(self, frames, **kwargs):
            assert frames.dtype == np.float32
            assert kwargs["infer_mode"] == "ours"
            return SimpleNamespace(
                frames=np.ones(
                    (frames.shape[0], 1, kwargs["height"], kwargs["width"]),
                    np.float32,
                )
            )

    adapter = ChronoDepthAdapter(device="cpu")
    adapter._pipe = Pipe()
    depth = adapter._predict_clip(sample)
    assert depth.shape == sample.rgb_seq.shape[:3]


def test_video_da_passes_numpy_sequence_to_official_helper() -> None:
    sample = _sample()

    class Model:
        def infer_video_depth(self, frames, **kwargs):
            assert isinstance(frames, np.ndarray)
            assert kwargs["input_size"] == 518
            return np.ones(frames.shape[:3], np.float32), None

    adapter = VideoDepthAnythingAdapter(device="cpu", fp16=False)
    adapter._model = Model()
    depth = adapter._predict_clip(sample)
    assert depth.shape == sample.rgb_seq.shape[:3]


def test_vigeo_uses_tensor_input_and_depth_pred_key() -> None:
    torch = pytest.importorskip("torch")
    sample = _sample()

    class Model:
        def infer(self, images, mode):
            assert images.shape == (3, 3, 16, 24)
            assert images.dtype == torch.float32
            assert 0 <= float(images.min()) <= float(images.max()) <= 1
            assert mode == "offline"
            return {"depth_pred": torch.ones((3, 1, 16, 24))}

    adapter = ViGeoAdapter(device="cpu")
    adapter._model = Model()
    depth = adapter._predict_clip(sample)
    assert depth.shape == sample.rgb_seq.shape[:3]


def test_hyden_metric_uses_point_z_and_metric_scale() -> None:
    torch = pytest.importorskip("torch")
    sample = _sample(t=1)

    class Model:
        def __call__(self, images, return_mask_and_scale):
            assert images.shape == (1, 3, 518, 518)
            assert return_mask_and_scale is True
            points = torch.zeros((1, 518, 518, 3))
            points[..., 2] = 2.0
            return {"points": points, "metric_scale": torch.tensor([3.0])}

    adapter = HyDen.__new__(HyDen)
    adapter._torch = torch
    adapter._model = Model()
    adapter._metric = True
    adapter.device = "cpu"
    depth = adapter(sample.rgb_seq[0])
    assert depth.shape == sample.rgb_seq.shape[1:3]
    assert np.allclose(depth, 6.0)


def test_vggt_uses_release_preprocessor_and_depth_key(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    sample = _sample()

    load_module = types.ModuleType("vggt.utils.load_fn")

    def load_images(paths):
        assert len(paths) == 3
        assert all(Path(path).is_file() for path in paths)
        return torch.ones((3, 3, 14, 14))

    load_module.load_and_preprocess_images = load_images
    monkeypatch.setitem(sys.modules, "vggt", types.ModuleType("vggt"))
    monkeypatch.setitem(sys.modules, "vggt.utils", types.ModuleType("vggt.utils"))
    monkeypatch.setitem(sys.modules, "vggt.utils.load_fn", load_module)

    class Model:
        def __call__(self, images):
            assert images.shape == (3, 3, 14, 14)
            return {"depth": torch.ones((1, 3, 14, 14, 1))}

    adapter = VGGTOmegaAdapter(device="cpu")
    adapter._model = Model()
    depth = adapter._predict_clip(sample)
    assert depth.shape == sample.rgb_seq.shape[:3]


def test_monst3r_reads_pred1_points_from_official_schema(monkeypatch) -> None:
    sample = _sample()
    torch_module = types.ModuleType("torch")
    inference_module = types.ModuleType("dust3r.inference")
    image_module = types.ModuleType("dust3r.utils.image")

    def load_images(paths, **kwargs):
        assert kwargs["size"] == 512
        return [{"path": path} for path in paths]

    def inference(pairs, model, device, **kwargs):
        del pairs, model, device, kwargs
        points = np.zeros((1, 8, 8, 3), np.float32)
        points[..., 2] = 2.0
        return {"pred1": {"pts3d": points}}

    image_module.load_images = load_images
    inference_module.inference = inference
    torch_module.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "dust3r", types.ModuleType("dust3r"))
    monkeypatch.setitem(sys.modules, "dust3r.utils", types.ModuleType("dust3r.utils"))
    monkeypatch.setitem(sys.modules, "dust3r.utils.image", image_module)
    monkeypatch.setitem(sys.modules, "dust3r.inference", inference_module)

    adapter = MonST3RAdapter(device="cpu")
    adapter._model = object()
    depth = adapter._predict_clip(sample)
    assert depth.shape == sample.rgb_seq.shape[:3]
    assert np.allclose(depth, 2.0)


def test_rolling_depth_uses_path_api_and_official_keyword_names(monkeypatch) -> None:
    sample = _sample()

    class Stream:
        width = 0
        height = 0
        pix_fmt = ""

        def encode(self, frame):
            del frame
            return [object()]

    class Container:
        def add_stream(self, codec, rate):
            assert (codec, rate) == ("ffv1", 30)
            return Stream()

        def mux(self, packet):
            del packet

        def close(self):
            pass

    fake_av = types.ModuleType("av")
    fake_av.open = lambda *args, **kwargs: Container()
    fake_av.VideoFrame = SimpleNamespace(from_ndarray=lambda frame, format: frame)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    class Pipe:
        def __call__(self, **kwargs):
            assert Path(kwargs["input_video_path"]).is_file()
            assert kwargs["snippet_lengths"] == [10]
            assert kwargs["restore_res"] is True
            return SimpleNamespace(depth_pred=np.ones((3, 1, 16, 24), dtype=np.float32))

    adapter = RollingDepthAdapter(device="cpu")
    adapter._pipe = Pipe()
    depth = adapter._predict_clip(sample)
    assert depth.shape == sample.rgb_seq.shape[:3]



def test_video_gate_allows_only_documented_optional_nan_metrics() -> None:
    metrics = {
        "absrel": 0.1,
        "rmse": 0.2,
        "delta1": 0.9,
        "delta2": 0.95,
        "delta3": 0.99,
        "tae": float("nan"),
        "opw": float("nan"),
        "tgm": float("nan"),
        "tcc": float("nan"),
    }
    assert set(
        gate._validate_metric_values(
            metrics,
            task="video",
            context="test",
        )
    ) == {"tae", "opw", "tgm", "tcc"}

    bad_core = dict(metrics, absrel=float("nan"))
    with pytest.raises(RuntimeError, match="absrel"):
        gate._validate_metric_values(
            bad_core,
            task="video",
            context="test",
        )

    bad_unknown = dict(metrics, unexpected=float("nan"))
    with pytest.raises(RuntimeError, match="unexpected"):
        gate._validate_metric_values(
            bad_unknown,
            task="video",
            context="test",
        )
