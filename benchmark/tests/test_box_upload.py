"""Lock the Box upload contract.

We can't talk to real Box in CI (token expires every 60 min), but we
can prove ``upload_tree`` produces the right (folder-id, file-bytes)
sequence by mocking the Box REST API. The contract this test pins:

1. Every file under ``local_dir`` ends up at the matching path under the
   declared ``remote_path`` on Box.
2. The scene/phase tree of predictions is preserved exactly (so a Box
   browser sees ``monocular_depth/<model>/<split>/<scene>/<phase>/<frame>.npz``,
   matching the local layout the user asked for).
3. Idempotency: running the upload a second time with identical local
   bytes results in zero new uploads.
4. Top-level metadata (``result.json``, ``summary.md``,
   ``comprehensive_metrics.json``) lands at the run-folder root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package; add it so we can import box_fetch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ────────────────────────────  Mock Box state  ───────────────────────────


class _MockBox:
    """In-memory Box.

    Folders are dicts: ``{folder_id: {"name": str, "parent": str, "items": {child_name: child_id}}}``.
    Files are dicts:   ``{file_id:   {"name": str, "parent": str, "size": int}}``.

    Generates fresh ids on each create call. Records every upload so
    tests can assert on the resulting tree shape and bytes.
    """

    def __init__(self):
        self._next_id = 1
        self.folders: dict = {"0": {"name": "All Files", "parent": None, "items": {}}}
        self.files: dict = {}
        self.uploads: list = []  # records (file_id, name, parent, bytes)

    def _new_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    # ── /folders/{id}/items
    def list_items(self, folder_id: str) -> list[dict]:
        f = self.folders.get(str(folder_id))
        if f is None:
            return []
        out = []
        for name, child_id in f["items"].items():
            if child_id in self.folders:
                out.append({"id": child_id, "type": "folder", "name": name, "size": 0})
            elif child_id in self.files:
                out.append(
                    {
                        "id": child_id,
                        "type": "file",
                        "name": name,
                        "size": self.files[child_id]["size"],
                    }
                )
        return out

    # ── POST /folders
    def create_folder(self, name: str, parent_id: str) -> dict:
        parent = self.folders.get(str(parent_id))
        if parent is None:
            raise AssertionError(f"parent {parent_id} doesn't exist in mock")
        if name in parent["items"]:
            # 409-conflict-style payload — upload_tree handles this
            existing_id = parent["items"][name]
            raise _MockConflict({"id": existing_id, "name": name, "parent": parent_id})
        new_id = self._new_id()
        self.folders[new_id] = {"name": name, "parent": parent_id, "items": {}}
        parent["items"][name] = new_id
        return {"id": new_id, "name": name, "parent": {"id": parent_id}}

    # ── upload (multipart)
    def upload_file(self, name: str, parent_id: str, payload: bytes) -> dict:
        parent = self.folders.get(str(parent_id))
        if parent is None:
            raise AssertionError(f"parent {parent_id} doesn't exist in mock")
        if name in parent["items"]:
            # New version of existing file — keep id, update bytes
            file_id = parent["items"][name]
            self.files[file_id]["size"] = len(payload)
        else:
            file_id = self._new_id()
            self.files[file_id] = {"name": name, "parent": parent_id, "size": len(payload)}
            parent["items"][name] = file_id
        self.uploads.append((file_id, name, parent_id, len(payload)))
        return {
            "entries": [
                {"id": file_id, "name": name, "parent": {"id": parent_id}, "size": len(payload)}
            ]
        }

    # ── path helpers used by tests
    def resolve_path(self, *parts: str) -> str | None:
        cur = "0"
        for p in parts:
            f = self.folders.get(cur)
            if f is None or p not in f["items"]:
                return None
            cur = f["items"][p]
        return cur


class _MockConflict(Exception):
    """Mimics Box's 409 conflict response payload."""

    def __init__(self, ctx):
        self.ctx = ctx


# ────────────────────────────  HTTP-level mocks  ─────────────────────────


def _mock_api_get(box: _MockBox):
    def _get(path, params=None):
        if path.startswith("/folders/") and path.endswith("/items"):
            folder_id = path.split("/")[2]
            return {
                "entries": box.list_items(folder_id),
                "total_count": len(box.list_items(folder_id)),
            }
        raise NotImplementedError(f"mock _api_get doesn't handle {path}")

    return _get


def _mock_api_post(box: _MockBox):
    """Mimic the real _api_post — on a name conflict, raise an
    HTTPError-shaped exception with status_code=409 so the caller's
    409-resolver path runs."""
    import requests

    def _post(path, payload, base=None):
        if path == "/folders":
            try:
                return box.create_folder(payload["name"], payload["parent"]["id"])
            except _MockConflict as conflict:
                # Build a synthetic Response with status_code=409 so
                # `_api_ensure_folder`'s `except requests.HTTPError`
                # branch sees what it expects.
                resp = requests.Response()
                resp.status_code = 409
                raise requests.HTTPError("409 conflict", response=resp) from conflict
        raise NotImplementedError(f"mock _api_post doesn't handle {path}")

    return _post


def _mock_upload_file(box: _MockBox):
    """Mirror box_fetch._api_upload_file's signature.

    Honours the size-matched-skip contract: if `existing[name]` has the
    same size as the local file, return ``{"skipped": True}`` without
    re-uploading bytes.
    """

    def _upload(local_path: Path, parent_folder_id: str, name=None, existing=None) -> dict:
        name = name or local_path.name
        size = local_path.stat().st_size
        if existing and name in existing and int(existing[name].get("size") or -1) == size:
            return {"id": existing[name]["id"], "name": name, "skipped": True}
        result = box.upload_file(
            name=name,
            parent_id=parent_folder_id,
            payload=local_path.read_bytes(),
        )
        entry = result["entries"][0]
        return {"id": entry["id"], "name": name, "skipped": False}

    return _upload


# ────────────────────────────  Fixtures  ─────────────────────────────────


@pytest.fixture
def fake_run_dir(tmp_path):
    """Build a fake run output dir mirroring run_depth.py's output:
    result.json + summary.md + comprehensive_metrics.json +
    predictions/<scene>/<phase>/<frame>.npz."""
    out = tmp_path / "rpx_results" / "ZoeDepth_NK" / "easy"
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps({"task": "monocular_depth"}))
    (out / "summary.md").write_text("# summary")
    (out / "comprehensive_metrics.json").write_text(json.dumps({"alignment": "none"}))

    # Two scenes × two phases × two frames each = 8 prediction files.
    for scene in ("scene_alpha.foo.bar", "scene_beta.baz.qux"):
        for phase in (0, 1):
            for frame in ("00000", "00001"):
                p = out / "predictions" / scene / str(phase) / f"{frame}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x00" * 1024)  # 1 KB per "depth"
    return out


@pytest.fixture
def mock_box(monkeypatch):
    box = _MockBox()
    from rpx_benchmark import box_upload  # noqa: PLC0415

    # Pretend a token is present so the auth path runs.
    monkeypatch.setattr(box_upload, "_token", lambda: "mock-token-XXXX")
    monkeypatch.setattr(box_upload, "_api_get", _mock_api_get(box))
    monkeypatch.setattr(box_upload, "_api_post", _mock_api_post(box))
    monkeypatch.setattr(box_upload, "_api_upload_file", _mock_upload_file(box))
    return box


# ────────────────────────────  Contract tests  ───────────────────────────


def test_upload_tree_mirrors_scene_phase_layout(fake_run_dir, mock_box):
    """The exact tree on Box must match the user-asked
    ``monocular_depth/<model>/<split>/<scene>/<phase>/<frame>.npz`` shape."""
    from rpx_benchmark.box_upload import upload_tree

    summary = upload_tree(
        fake_run_dir,
        remote_path="monocular_depth/ZoeDepth_NK/easy",
        root_folder_id="0",
        verbose=False,
    )

    # 1. Total upload count = 11 (3 metadata + 8 predictions).
    assert summary["uploaded"] == 11
    assert summary["bytes_uploaded"] >= 8 * 1024  # at least the prediction bytes

    # 2. Top-level metadata at <root>/monocular_depth/ZoeDepth_NK/easy/.
    for name in ("result.json", "summary.md", "comprehensive_metrics.json"):
        fid = mock_box.resolve_path("monocular_depth", "ZoeDepth_NK", "easy", name)
        assert fid is not None, f"missing {name} on Box"

    # 3. Predictions preserve scene/phase/frame layout.
    for scene in ("scene_alpha.foo.bar", "scene_beta.baz.qux"):
        for phase in ("0", "1"):
            for frame in ("00000.npz", "00001.npz"):
                fid = mock_box.resolve_path(
                    "monocular_depth",
                    "ZoeDepth_NK",
                    "easy",
                    "predictions",
                    scene,
                    phase,
                    frame,
                )
                assert fid is not None, f"missing {scene}/{phase}/{frame}"


def test_upload_tree_idempotent_on_rerun(fake_run_dir, mock_box):
    """Re-running upload_tree against the same local dir uploads zero
    new files (every file's Box copy already has the matching size)."""
    from rpx_benchmark.box_upload import upload_tree

    s1 = upload_tree(fake_run_dir, "monocular_depth/M/easy", root_folder_id="0", verbose=False)
    n_uploads_first = len(mock_box.uploads)
    s2 = upload_tree(fake_run_dir, "monocular_depth/M/easy", root_folder_id="0", verbose=False)
    n_uploads_second = len(mock_box.uploads)

    assert s1["uploaded"] == 11
    assert s2["uploaded"] == 0, f"second upload re-uploaded {s2['uploaded']} files"
    assert n_uploads_second == n_uploads_first, (
        f"second run hit upload endpoint {n_uploads_second - n_uploads_first} extra times"
    )
    assert s2["skipped"] == 11


# ───────────────  upload_run_dir + TaskRunConfig wiring  ─────────────────


def test_upload_run_dir_canonical_path(fake_run_dir, mock_box):
    """``upload_run_dir`` always lays the canonical
    ``<task>/<safe_model_name>/<split>/`` path under root_folder_id."""
    from rpx_benchmark.box_upload import upload_run_dir

    out = upload_run_dir(
        fake_run_dir,
        task="monocular_depth",
        model_name="ZoeDepth_NK",
        split="easy",
        root_folder_id="0",
        verbose=False,
    )
    assert out["remote_path"] == "monocular_depth/ZoeDepth_NK/easy"
    # Every file made it (3 metadata + 8 predictions).
    assert out["uploaded"] == 11


def test_upload_run_dir_escapes_slash_in_model_name(fake_run_dir, mock_box):
    """HF-style ``namespace/model`` names must collapse to one Box
    folder, not nested folders that would split runs apart."""
    from rpx_benchmark.box_upload import upload_run_dir

    out = upload_run_dir(
        fake_run_dir,
        task="monocular_depth",
        model_name="Intel/zoedepth-nyu-kitti",
        split="hard",
        root_folder_id="0",
        verbose=False,
    )
    assert out["remote_path"] == "monocular_depth/Intel__zoedepth-nyu-kitti/hard"


def test_token_required_raises_with_clear_hint(monkeypatch):
    """Lazy token read must give a ConfigError that names the env var
    rather than letting requests emit an auth-header error."""
    from rpx_benchmark.box_upload import _token
    from rpx_benchmark.exceptions import ConfigError

    monkeypatch.delenv("BOX_DEVELOPER_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="BOX_DEVELOPER_TOKEN"):
        _token()


def test_task_run_config_defaults_box_off():
    """Library-default must not surprise BYO users with a Box upload."""
    from rpx_benchmark.adapters import BenchmarkableModel
    from rpx_benchmark.tasks._pipeline import TaskRunConfig

    class _StubModel(BenchmarkableModel):
        def __init__(self):
            self.name = "stub"
            self.model = None

        def setup(self):
            pass

        def predict(self, batch):
            return []

    cfg = TaskRunConfig(model=_StubModel(), split="easy")
    assert cfg.upload_to_box is False
    assert cfg.box_folder_id is None


def test_task_run_config_accepts_box_flags():
    """Setting the flags should round-trip on the config dataclass."""
    from rpx_benchmark.adapters import BenchmarkableModel
    from rpx_benchmark.tasks._pipeline import TaskRunConfig

    class _StubModel(BenchmarkableModel):
        def __init__(self):
            self.name = "stub"
            self.model = None

        def setup(self):
            pass

        def predict(self, batch):
            return []

    cfg = TaskRunConfig(
        model=_StubModel(),
        split="easy",
        upload_to_box=True,
        box_folder_id="123456",
    )
    assert cfg.upload_to_box is True
    assert cfg.box_folder_id == "123456"


# ─────────────────────────  OAuth resolver tests  ────────────────────────
#
# Cover the post-fetch auth path: tokens on disk → refresh if expired →
# return access token; fallback to BOX_DEVELOPER_TOKEN; raise when neither
# is configured. The interactive `oauth_login` is pragma:no-cover (it
# spawns a browser + HTTPServer).


@pytest.fixture
def _isolated_tokens_path(tmp_path, monkeypatch):
    """Redirect the OAuth tokens file to a temp dir + clear all Box env vars."""
    from rpx_benchmark import box_upload

    fake_path = tmp_path / "box_tokens.json"
    monkeypatch.setattr(box_upload, "TOKENS_PATH", fake_path)
    for var in ("BOX_DEVELOPER_TOKEN", "BOX_CLIENT_ID", "BOX_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    return fake_path


def test_tokens_roundtrip_persists_only_rotating_fields(_isolated_tokens_path):
    """_stash_tokens_from_response writes ONLY the rotating fields — no client
    creds on disk (those stay in env vars)."""
    import time as _time

    from rpx_benchmark import box_upload

    resp = {
        "access_token": "AT-1",
        "refresh_token": "RT-1",
        "expires_in": 3600,
    }
    t0 = _time.time()
    persisted = box_upload._stash_tokens_from_response(resp)

    assert persisted["access_token"] == "AT-1"
    assert persisted["refresh_token"] == "RT-1"
    assert persisted["expires_at"] >= t0 + 3500
    assert "client_id" not in persisted
    assert "client_secret" not in persisted

    # And the on-disk JSON matches what we got back
    loaded = box_upload._load_tokens()
    assert loaded == persisted


def test_load_tokens_returns_none_when_file_missing(_isolated_tokens_path):
    from rpx_benchmark import box_upload

    assert box_upload._load_tokens() is None


def test_load_tokens_returns_none_on_corrupt_json(_isolated_tokens_path):
    from rpx_benchmark import box_upload

    _isolated_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    _isolated_tokens_path.write_text("{not valid json")
    assert box_upload._load_tokens() is None


def test_token_prefers_oauth_when_unexpired(_isolated_tokens_path):
    """Stored OAuth tokens that aren't near expiry are returned as-is — no
    refresh call, no env lookups."""
    import time as _time

    from rpx_benchmark import box_upload

    box_upload._save_tokens(
        {"access_token": "OA-FRESH", "refresh_token": "RT", "expires_at": _time.time() + 9999}
    )
    assert box_upload._token() == "OA-FRESH"


def test_token_falls_back_to_dev_token_when_no_oauth(_isolated_tokens_path, monkeypatch):
    """No OAuth tokens on disk → BOX_DEVELOPER_TOKEN is the answer."""
    from rpx_benchmark import box_upload

    monkeypatch.setenv("BOX_DEVELOPER_TOKEN", "DEV-XYZ")
    assert box_upload._token() == "DEV-XYZ"


def test_token_raises_when_nothing_configured(_isolated_tokens_path):
    """Neither OAuth tokens nor dev token → ConfigError."""
    from rpx_benchmark import box_upload
    from rpx_benchmark.exceptions import ConfigError

    with pytest.raises(ConfigError):
        box_upload._token()


def test_token_refreshes_when_near_expiry(_isolated_tokens_path, monkeypatch):
    """Expired OAuth access token + creds in env → calls refresh, persists
    new tokens, returns the new access token."""
    import time as _time

    from rpx_benchmark import box_upload

    box_upload._save_tokens(
        {"access_token": "OA-STALE", "refresh_token": "RT-OLD", "expires_at": _time.time() - 60}
    )
    monkeypatch.setenv("BOX_CLIENT_ID", "CID")
    monkeypatch.setenv("BOX_CLIENT_SECRET", "CSEC")

    captured = {}

    def _fake_refresh(refresh_token, client_id, client_secret):
        captured["refresh_token"] = refresh_token
        captured["client_id"] = client_id
        captured["client_secret"] = client_secret
        return {"access_token": "OA-NEW", "refresh_token": "RT-NEW", "expires_in": 3600}

    monkeypatch.setattr(box_upload, "_refresh_access_token", _fake_refresh)

    assert box_upload._token() == "OA-NEW"
    assert captured == {"refresh_token": "RT-OLD", "client_id": "CID", "client_secret": "CSEC"}

    on_disk = box_upload._load_tokens()
    assert on_disk["access_token"] == "OA-NEW"
    assert on_disk["refresh_token"] == "RT-NEW"
    assert "client_id" not in on_disk
    assert "client_secret" not in on_disk


def test_token_expired_oauth_without_env_raises(_isolated_tokens_path):
    """Expired OAuth tokens + no client creds in env → clear ConfigError."""
    import time as _time

    from rpx_benchmark import box_upload
    from rpx_benchmark.exceptions import ConfigError

    box_upload._save_tokens(
        {"access_token": "OA-STALE", "refresh_token": "RT", "expires_at": _time.time() - 1}
    )
    with pytest.raises(ConfigError) as excinfo:
        box_upload._token()
    assert "client_id" in str(excinfo.value).lower()


def test_refresh_access_token_unwraps_response(monkeypatch):
    """Happy path: HTTP 200 with a JSON body → dict is returned verbatim."""
    from rpx_benchmark import box_upload

    class _R:
        ok = True
        status_code = 200

        def json(self):
            return {"access_token": "A", "refresh_token": "B", "expires_in": 3600}

    monkeypatch.setattr(box_upload.requests, "post", lambda *a, **k: _R())
    out = box_upload._refresh_access_token("RT", "CID", "CSEC")
    assert out == {"access_token": "A", "refresh_token": "B", "expires_in": 3600}


def test_refresh_access_token_raises_on_http_error(monkeypatch):
    """Box returns 400 / 401 → ConfigError that points the user at `login`."""
    from rpx_benchmark import box_upload
    from rpx_benchmark.exceptions import ConfigError

    class _R:
        ok = False
        status_code = 400
        text = "invalid_grant"

        def json(self):
            return {}

    monkeypatch.setattr(box_upload.requests, "post", lambda *a, **k: _R())
    with pytest.raises(ConfigError) as excinfo:
        box_upload._refresh_access_token("RT", "CID", "CSEC")
    assert "login" in str(excinfo.value).lower()
