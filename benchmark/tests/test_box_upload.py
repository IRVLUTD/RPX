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
import os
import sys
from pathlib import Path
from unittest import mock

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
        self.files:   dict = {}
        self.uploads: list = []   # records (file_id, name, parent, bytes)

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
                out.append({"id": child_id, "type": "file", "name": name,
                            "size": self.files[child_id]["size"]})
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
            self.files[file_id] = {"name": name, "parent": parent_id,
                                    "size": len(payload)}
            parent["items"][name] = file_id
        self.uploads.append((file_id, name, parent_id, len(payload)))
        return {"entries": [{"id": file_id, "name": name,
                              "parent": {"id": parent_id}, "size": len(payload)}]}

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
            return {"entries": box.list_items(folder_id), "total_count":
                    len(box.list_items(folder_id))}
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
            except _MockConflict:
                # Build a synthetic Response with status_code=409 so
                # `_api_ensure_folder`'s `except requests.HTTPError`
                # branch sees what it expects.
                resp = requests.Response()
                resp.status_code = 409
                err = requests.HTTPError("409 conflict", response=resp)
                raise err
        raise NotImplementedError(f"mock _api_post doesn't handle {path}")
    return _post


def _mock_upload_file(box: _MockBox):
    """Mirror box_fetch._api_upload_file's signature.

    Honours the size-matched-skip contract: if `existing[name]` has the
    same size as the local file, return ``{"skipped": True}`` without
    re-uploading bytes.
    """
    def _upload(local_path: Path, parent_folder_id: str,
                name=None, existing=None) -> dict:
        name = name or local_path.name
        size = local_path.stat().st_size
        if existing and name in existing and int(existing[name].get("size") or -1) == size:
            return {"id": existing[name]["id"], "name": name, "skipped": True}
        result = box.upload_file(
            name=name, parent_id=parent_folder_id,
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
                p.write_bytes(b"\x00" * 1024)               # 1 KB per "depth"
    return out


@pytest.fixture
def mock_box(monkeypatch):
    box = _MockBox()
    import box_fetch  # noqa: PLC0415
    # Pretend a token is present so the auth path runs.
    monkeypatch.setattr(box_fetch, "DEV_TOKEN", "mock-token-XXXX", raising=False)
    monkeypatch.setattr(box_fetch, "_api_get",  _mock_api_get(box))
    monkeypatch.setattr(box_fetch, "_api_post", _mock_api_post(box))
    monkeypatch.setattr(box_fetch, "_api_upload_file", _mock_upload_file(box))
    return box


# ────────────────────────────  Contract tests  ───────────────────────────


def test_upload_tree_mirrors_scene_phase_layout(fake_run_dir, mock_box):
    """The exact tree on Box must match the user-asked
    ``monocular_depth/<model>/<split>/<scene>/<phase>/<frame>.npz`` shape."""
    from box_fetch import upload_tree

    summary = upload_tree(
        fake_run_dir,
        remote_path="monocular_depth/ZoeDepth_NK/easy",
        root_folder_id="0",
        verbose=False,
    )

    # 1. Total upload count = 11 (3 metadata + 8 predictions).
    assert summary["uploaded"] == 11
    assert summary["bytes_uploaded"] >= 8 * 1024     # at least the prediction bytes

    # 2. Top-level metadata at <root>/monocular_depth/ZoeDepth_NK/easy/.
    for name in ("result.json", "summary.md", "comprehensive_metrics.json"):
        fid = mock_box.resolve_path("monocular_depth", "ZoeDepth_NK", "easy", name)
        assert fid is not None, f"missing {name} on Box"

    # 3. Predictions preserve scene/phase/frame layout.
    for scene in ("scene_alpha.foo.bar", "scene_beta.baz.qux"):
        for phase in ("0", "1"):
            for frame in ("00000.npz", "00001.npz"):
                fid = mock_box.resolve_path(
                    "monocular_depth", "ZoeDepth_NK", "easy",
                    "predictions", scene, phase, frame,
                )
                assert fid is not None, f"missing {scene}/{phase}/{frame}"


def test_upload_tree_idempotent_on_rerun(fake_run_dir, mock_box):
    """Re-running upload_tree against the same local dir uploads zero
    new files (every file's Box copy already has the matching size)."""
    from box_fetch import upload_tree

    s1 = upload_tree(fake_run_dir, "monocular_depth/M/easy",
                     root_folder_id="0", verbose=False)
    n_uploads_first = len(mock_box.uploads)
    s2 = upload_tree(fake_run_dir, "monocular_depth/M/easy",
                     root_folder_id="0", verbose=False)
    n_uploads_second = len(mock_box.uploads)

    assert s1["uploaded"] == 11
    assert s2["uploaded"] == 0, f"second upload re-uploaded {s2['uploaded']} files"
    assert n_uploads_second == n_uploads_first, \
        f"second run hit upload endpoint {n_uploads_second - n_uploads_first} extra times"
    assert s2["skipped"] == 11
