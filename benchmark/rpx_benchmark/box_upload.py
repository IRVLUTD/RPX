"""Box upload primitives — used by every task runner and the post-hoc
sync script.

Two public entry points:

* :func:`upload_tree(local_dir, remote_path, *, root_folder_id)` —
  recursively mirror a directory to Box. Size-matched idempotent; safe
  to re-run.
* :func:`upload_run_dir(out_dir, *, task, model_name, split, ...)` —
  the high-level wrapper that lays a per-run output directory down at
  the canonical ``<root_folder_id>/<task>/<model>/<split>/`` path the
  team agreed on.

Auth: reads ``BOX_DEVELOPER_TOKEN`` from the environment at call time
(tokens expire every 60 minutes, so we don't cache them at import).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

USER_AGENT = "rpx-benchmark/box-upload"
BOX_API = "https://api.box.com/2.0"

DEFAULT_BOX_FOLDER_ID = "380510613151"  # team's RPX-Outputs folder

__all__ = [
    "DEFAULT_BOX_FOLDER_ID",
    "upload_tree",
    "upload_run_dir",
]


def _human_bytes(n: int | float) -> str:
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _token() -> str:
    """Read BOX_DEVELOPER_TOKEN at call time (not import) so a refreshed
    token mid-process is picked up."""
    tok = os.environ.get("BOX_DEVELOPER_TOKEN")
    if not tok:
        from .exceptions import ConfigError

        raise ConfigError(
            "BOX_DEVELOPER_TOKEN required for Box upload.",
            hint="Generate a 60-min developer token at "
            "https://app.box.com/developers/console and `export BOX_DEVELOPER_TOKEN=...`.",
        )
    return tok


def _api_get(path: str, params: Optional[dict] = None) -> dict:
    r = requests.get(
        f"{BOX_API}{path}",
        headers={"Authorization": f"Bearer {_token()}", "User-Agent": USER_AGENT},
        params=params or {},
        timeout=60,
    )
    if r.status_code in (401, 403):
        from .exceptions import DatasetError

        raise DatasetError(
            f"Box API {r.status_code} — token expired or insufficient: {r.text[:200]}",
            hint="Token may be expired (60-min window) or lacks permissions for this folder.",
        )
    r.raise_for_status()
    return r.json()


def _api_post(path: str, payload: dict) -> dict:
    r = requests.post(
        f"{BOX_API}{path}",
        headers={
            "Authorization": f"Bearer {_token()}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if r.status_code in (401, 403):
        from .exceptions import DatasetError

        raise DatasetError(
            f"Box API {r.status_code} — token expired or insufficient: {r.text[:200]}",
            hint="Token may be expired (60-min window) or lacks permissions for this folder.",
        )
    r.raise_for_status()
    return r.json()


def _api_list_folder(folder_id: str) -> list[dict]:
    items: list[dict] = []
    offset = 0
    while True:
        data = _api_get(
            f"/folders/{folder_id}/items",
            params={"fields": "id,type,name,size", "limit": 1000, "offset": offset},
        )
        entries = data.get("entries", []) or []
        for it in entries:
            if it.get("type") not in ("file", "folder"):
                continue
            items.append(
                {
                    "id": str(it["id"]),
                    "type": it["type"],
                    "name": it.get("name") or str(it["id"]),
                    "size": int(it.get("size") or 0),
                }
            )
        if len(entries) < 1000:
            break
        offset += 1000
    return items


def _api_ensure_folder(name: str, parent_folder_id: str) -> str:
    """Return folder id for ``name`` under ``parent_folder_id``; create on miss."""
    try:
        data = _api_post("/folders", {"name": name, "parent": {"id": str(parent_folder_id)}})
        return str(data["id"])
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 409:
            raise
        for it in _api_list_folder(parent_folder_id):
            if it["type"] == "folder" and it["name"] == name:
                return it["id"]
        from .exceptions import DatasetError

        raise DatasetError(
            f"folder {name!r} reported conflict but not found under {parent_folder_id}",
            hint="Race condition or stale Box state. Re-run; the conflict resolver "
            "should pick up the existing folder on retry.",
        ) from e


def _api_resolve_path(parts: list[str], root_folder_id: str) -> str:
    cur = str(root_folder_id)
    for part in parts:
        if not part:
            continue
        cur = _api_ensure_folder(part, cur)
    return cur


def _api_existing_files(folder_id: str) -> dict[str, dict]:
    return {it["name"]: it for it in _api_list_folder(folder_id) if it["type"] == "file"}


def _api_upload_file(
    local: Path,
    parent_folder_id: str,
    name: Optional[str] = None,
    existing: Optional[dict[str, dict]] = None,
) -> dict:
    """Upload one file. If ``existing[name]`` matches local size, skip. If exists
    with different size, upload as a new version on the same file id."""
    name = name or local.name
    size = local.stat().st_size
    existing = existing if existing is not None else _api_existing_files(parent_folder_id)
    if name in existing and int(existing[name].get("size") or -1) == size:
        return {"id": existing[name]["id"], "name": name, "skipped": True}

    headers = {"Authorization": f"Bearer {_token()}", "User-Agent": USER_AGENT}
    if name in existing:
        url = f"https://upload.box.com/api/2.0/files/{existing[name]['id']}/content"
        attrs = {"name": name}
    else:
        url = "https://upload.box.com/api/2.0/files/content"
        attrs = {"name": name, "parent": {"id": str(parent_folder_id)}}
    with local.open("rb") as f:
        r = requests.post(
            url,
            headers=headers,
            timeout=600,
            data={"attributes": json.dumps(attrs)},
            files={"file": (name, f)},
        )
    if r.status_code in (401, 403):
        from .exceptions import DatasetError

        raise DatasetError(
            f"Box upload {r.status_code}: {r.text[:200]}",
            hint="Check the developer token is fresh and has write access to the target folder.",
        )
    r.raise_for_status()
    entry = r.json().get("entries", [{}])[0]
    return {"id": entry.get("id"), "name": name, "skipped": False}


def upload_tree(
    local_dir: Path | str,
    remote_path: str,
    *,
    root_folder_id: str = "0",
    verbose: bool = True,
) -> dict:
    """Upload every file under ``local_dir`` to ``<root>/<remote_path>``.

    Mirrors directory structure; size-matched files are skipped (idempotent).

    Parameters
    ----------
    local_dir
        Local directory whose contents should be mirrored.
    remote_path
        Slash-separated path *under* ``root_folder_id``. Intermediate
        folders are auto-created.
    root_folder_id
        Box folder id (not URL) to root the upload under. Default ``"0"`` =
        the auth token's "My Files".
    verbose
        If True, log one line per file to stderr.

    Returns
    -------
    dict
        ``{"uploaded": int, "skipped": int, "bytes_uploaded": int,
        "remote_folder_id": str}``.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)
    base_id = _api_resolve_path(remote_path.strip("/").split("/"), root_folder_id)
    n_up = n_skip = bytes_up = 0
    by_parent: dict[Path, list[Path]] = {}
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            by_parent.setdefault(p.parent, []).append(p)
    for parent_dir, files in by_parent.items():
        rel = parent_dir.relative_to(local_dir)
        parent_id = _api_resolve_path(str(rel).split("/") if str(rel) != "." else [], base_id)
        existing = _api_existing_files(parent_id)
        for f in files:
            if verbose:
                sys.stderr.write(f"[box] {f.relative_to(local_dir)} ... ")
                sys.stderr.flush()
            r = _api_upload_file(f, parent_id, existing=existing)
            if r["skipped"]:
                n_skip += 1
                if verbose:
                    sys.stderr.write("skip (already on Box)\n")
            else:
                n_up += 1
                bytes_up += f.stat().st_size
                if verbose:
                    sys.stderr.write(f"uploaded ({_human_bytes(f.stat().st_size)})\n")
    return {
        "uploaded": n_up,
        "skipped": n_skip,
        "bytes_uploaded": bytes_up,
        "remote_folder_id": base_id,
    }


def upload_run_dir(
    out_dir: Path | str,
    *,
    task: str,
    model_name: str,
    split: str,
    root_folder_id: str = DEFAULT_BOX_FOLDER_ID,
    verbose: bool = True,
) -> dict:
    """Mirror a single per-run output directory to Box.

    Lays everything under ``out_dir`` at the canonical path
    ``<root_folder_id>/<task>/<safe_model_name>/<split>/`` so the
    layout matches every other run (depth, pose, future tasks, post-hoc
    sync). ``safe_model_name`` swaps ``/`` for ``__`` so HF-style
    namespaces don't accidentally create nested Box folders.

    Returns the same dict as :func:`upload_tree`, plus the resolved
    ``remote_path`` for log lines.
    """
    safe_name = model_name.replace("/", "__")
    remote = f"{task}/{safe_name}/{split}"
    if verbose:
        sys.stderr.write(f"[box] mirror {Path(out_dir)} → box:{remote}\n")
    summary = upload_tree(
        out_dir, remote_path=remote, root_folder_id=root_folder_id, verbose=verbose
    )
    summary["remote_path"] = remote
    return summary
