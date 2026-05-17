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

Auth has two modes, in priority order:

1. **OAuth 2.0 with refresh-token rotation** (preferred for long runs).
   Set ``BOX_CLIENT_ID`` + ``BOX_CLIENT_SECRET`` (+ ``BOX_REDIRECT_URI``,
   default ``http://localhost:8765/callback``) and run::

       python -m rpx_benchmark.box_upload login

   once to capture the refresh token. Tokens are stored at
   ``~/.config/rpx_benchmark/box_tokens.json`` (mode 600) and the
   access token is auto-renewed via the refresh token on every call,
   so multi-hour benchmark sweeps don't expire mid-run.
2. **Developer token** (legacy / quick testing). If no OAuth tokens
   exist on disk we fall back to ``BOX_DEVELOPER_TOKEN`` from the
   environment — but those are capped at ~60 minutes by Box.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests

log = logging.getLogger(__name__)

USER_AGENT = "rpx-benchmark/box-upload"
BOX_API = "https://api.box.com/2.0"
BOX_AUTHORIZE_URL = "https://account.box.com/api/oauth2/authorize"
BOX_TOKEN_URL = "https://api.box.com/oauth2/token"

DEFAULT_BOX_FOLDER_ID = "380510613151"  # team's RPX-Outputs folder
DEFAULT_REDIRECT_URI = "http://localhost:8765/callback"
TOKENS_PATH = Path("~/.config/rpx_benchmark/box_tokens.json").expanduser()
# Refresh the access token if it's within this many seconds of expiry,
# so long-running uploads don't see a 401 mid-call.
EXPIRY_SKEW_SEC = 60

__all__ = [
    "DEFAULT_BOX_FOLDER_ID",
    "upload_tree",
    "upload_run_dir",
    "oauth_login",
]


def _human_bytes(n: int | float) -> str:
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


# ------------------------------------------------------------------ #
# OAuth 2.0 with refresh-token rotation
# ------------------------------------------------------------------ #
def _load_tokens() -> Optional[dict[str, Any]]:
    if not TOKENS_PATH.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(TOKENS_PATH.read_text())
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _save_tokens(tokens: dict[str, Any]) -> None:
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_PATH.write_text(json.dumps(tokens, indent=2))
    try:
        TOKENS_PATH.chmod(0o600)
    except OSError:
        pass


def _refresh_access_token(refresh_token: str, client_id: str, client_secret: str) -> dict[str, Any]:
    """Exchange a refresh token for a new (access_token, refresh_token) pair.

    Box rotates refresh tokens on use: the response contains a NEW refresh
    token that supersedes the old one. The old token is then invalid.
    """
    r = requests.post(
        BOX_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if not r.ok:
        from .exceptions import ConfigError

        raise ConfigError(
            f"Box OAuth refresh failed ({r.status_code}): {r.text[:200]}",
            hint="The refresh token may have expired (60-day max idle). "
            "Re-run `python -m rpx_benchmark.box_upload login` to re-authorize.",
        )
    body: dict[str, Any] = r.json()
    return body


def _stash_tokens_from_response(resp: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Box token response into our on-disk format + persist.

    We deliberately do NOT persist client_id / client_secret here — they
    live in env vars (typically exported in the user's shell rc) so the
    on-disk file only carries the rotating tokens.
    """
    tokens = {
        "access_token": resp["access_token"],
        "refresh_token": resp["refresh_token"],
        "expires_at": time.time() + int(resp.get("expires_in", 3600)),
    }
    _save_tokens(tokens)
    return tokens


def _read_oauth_creds_from_env() -> tuple[Optional[str], Optional[str]]:
    """Return (client_id, client_secret) from BOX_CLIENT_ID / BOX_CLIENT_SECRET."""
    return os.environ.get("BOX_CLIENT_ID"), os.environ.get("BOX_CLIENT_SECRET")


def _token() -> str:
    """Return a valid Box access token, refreshing via OAuth if available.

    Order:
      1. OAuth tokens on disk → refresh if near expiry, return access token.
         Uses ``BOX_CLIENT_ID`` / ``BOX_CLIENT_SECRET`` from the environment
         to authenticate the refresh call.
      2. ``BOX_DEVELOPER_TOKEN`` env var (legacy 60-min token).
    """
    tokens = _load_tokens()
    if tokens:
        if time.time() + EXPIRY_SKEW_SEC >= float(tokens.get("expires_at", 0)):
            client_id, client_secret = _read_oauth_creds_from_env()
            if not (client_id and client_secret):
                from .exceptions import ConfigError

                raise ConfigError(
                    "OAuth access token expired and BOX_CLIENT_ID / BOX_CLIENT_SECRET not in env.",
                    hint="Export both (typically from your shell rc) or re-run "
                    "`python -m rpx_benchmark.box_upload login`.",
                )
            log.info("[box] access token expired or near expiry — refreshing")
            resp = _refresh_access_token(tokens["refresh_token"], client_id, client_secret)
            tokens = _stash_tokens_from_response(resp)
        return str(tokens["access_token"])

    tok = os.environ.get("BOX_DEVELOPER_TOKEN")
    if not tok:
        from .exceptions import ConfigError

        raise ConfigError(
            "No Box credentials available.",
            hint="Either run `python -m rpx_benchmark.box_upload login` (OAuth, "
            "auto-renews — recommended for long runs) or set "
            "BOX_DEVELOPER_TOKEN to a 60-min developer token.",
        )
    return tok


class _CodeCatcher(BaseHTTPRequestHandler):  # pragma: no cover
    """Single-shot HTTP handler that captures the OAuth ?code= callback."""

    code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        q = parse_qs(urlparse(self.path).query)
        _CodeCatcher.code = q.get("code", [None])[0]
        _CodeCatcher.error = q.get("error_description", q.get("error", [None]))[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body = (
            "<h2>Box login complete.</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            if _CodeCatcher.code
            else f"<h2>Box login failed.</h2><pre>{_CodeCatcher.error}</pre>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        return  # silence stderr access log


def oauth_login(  # pragma: no cover — interactive: spawns a browser + local HTTPServer
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    redirect_uri: Optional[str] = None,
) -> dict[str, Any]:
    """Interactive one-time login. Opens a browser, captures the auth code on
    a local HTTP server, exchanges it for tokens, writes them to disk.

    Args default to ``BOX_CLIENT_ID`` / ``BOX_CLIENT_SECRET`` /
    ``BOX_REDIRECT_URI`` from the environment. ``BOX_REDIRECT_URI`` must
    match what's configured on the Box app; default is
    ``http://localhost:8765/callback``.
    """
    client_id = client_id or os.environ.get("BOX_CLIENT_ID")
    client_secret = client_secret or os.environ.get("BOX_CLIENT_SECRET")
    redirect_uri = redirect_uri or os.environ.get("BOX_REDIRECT_URI") or DEFAULT_REDIRECT_URI
    if not (client_id and client_secret):
        from .exceptions import ConfigError

        raise ConfigError(
            "BOX_CLIENT_ID and BOX_CLIENT_SECRET required for OAuth login.",
            hint="Set them from your Box app at https://app.box.com/developers/console "
            "and re-run `python -m rpx_benchmark.box_upload login`.",
        )

    parsed = urlparse(redirect_uri)
    if parsed.hostname not in ("localhost", "127.0.0.1") or not parsed.port:
        from .exceptions import ConfigError

        raise ConfigError(
            f"redirect_uri must be http://localhost:<port>/<path>; got {redirect_uri!r}",
            hint="Set BOX_REDIRECT_URI (and the same value in the Box app config) "
            "to e.g. http://localhost:8765/callback.",
        )

    authorize = f"{BOX_AUTHORIZE_URL}?" + urlencode(
        {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri}
    )
    server = HTTPServer((parsed.hostname, parsed.port), _CodeCatcher)
    sys.stderr.write(f"[box] opening browser to: {authorize}\n")
    sys.stderr.write(f"[box] waiting for redirect on {redirect_uri} ...\n")
    webbrowser.open(authorize)
    server.handle_request()  # blocks until one callback received
    server.server_close()
    if _CodeCatcher.error or not _CodeCatcher.code:
        from .exceptions import ConfigError

        raise ConfigError(
            f"OAuth login failed: {_CodeCatcher.error or 'no code returned'}",
            hint="Check the Box app's redirect URI matches BOX_REDIRECT_URI exactly.",
        )

    r = requests.post(
        BOX_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": _CodeCatcher.code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if not r.ok:
        from .exceptions import ConfigError

        raise ConfigError(
            f"Box token exchange failed ({r.status_code}): {r.text[:200]}",
            hint="Confirm client_id/client_secret are correct and OAuth 2.0 is enabled on the app.",
        )
    tokens = _stash_tokens_from_response(r.json())
    sys.stderr.write(
        f"[box] tokens saved to {TOKENS_PATH} (mode 600)\n"
        f"[box] access expires in {int(tokens['expires_at'] - time.time())}s; "
        "refresh token auto-renews on use\n"
    )
    return tokens


def _api_get(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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
    return r.json()  # type: ignore[no-any-return]


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
    return r.json()  # type: ignore[no-any-return]


def _api_list_folder(folder_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
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
                return str(it["id"])
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


def _api_existing_files(folder_id: str) -> dict[str, dict[str, Any]]:
    return {it["name"]: it for it in _api_list_folder(folder_id) if it["type"] == "file"}


def _api_upload_file(
    local: Path,
    parent_folder_id: str,
    name: Optional[str] = None,
    existing: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Upload one file. If ``existing[name]`` matches local size, skip. If exists
    with different size, upload as a new version on the same file id."""
    name = name or local.name
    size = local.stat().st_size
    existing = existing if existing is not None else _api_existing_files(parent_folder_id)
    if name in existing and int(existing[name].get("size") or -1) == size:
        return {"id": existing[name]["id"], "name": name, "skipped": True}

    headers = {"Authorization": f"Bearer {_token()}", "User-Agent": USER_AGENT}
    attrs: dict[str, Any]
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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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


def _cli() -> int:  # pragma: no cover — CLI dispatcher; logic lives in the helpers above
    import argparse

    ap = argparse.ArgumentParser(prog="python -m rpx_benchmark.box_upload")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_login = sub.add_parser(
        "login",
        help="One-time OAuth login. Captures a refresh token for unattended uploads.",
    )
    p_login.add_argument("--client-id", default=None, help="default: $BOX_CLIENT_ID")
    p_login.add_argument("--client-secret", default=None, help="default: $BOX_CLIENT_SECRET")
    p_login.add_argument(
        "--redirect-uri",
        default=None,
        help=f"default: $BOX_REDIRECT_URI or {DEFAULT_REDIRECT_URI}",
    )

    sub.add_parser("whoami", help="Print the Box user the current token authenticates as.")
    sub.add_parser("logout", help="Delete cached OAuth tokens from disk.")

    args = ap.parse_args()
    if args.cmd == "login":
        oauth_login(args.client_id, args.client_secret, args.redirect_uri)
        return 0
    if args.cmd == "whoami":
        r = requests.get(
            f"{BOX_API}/users/me",
            headers={"Authorization": f"Bearer {_token()}", "User-Agent": USER_AGENT},
            timeout=30,
        )
        if not r.ok:
            sys.stderr.write(f"[box] {r.status_code}: {r.text[:200]}\n")
            return 1
        d = r.json()
        print(f"{d.get('name')} <{d.get('login')}>  id={d.get('id')}")
        return 0
    if args.cmd == "logout":
        if TOKENS_PATH.exists():
            TOKENS_PATH.unlink()
            print(f"removed {TOKENS_PATH}")
        else:
            print("no tokens on disk")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
