"""Pull file(s) from UTD-Box shared links into a local cache.

Designed for the dev workflow where a teammate sends a Box URL and you want the
file(s) accessible to a pipeline without manually clicking Download. Three URL
shapes supported:

    1. Vanity short URL    https://<host>/s/<token>           (file or folder)
    2. Canonical file URL  https://<host>/file/<id>?s=<tok>
    3. Canonical folder    https://<host>/folder/<id>?s=<tok>

Anonymous-shared links work without Box auth (Box's `index.php?rm=box_download_shared_file`
endpoint). Auth-gated links require BOX_DEVELOPER_TOKEN (60-min token from the
Box developer console: https://app.box.com/developers/console).

Usage
-----
    # List a folder without downloading
    python scripts/box_fetch.py 'https://utdallas.box.com/s/<tok>' --list

    # Download a single file (cached on repeat)
    python scripts/box_fetch.py 'https://utdallas.box.com/s/<tok>'

    # Download specific items from a folder by name (regex)
    python scripts/box_fetch.py 'https://utdallas.box.com/s/<tok>' --include '\\.xlsx$'

    # Recurse into subfolders (default off — large folders can be huge)
    python scripts/box_fetch.py 'https://utdallas.box.com/s/<tok>' --recurse

    # Batch from a manifest (one URL per line, '#' for comments)
    python scripts/box_fetch.py --manifest urls.txt

    # Python API
    from box_fetch import fetch, list_folder
    items = list_folder("https://utdallas.box.com/s/<tok>")  # no download
    paths = fetch("https://utdallas.box.com/s/<tok>", include=r"\\.xlsx$")

Cache layout: $RPX_BOX_CACHE/<file_id>/<filename>, default $HOME/.cache/rpx-box.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Union

import requests

from urllib3.exceptions import InsecureRequestWarning  # noqa: F401  (some envs need silencing)

CACHE_ROOT = Path(os.environ.get("RPX_BOX_CACHE", Path.home() / ".cache" / "rpx-box"))
DEV_TOKEN = os.environ.get("BOX_DEVELOPER_TOKEN")  # optional 60-min token
USER_AGENT = "rpx-box-fetch/0.4"
BOX_API = "https://api.box.com/2.0"

LARGE_PULL_GB = 5  # warn if a recursive pull would exceed this


# ────────────────────────────  Box REST API (auth path)  ─────────────────────

def _api_get(path: str, params: Optional[dict] = None) -> dict:
    """GET https://api.box.com/2.0{path} with Bearer token. Raises if no token."""
    if not DEV_TOKEN:
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            "BOX_DEVELOPER_TOKEN not set; cannot use Box REST API.",
            hint="Generate a 60-min developer token at "
                 "https://app.box.com/developers/console and `export "
                 "BOX_DEVELOPER_TOKEN=...` before re-running.",
        )
    r = requests.get(
        f"{BOX_API}{path}",
        headers={"Authorization": f"Bearer {DEV_TOKEN}", "User-Agent": USER_AGENT},
        params=params or {}, timeout=30,
    )
    if r.status_code == 401:
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            "Box API returned 401 — token expired or invalid.",
            hint="Box developer tokens expire after 60 min. Regenerate at "
                 "https://app.box.com/developers/console and re-export.",
        )
    r.raise_for_status()
    return r.json()


def _api_list_folder(folder_id: str) -> list[dict]:
    """List one level of a Box folder via REST API."""
    items = []
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
            items.append({
                "id": str(it["id"]),
                "type": it["type"],
                "name": it.get("name") or str(it["id"]),
                "size": int(it.get("size") or 0),
            })
        if len(entries) < 1000:
            break
        offset += 1000
    return items


def _api_download_file(file_id: str, name: str, size: int, cache_dir: Path) -> Path:
    """Download via /files/{id}/content with Bearer token. Requires DEV_TOKEN."""
    out_dir = cache_dir / file_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    if out.exists() and (size == 0 or out.stat().st_size == size):
        return out
    with requests.get(
        f"{BOX_API}/files/{file_id}/content",
        stream=True, timeout=600,
        headers={"Authorization": f"Bearer {DEV_TOKEN}", "User-Agent": USER_AGENT},
    ) as r:
        if r.status_code == 401:
            from rpx_benchmark.exceptions import ConfigError
            raise ConfigError(
                "Box API returned 401 — token expired or invalid.",
                hint="Box developer tokens expire after 60 min. Regenerate and re-export.",
            )
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.rename(out)
    if size and out.stat().st_size != size:
        out.unlink(missing_ok=True)
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"size mismatch on {name}: got {out.stat().st_size}, expected {size}",
            hint="The download is corrupt — delete the cached file and re-run.",
        )
    return out


def _api_list_folder_recursive(folder_id: str, prefix: str, recurse: bool) -> list[dict]:
    items = _api_list_folder(folder_id)
    out = []
    for it in items:
        path = f"{prefix}/{it['name']}" if prefix else it["name"]
        out.append({**it, "path": path})
        if it["type"] == "folder" and recurse:
            try:
                out.extend(_api_list_folder_recursive(it["id"], path, recurse))
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"warning: cannot list {path}: {e}\n")
    return out


# ────────────────────────────  Box REST API (writer side)  ───────────────────

def _api_post(path: str, payload: dict, base: str = BOX_API) -> dict:
    if not DEV_TOKEN:
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            "BOX_DEVELOPER_TOKEN required for upload operations.",
            hint="Generate a 60-min developer token at "
                 "https://app.box.com/developers/console and `export BOX_DEVELOPER_TOKEN=...`.",
        )
    r = requests.post(
        f"{base}{path}",
        headers={"Authorization": f"Bearer {DEV_TOKEN}",
                 "User-Agent": USER_AGENT,
                 "Content-Type": "application/json"},
        json=payload, timeout=60,
    )
    if r.status_code in (401, 403):
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"Box API {r.status_code} — token expired or insufficient: {r.text[:200]}",
            hint="Token may be expired (60-min window) or lacks permissions for this folder.",
        )
    r.raise_for_status()
    return r.json()


def _api_ensure_folder(name: str, parent_folder_id: str) -> str:
    """Return folder id for `name` under `parent_folder_id`; create if absent."""
    # Try create first; on 409 conflict, look up existing.
    try:
        data = _api_post("/folders", {"name": name, "parent": {"id": str(parent_folder_id)}})
        return str(data["id"])
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 409:
            raise
        # Conflict: folder already exists. Look it up by name.
        for it in _api_list_folder(parent_folder_id):
            if it["type"] == "folder" and it["name"] == name:
                return it["id"]
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"folder {name!r} reported conflict but not found under {parent_folder_id}",
            hint="Race condition or stale Box state. Re-run; the conflict resolver should pick up "
                 "the existing folder on retry.",
        ) from e


def _api_resolve_path(parts: list[str], root_folder_id: str) -> str:
    """Walk + create-on-missing a /-separated subpath under root_folder_id."""
    cur = str(root_folder_id)
    for part in parts:
        if not part:
            continue
        cur = _api_ensure_folder(part, cur)
    return cur


def _api_existing_files(folder_id: str) -> dict[str, dict]:
    """Map filename → {id, size} for files directly in folder_id."""
    out = {}
    for it in _api_list_folder(folder_id):
        if it["type"] == "file":
            out[it["name"]] = it
    return out


def _api_upload_file(local: Path, parent_folder_id: str,
                      name: Optional[str] = None,
                      existing: Optional[dict[str, dict]] = None) -> dict:
    """Upload one file. If `existing[name]` matches local size, skip. If exists
    with different size, upload as a new version on the same file id."""
    if not DEV_TOKEN:
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            "BOX_DEVELOPER_TOKEN required for upload operations.",
            hint="Generate a 60-min developer token at "
                 "https://app.box.com/developers/console and `export BOX_DEVELOPER_TOKEN=...`.",
        )
    name = name or local.name
    size = local.stat().st_size
    existing = existing if existing is not None else _api_existing_files(parent_folder_id)
    if name in existing and int(existing[name].get("size") or -1) == size:
        return {"id": existing[name]["id"], "name": name, "skipped": True}

    headers = {"Authorization": f"Bearer {DEV_TOKEN}", "User-Agent": USER_AGENT}
    if name in existing:
        # New version of an existing file
        url = f"https://upload.box.com/api/2.0/files/{existing[name]['id']}/content"
        attrs = {"name": name}
    else:
        url = "https://upload.box.com/api/2.0/files/content"
        attrs = {"name": name, "parent": {"id": str(parent_folder_id)}}
    with local.open("rb") as f:
        r = requests.post(
            url, headers=headers, timeout=600,
            data={"attributes": json.dumps(attrs)},
            files={"file": (name, f)},
        )
    if r.status_code in (401, 403):
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"Box upload {r.status_code}: {r.text[:200]}",
            hint="Check the developer token is fresh and has write access to the target folder.",
        )
    r.raise_for_status()
    entry = r.json().get("entries", [{}])[0]
    return {"id": entry.get("id"), "name": name, "skipped": False}


def upload_tree(local_dir: Path, remote_path: str, *,
                root_folder_id: str = "0", verbose: bool = True) -> dict:
    """Upload every file under `local_dir` to Box at `<root>/<remote_path>`.

    Mirrors directory structure. Skips files whose Box copy has identical size.
    Uses BOX_DEVELOPER_TOKEN for auth (raises otherwise). `root_folder_id` is
    the id (not URL) of the Box folder to root the upload under; "0" = My Files.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)
    base_id = _api_resolve_path(remote_path.strip("/").split("/"), root_folder_id)
    n_up = n_skip = bytes_up = 0
    # Group files by their immediate parent so we list each subfolder once.
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
                if verbose: sys.stderr.write("skip (already on Box)\n")
            else:
                n_up += 1
                bytes_up += f.stat().st_size
                if verbose: sys.stderr.write(f"uploaded ({_human_bytes(f.stat().st_size)})\n")
    return {"uploaded": n_up, "skipped": n_skip, "bytes_uploaded": bytes_up,
            "remote_folder_id": base_id}


# ────────────────────────────  HTML scraping  ────────────────────────────────

_PREFETCH_RE = re.compile(r"Box\.prefetchedData\s*=\s*(\{.+?\});", re.DOTALL)
_POSTSTREAM_RE = re.compile(r"Box\.postStreamData\s*=\s*(\{.+?\});", re.DOTALL)


def _scrape(url: str) -> tuple[dict, dict]:
    """Return (prefetchedData, postStreamData) from a Box web page."""
    r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT},
                     allow_redirects=True)
    r.raise_for_status()
    pre_m = _PREFETCH_RE.search(r.text)
    post_m = _POSTSTREAM_RE.search(r.text)
    pre = json.loads(pre_m.group(1)) if pre_m else {}
    post = json.loads(post_m.group(1)) if post_m else {}
    if not pre and not post:
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"Box page exposed no metadata (URL: {url}). The link may be "
            "private/password-protected, or Box's page structure changed. "
            "Set BOX_DEVELOPER_TOKEN for the auth path, or open the URL in a "
            "browser to verify the link is valid."
        )
    return pre, post


# ────────────────────────────  URL parsing  ──────────────────────────────────

def _origin(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", url)
    return m.group(1) if m else "https://utdallas.app.box.com"


def _parse_url(url: str) -> tuple[str, str, str]:
    """Return (kind, id, token) where kind is 'file' or 'folder'.

    Handles vanity (/s/<token>) and canonical (/file or /folder) URL shapes.
    """
    m_kind = re.search(r"/(file|folder)/(\d+)", url)
    m_tok_query = re.search(r"[?&]s=([A-Za-z0-9_-]+)", url)
    m_tok_short = re.search(r"/s/([A-Za-z0-9_-]+)", url)
    token = (m_tok_query or m_tok_short).group(1) if (m_tok_query or m_tok_short) else None

    if m_kind and token:
        return m_kind.group(1), m_kind.group(2), token

    if m_tok_short and not m_kind:
        # Vanity URL: scrape the page to find what /s/<token> resolves to.
        _, post = _scrape(url)
        shared = post.get("/app-api/enduserapp/shared-item") or {}
        kind = shared.get("itemType")
        item_id = str(shared.get("itemID") or "")
        if kind in ("file", "folder") and item_id:
            return kind, item_id, token

    from rpx_benchmark.exceptions import ConfigError
    raise ConfigError(
        f"could not parse Box URL: {url}\n"
        "Expected one of:\n"
        "  https://<host>/s/<token>\n"
        "  https://<host>/file/<id>?s=<token>\n"
        "  https://<host>/folder/<id>?s=<token>"
    )


def _build_folder_url(origin: str, folder_id: str, token: str) -> str:
    return f"{origin}/folder/{folder_id}?s={token}"


# ────────────────────────────  File metadata  ────────────────────────────────

def _file_meta(pre: dict, post: dict, file_id: str) -> dict:
    """Find name/size/can_download for a file id across both prefetch and postStream."""
    # File-page shape: prefetchedData.preview_metadata
    pm = pre.get("preview_metadata") or {}
    if str(pm.get("id")) == file_id:
        return {
            "name": pm["name"],
            "size": int(pm.get("size") or 0),
            "downloadable": bool(pm.get("is_download_available", False)),
        }
    # Folder-page shape: postStreamData[".../shared-folder"].items[*]
    # File-from-folder-page also goes through .../item/f_<id>
    sources = list(post.values())
    for v in sources:
        if not isinstance(v, dict):
            continue
        for it in v.get("items") or []:
            if str(it.get("id")) == file_id and it.get("type") == "file":
                return {
                    "name": it.get("name") or file_id,
                    "size": int(it.get("itemSize") or 0),
                    "downloadable": bool(it.get("canDownload", True)),
                }
    from rpx_benchmark.exceptions import DatasetError
    raise DatasetError(
        f"no metadata for file {file_id} in Box page payload",
        hint="The shared link may be expired, password-protected, or Box's "
             "page structure changed. Try the BOX_DEVELOPER_TOKEN auth path.",
    )


# ────────────────────────────  Folder listing  ───────────────────────────────

def _folder_items(post: dict) -> list[dict]:
    """Extract item list from a folder page's postStreamData."""
    payload = (
        post.get("/app-api/enduserapp/shared-folder")
        or post.get("/app-api/enduserapp/folder")
        or {}
    )
    items = payload.get("items") or []
    if not items:
        # Fallback: scan all values for the first list of items
        for v in post.values():
            if isinstance(v, dict) and v.get("items"):
                items = v["items"]
                break
    out = []
    for it in items:
        kind = it.get("type")
        if kind not in ("file", "folder"):
            continue
        out.append({
            "id": str(it["id"]),
            "type": kind,
            "name": it.get("name") or str(it["id"]),
            "size": int(it.get("itemSize") or 0),
        })
    return out


def list_folder(url: str, recurse: bool = False) -> list[dict]:
    """Return the items under a Box folder URL (no downloads).

    With BOX_DEVELOPER_TOKEN: uses Box REST API → full subtree access.
    Without token: HTML-scrape path → top-level only (subfolder warnings).

    Each item is {id, type, name, size, path}.
    """
    kind, folder_id, token = _parse_url(url)
    if kind != "folder":
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            f"not a folder URL: {url}",
            hint="`list_folder` only accepts /folder/<id> or /s/<token-of-folder> URLs.",
        )
    if DEV_TOKEN:
        return _api_list_folder_recursive(folder_id, prefix="", recurse=recurse)
    origin = _origin(url)
    return _list_folder_recursive(url, token, origin, prefix="", recurse=recurse)


def _list_folder_recursive(url: str, token: str, origin: str,
                           prefix: str, recurse: bool) -> list[dict]:
    try:
        _, post = _scrape(url)
    except RuntimeError as e:
        sys.stderr.write(f"warning: cannot list {prefix or url}: {e}\n")
        return []
    items = _folder_items(post)
    out = []
    for it in items:
        path = f"{prefix}/{it['name']}" if prefix else it["name"]
        out.append({**it, "path": path})
        if it["type"] == "folder" and recurse:
            child_url = _build_folder_url(origin, it["id"], token)
            out.extend(_list_folder_recursive(child_url, token, origin, path, recurse))
    return out


# ────────────────────────────  Download  ─────────────────────────────────────

def _download_file(origin: str, file_id: str, token: str,
                   name: str, size: int, cache_dir: Path) -> Path:
    out_dir = cache_dir / file_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    if out.exists() and (size == 0 or out.stat().st_size == size):
        return out

    download_url = (
        f"{origin}/index.php"
        f"?rm=box_download_shared_file&shared_name={token}&file_id=f_{file_id}"
    )
    headers = {"User-Agent": USER_AGENT}
    if DEV_TOKEN:
        headers["Authorization"] = f"Bearer {DEV_TOKEN}"

    with requests.get(download_url, stream=True, timeout=300, headers=headers) as r:
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.rename(out)

    if size and out.stat().st_size != size:
        out.unlink(missing_ok=True)
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"size mismatch on {name}: got {out.stat().st_size}, expected {size}.",
            hint="Link may have expired or been revoked. Re-fetch the URL or "
                 "regenerate the share token.",
        )
    return out


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if isinstance(n, float) else f"{n} {unit}"
        n = n / 1024
    return f"{n:.1f} PB"


def fetch(
    url: str,
    cache_dir: Optional[Path] = None,
    *,
    recurse: bool = False,
    include: Optional[str] = None,
    confirm_large: bool = False,
) -> Union[Path, list[Path]]:
    """Download a file or every file under a folder URL (cached).

    Parameters
    ----------
    recurse       descend into subfolders (default True; pass False to grab
                  only the top level)
    include       regex; only files whose path matches are downloaded
    confirm_large bypass the >LARGE_PULL_GB safety prompt (caller asserts they
                  meant it; only relevant for folder URLs)
    """
    cache_dir = cache_dir or CACHE_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)
    kind, item_id, token = _parse_url(url)
    origin = _origin(url)

    if kind == "file":
        if DEV_TOKEN:
            info = _api_get(f"/files/{item_id}", params={"fields": "name,size"})
            return _api_download_file(item_id, info["name"], int(info.get("size") or 0), cache_dir)
        pre, post = _scrape(url)
        meta = _file_meta(pre, post, item_id)
        if not meta["downloadable"]:
            from rpx_benchmark.exceptions import DatasetError
            raise DatasetError(
                f"Box reports {meta['name']} is not downloadable.",
                hint="The shared link may have been revoked or the file is in a "
                     "permission-restricted state. Check the Box web UI.",
            )
        return _download_file(origin, item_id, token, meta["name"], meta["size"], cache_dir)

    # folder
    if DEV_TOKEN:
        items = _api_list_folder_recursive(item_id, prefix="", recurse=recurse)
    else:
        folder_url = _build_folder_url(origin, item_id, token)
        items = _list_folder_recursive(folder_url, token, origin, prefix="", recurse=recurse)
    pat = re.compile(include) if include else None
    files = [it for it in items if it["type"] == "file" and (not pat or pat.search(it["path"]))]
    total = sum(f["size"] for f in files)

    if not confirm_large and total > LARGE_PULL_GB * (1024 ** 3):
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            f"refusing to pull {_human_bytes(total)} ({len(files)} files) without explicit "
            f"confirmation.",
            hint="Re-run with confirm_large=True (or --yes from the CLI), or "
                 "narrow the scope with --include / --no-recurse / --list first."
        )

    paths = []
    for f in files:
        if DEV_TOKEN:
            paths.append(_api_download_file(f["id"], f["name"], f["size"], cache_dir))
        else:
            paths.append(_download_file(origin, f["id"], token, f["name"], f["size"], cache_dir))
    return paths


# ────────────────────────────  CLI  ──────────────────────────────────────────

def fetch_manifest(manifest_path: Path, cache_dir: Optional[Path] = None,
                   **kwargs) -> list[Path]:
    paths = []
    for raw in manifest_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        result = fetch(line, cache_dir=cache_dir, **kwargs)
        paths.extend(result if isinstance(result, list) else [result])
    return paths


def _print_listing(items: list[dict]) -> None:
    """Pretty-print a list_folder() result."""
    nfile = sum(1 for it in items if it["type"] == "file")
    nfold = sum(1 for it in items if it["type"] == "folder")
    total = sum(it["size"] for it in items if it["type"] == "file")
    for it in items:
        flag = "/" if it["type"] == "folder" else " "
        size = _human_bytes(it["size"]) if it["type"] == "file" else "      "
        print(f"  {flag}  {size:>10}  {it['path']}")
    print(f"\n  {nfold} folder(s), {nfile} file(s), {_human_bytes(total)} total")


def _cli():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("url", nargs="?", help="Box shared URL (file or folder)")
    ap.add_argument("--list", action="store_true",
                    help="list folder contents without downloading")
    ap.add_argument("--include", default=None,
                    help="regex; only files whose path matches are downloaded")
    ap.add_argument("--recurse", action="store_true",
                    help="descend into subfolders (default: top-level only; "
                         "subfolders inside an anonymous-shared parent often "
                         "require their own share links or BOX_DEVELOPER_TOKEN)")
    ap.add_argument("--yes", action="store_true",
                    help=f"confirm pulls larger than {LARGE_PULL_GB} GB")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="text file with one Box URL per line")
    ap.add_argument("--to", type=Path, default=None,
                    help="cache root override (default: $RPX_BOX_CACHE or ~/.cache/rpx-box)")
    args = ap.parse_args()

    if not args.url and not args.manifest:
        ap.error("provide a URL or --manifest")

    try:
        if args.list:
            if not args.url:
                ap.error("--list requires a URL")
            items = list_folder(args.url, recurse=args.recurse)
            _print_listing(items)
            return
        kw = dict(recurse=args.recurse, include=args.include, confirm_large=args.yes)
        paths = []
        if args.url:
            r = fetch(args.url, cache_dir=args.to, **kw)
            paths.extend(r if isinstance(r, list) else [r])
        if args.manifest:
            paths.extend(fetch_manifest(args.manifest, cache_dir=args.to, **kw))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"box_fetch error: {type(e).__name__}: {e}")
    for p in paths:
        print(p)


if __name__ == "__main__":
    _cli()
