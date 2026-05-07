"""Upload a packed staging directory to a HuggingFace dataset repo.

Wraps :func:`huggingface_hub.upload_large_folder` (the workhorse for
multi-hundred-GB pushes — chunked, multi-commit, resumable). For small
mock uploads / dry runs we fall back to :func:`upload_folder`.

The uploader does *no* packing. Run ``pack_capture_tree`` first, then
``build_frame_manifest``, then point this at the resulting staging dir.

Resumability: ``upload_large_folder`` writes its checkpoint state in
``staging_dir/.huggingface/`` so a re-invocation after a crash picks up
where it left off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from ..exceptions import ConfigError, DownloadError
from ..logging_utils import get_logger
from .recipes import DEFAULT_REPO_ID

log = get_logger(__name__)


# Patterns we always exclude from upload, regardless of caller settings.
# Build artefacts and editor leftovers should never end up in the repo.
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".DS_Store",
    "__pycache__",
    "*.pyc",
    "*.tmp",
    "*.part",
    ".huggingface/*",  # HF's own checkpoint state
)


@dataclass(frozen=True)
class UploadPlan:
    """Inputs needed to push a staging directory to a HF dataset repo."""

    staging_root: Path
    repo_id: str = DEFAULT_REPO_ID
    revision: str = "main"
    private: bool = True
    commit_message: str = "RPX dataset hub upload"
    create_if_missing: bool = True
    use_large_folder: bool = True
    allow_patterns: Optional[Sequence[str]] = None
    ignore_patterns: Sequence[str] = DEFAULT_IGNORE_PATTERNS
    dry_run: bool = False


@dataclass(frozen=True)
class UploadResult:
    """Reported back so callers can log / verify."""

    repo_id: str
    revision: str
    files_planned: int
    bytes_planned: int
    commit_url: Optional[str] = None
    skipped: List[Path] = field(default_factory=list)


def _hub():
    try:
        import huggingface_hub as hf  # noqa: WPS433
    except ImportError as e:
        raise DownloadError(
            "huggingface_hub is required for RPX uploads.",
            hint="Install with: pip install 'rpx-benchmark[hub]'",
        ) from e
    return hf


def _enumerate_uploadable(
    root: Path,
    ignore: Sequence[str],
    allow: Optional[Sequence[str]] = None,
) -> tuple[List[Path], int]:
    """Walk ``root``, return (files, total_bytes), respecting ignore + allow.

    ``allow`` mirrors ``huggingface_hub`` semantics: when set, *only*
    paths matching at least one allow glob are considered. ``ignore``
    is always applied. Both lists match against the POSIX path relative
    to ``root`` *and* against the bare file name.
    """
    import fnmatch

    keep: List[Path] = []
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in ignore):
            continue
        if allow is not None and not any(
            fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in allow
        ):
            continue
        keep.append(p)
        total += p.stat().st_size
    return keep, total


def upload_staging(plan: UploadPlan) -> UploadResult:
    """Push everything under ``plan.staging_root`` to the HF dataset repo.

    Behaviour:

    * If the repo doesn't exist and ``create_if_missing`` is True, it is
      created (private by default, dataset type).
    * If ``dry_run`` is True, no network call is made — we walk the tree
      and report what *would* be uploaded.
    * Otherwise, ``upload_large_folder`` is preferred for true resumable
      multi-commit pushes; ``upload_folder`` is the fallback for tiny
      mock uploads where the multi-commit overhead isn't worth it.
    """
    if not plan.staging_root.is_dir():
        raise ConfigError(
            f"staging dir does not exist: {plan.staging_root}",
            hint="Run pack_capture_tree() and build_frame_manifest() first.",
        )

    files, total = _enumerate_uploadable(
        plan.staging_root,
        plan.ignore_patterns,
        plan.allow_patterns,
    )
    log.info(
        "upload plan: %d files, %d bytes → %s (revision=%s, dry_run=%s)",
        len(files),
        total,
        plan.repo_id,
        plan.revision,
        plan.dry_run,
    )

    if plan.dry_run:
        return UploadResult(
            repo_id=plan.repo_id,
            revision=plan.revision,
            files_planned=len(files),
            bytes_planned=total,
        )

    hf = _hub()
    api = hf.HfApi()

    if plan.create_if_missing:
        try:
            api.create_repo(
                repo_id=plan.repo_id,
                repo_type="dataset",
                private=plan.private,
                exist_ok=True,
            )
        except Exception as e:
            raise DownloadError(
                f"failed to create or verify dataset repo {plan.repo_id}: {e}",
                hint="Run `hf auth login` and confirm you can push to the org.",
            ) from e

    common_kwargs = dict(
        repo_id=plan.repo_id,
        repo_type="dataset",
        folder_path=str(plan.staging_root),
        revision=plan.revision,
        ignore_patterns=list(plan.ignore_patterns),
    )
    if plan.allow_patterns:
        common_kwargs["allow_patterns"] = list(plan.allow_patterns)

    try:
        if plan.use_large_folder and hasattr(api, "upload_large_folder"):
            # `upload_large_folder` does its own multi-commit chunking
            # and writes checkpoint state under .huggingface/ so re-runs
            # resume cleanly.
            api.upload_large_folder(**common_kwargs)
            commit_url = None
        else:
            commit = api.upload_folder(
                commit_message=plan.commit_message,
                **common_kwargs,
            )
            commit_url = getattr(commit, "commit_url", None)
    except Exception as e:
        raise DownloadError(
            f"upload to {plan.repo_id} failed: {e}",
            hint=(
                "Re-run to resume; HF stores checkpoint state under "
                "<staging>/.huggingface/. If the failure is auth, run "
                "`hf auth login` and confirm push access to the org."
            ),
        ) from e

    return UploadResult(
        repo_id=plan.repo_id,
        revision=plan.revision,
        files_planned=len(files),
        bytes_planned=total,
        commit_url=commit_url,
    )


# Allow uploading just one or a few patterns (e.g. only the manifest, or
# only one modality) without materialising a separate staging dir.
def upload_paths(
    paths: Sequence[Path],
    staging_root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    private: bool = True,
    commit_message: str = "RPX dataset hub: partial upload",
    dry_run: bool = False,
) -> UploadResult:
    """Upload a specific list of files (relative to ``staging_root``).

    Useful for shipping a single modality or just the manifest after a
    partial re-pack. Internally builds an ``allow_patterns`` list and
    delegates to :func:`upload_staging` (which still calls
    ``upload_large_folder``/``upload_folder`` under the hood).
    """
    rels = []
    total = 0
    for p in paths:
        rel = p.relative_to(staging_root).as_posix()
        rels.append(rel)
        total += p.stat().st_size

    plan = UploadPlan(
        staging_root=staging_root,
        repo_id=repo_id,
        revision=revision,
        private=private,
        commit_message=commit_message,
        allow_patterns=tuple(rels),
        use_large_folder=False,
        dry_run=dry_run,
    )
    return upload_staging(plan)
