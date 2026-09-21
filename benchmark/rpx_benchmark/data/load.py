"""One-line ``datasets.load_dataset`` entry point for RPX.

Example
-------

::

    from rpx_benchmark.data import load_hf

    ds = load_hf("monocular_depth", split="hard")
    for batch in ds:
        # each sample is a full rpx_benchmark.Sample
        ...
"""

from __future__ import annotations

from typing import Any

from ..api import Difficulty, TaskType
from ..exceptions import DownloadError
from ..hub import DEFAULT_REPO_ID, DEFAULT_REVISION
from ..logging_utils import get_logger
from .hf_bridge import RPXHFBridge

log = get_logger(__name__)

def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Lazy import of :func:`datasets.load_dataset`.

    Kept lazy so ``import rpx_benchmark.data.load`` does not force a
    pyarrow / datasets import at package load time for users who only
    wanted the convenience re-export surface.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rpx_benchmark.data.load_hf requires the `datasets` library. "
            "Install with: pip install 'rpx-benchmark[hf-datasets]'"
        ) from e
    return load_dataset(*args, **kwargs)


def load_hf(
    task: TaskType | str,
    split: Difficulty | str = Difficulty.HARD,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = DEFAULT_REVISION,
    cache_dir: str | None = None,
    streaming: bool = False,
    batch_size: int = 1,
    **load_dataset_kwargs: Any,
) -> RPXHFBridge:
    """Load an RPX task slice as an :class:`RPXHFBridge`.

    Parameters
    ----------
    task : TaskType or str
        Task config name on the HF repo. Uses the enum ``.value`` if a
        :class:`TaskType` is passed.
    split : Difficulty or str, default ``"hard"``
        ESD split to fetch. Maps to ``datasets.load_dataset``'s
        ``split`` kwarg.
    repo_id : str
        Hugging Face dataset repo id. Defaults to
        ``anonymous/RPX`` (overridable via ``RPX_HF_REPO``).
    revision : str, optional
        Git revision (tag, branch, or commit) to pin. Defaults to the
        consolidated audited release; override it with ``RPX_HF_REVISION``.
    cache_dir : str, optional
        Hugging Face cache directory. When None the library default is
        used (``~/.cache/huggingface``).
    streaming : bool, default False
        Pass through to ``datasets.load_dataset``. Enables row-level
        streaming from the Hub without materialising the full Arrow
        tables locally.
    batch_size : int, default 1
        Forwarded into the returned :class:`RPXHFBridge`.
    **load_dataset_kwargs
        Any extra kwargs are forwarded to ``datasets.load_dataset``.

    Returns
    -------
    RPXHFBridge
        An iterable wrapper that yields ``list[Sample]`` batches.

    Raises
    ------
    DownloadError
        If the underlying ``datasets.load_dataset`` call fails.
    """
    task_enum = TaskType(task) if isinstance(task, str) else task
    split_name = split.value if isinstance(split, Difficulty) else str(split)

    log.info(
        "loading HF dataset repo=%s config=%s split=%s streaming=%s",
        repo_id,
        task_enum.value,
        split_name,
        streaming,
    )
    try:
        hf_ds = _load_dataset(
            repo_id,
            name=task_enum.value,
            split=split_name,
            revision=revision,
            cache_dir=cache_dir,
            streaming=streaming,
            **load_dataset_kwargs,
        )
    except Exception as e:
        raise DownloadError(
            f"datasets.load_dataset failed for {repo_id}/{task_enum.value} "
            f"(split={split_name}): {e}",
            hint=(
                "Check the repo id and your HF_TOKEN for private datasets. "
                "For offline reuse set HF_DATASETS_OFFLINE=1 after a first "
                "online load populates the cache."
            ),
        ) from e

    return RPXHFBridge(hf_dataset=hf_ds, task=task_enum, batch_size=batch_size)
