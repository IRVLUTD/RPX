"""PyTorch ``DataLoader`` adapter for RPX datasets.

Wraps either :class:`rpx_benchmark.loader.RPXDataset` or
:class:`rpx_benchmark.data.hf_bridge.RPXHFBridge` so users can plug
into the standard PyTorch training loop while still receiving
``list[Sample]`` batches (which adapters and the runner expect).

The collate function is deliberately trivial — it returns the list of
samples unchanged. Task-specific tensor stacking belongs in the
:class:`InputAdapter`, not here, so the dataloader stays model-agnostic.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from ..api import Sample


def rpx_collate(batch: Sequence[Any]) -> List[Sample]:
    """Collate a batch of either ``Sample`` objects or already-batched lists.

    ``RPXDataset`` / ``RPXHFBridge`` produce ``list[Sample]`` per
    iteration, whereas PyTorch's ``DataLoader`` expects a dataset whose
    ``__getitem__`` returns one item. We therefore flatten in case the
    inner dataset is already batched.
    """
    flat: List[Sample] = []
    for item in batch:
        if isinstance(item, Sample):
            flat.append(item)
        elif isinstance(item, Sequence):
            flat.extend(s for s in item if isinstance(s, Sample))
    return flat


def to_torch_dataloader(
    rpx_iterable: Iterable[Any],
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Any:
    """Wrap an RPX iterable in a ``torch.utils.data.DataLoader``.

    The returned DataLoader yields ``list[Sample]`` batches of the same
    size as the wrapped iterable's ``batch_size``.

    Parameters
    ----------
    rpx_iterable : Iterable
        Any object with ``__iter__`` yielding ``list[Sample]``.
    num_workers : int, default 0
        Passed to ``DataLoader``.
    pin_memory : bool, default False
        Passed to ``DataLoader``.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader, IterableDataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_torch_dataloader requires torch. "
            "Install with: pip install 'rpx-benchmark[depth-hf]' "
            "(or any extra that pulls torch)."
        ) from e

    class _RPXIterableDataset(IterableDataset):
        def __init__(self, source: Iterable[Any]) -> None:
            self._source = source

        def __iter__(self):  # type: ignore[override]
            for batch in self._source:
                yield batch

    return DataLoader(
        _RPXIterableDataset(rpx_iterable),
        batch_size=None,       # the source already yields batches
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=rpx_collate,
    )
