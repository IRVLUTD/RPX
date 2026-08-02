"""FE2E — paper roster entry; UNVERIFIED upstream weights.

The paper roster lists FE2E (relative-depth model, paper_ref="fe2e").
As of 2026-06-25 there is no verified-official HF or GitHub release:
the candidate found on HF is ``exander/FE2E`` (community handle,
empty README, no documented lineage to the FE2E paper authors).

This adapter ships with the same safety-rail pattern as the Video
Depth side's D4RT wrapper: building succeeds, but
``__init__`` raises :class:`UnverifiedAdapterError` unless the caller
passes ``acknowledge_unverified=True``.

Once the team confirms the candidate weights are the paper's actual
model (or replaces them with the verified-official release), the
safety rail can be removed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from rpx_benchmark.exceptions import AdapterError, UnverifiedAdapterError

_CANDIDATE_HF = "exander/FE2E"


class FE2EAdapter:
    """Callable adapter for FE2E (relative depth).

    Behind the unverified-weights safety rail — see module docstring.
    Once instantiated with ``acknowledge_unverified=True``, the
    constructor raises ``NotImplementedError`` because no upstream
    model class is wired yet (the candidate HF repo has no code).
    """

    #: FE2E is affine-invariant per the paper headline; runner applies
    #: per-scene-phase scale-and-shift alignment.
    native_alignment: str = "ls_affine"
    native_precision: str = "fp16"

    UNVERIFIED = True
    UNVERIFIED_NOTE = (
        "Candidate weights at huggingface.co/exander/FE2E are a community "
        "upload with an empty README and no documented lineage to the "
        "FE2E paper authors. Lineage must be verified before publishing "
        "benchmark numbers under this model name."
    )
    UNVERIFIED_CANDIDATE_HF = _CANDIDATE_HF

    def __init__(
        self,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
        *,
        acknowledge_unverified: bool = False,
    ) -> None:
        if not acknowledge_unverified:
            raise UnverifiedAdapterError(
                f"{type(self).__name__} is marked UNVERIFIED. "
                f"{self.UNVERIFIED_NOTE}\n"
                f"Candidate HF path: huggingface.co/{self.UNVERIFIED_CANDIDATE_HF}\n"
                f"\n"
                f"To run anyway:\n"
                f"  * CLI:   pass --acknowledge-unverified\n"
                f"  * Code:  pass acknowledge_unverified=True to the constructor\n"
                f"\n"
                f"BEFORE PUBLISHING NUMBERS FROM THIS ADAPTER: confirm "
                f"with the paper authors that the candidate weights match "
                f"their published model, then update the adapter to drop "
                f"this safety rail and point at the verified-official "
                f"release path."
            )
        self.device = device
        self.batch_size = int(batch_size)
        # Once acknowledged, attempting to actually load fails with a
        # clear "wire the upstream code" instruction.
        raise NotImplementedError(
            "FE2EAdapter: safety rail cleared via acknowledge_unverified, "
            "but no upstream model class is wired yet. Locate the FE2E "
            "paper's official model code, import it here, download "
            f"weights via huggingface_hub from '{_CANDIDATE_HF}', load "
            "the state_dict, and replace this raise with the model "
            "instantiation. Then drop the UnverifiedAdapter safety rail "
            "once the paper-authors lineage is confirmed."
        )

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        raise AdapterError(
            "FE2EAdapter.__call__ unreachable — constructor did not "
            "complete (no upstream model class wired). See FE2EAdapter "
            "module docstring for the team task."
        )
