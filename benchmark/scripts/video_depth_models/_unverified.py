"""Safety-rail mixin for adapters whose upstream weights aren't verified-official.

Some canonical-roster models (D4RT on the Video Depth side;
FE2E on Image Depth) ship only as candidate community HF uploads with
empty READMEs and no clear lineage to the paper authors. We want the
runner to:

1. **Accept the canonical CLI name** so all 20 paper-roster slots
   resolve via ``--model <name>`` without a confusing ImportError.
2. **Refuse to actually run** the unverified adapter unless the team
   has explicitly acknowledged the risk — protects against
   accidentally publishing benchmark numbers under a model name whose
   weights might not match the paper's actual model.

The contract: a subclass sets ``UNVERIFIED = True`` and
``UNVERIFIED_NOTE = "<one-sentence reason>"`` on the class, and
inherits :meth:`_assert_acknowledged` which is called in ``setup``.
If the constructor was not passed ``acknowledge_unverified=True``,
:meth:`_assert_acknowledged` raises :class:`UnverifiedAdapterError`
with the note + full instructions for clearing it.
"""

from __future__ import annotations

from rpx_benchmark.exceptions import UnverifiedAdapterError


class UnverifiedAdapterMixin:
    """Mixin that gates ``setup()`` behind an explicit acknowledgement.

    Subclasses must set:

    * ``UNVERIFIED = True``
    * ``UNVERIFIED_NOTE = "<one-sentence note about the candidate HF
      repo and why it's unverified>"``
    * ``UNVERIFIED_CANDIDATE_HF = "<owner>/<model>"``

    And accept ``acknowledge_unverified: bool = False`` in
    ``__init__`` (stored as ``self._acknowledged``).
    """

    UNVERIFIED: bool = True
    UNVERIFIED_NOTE: str = ""
    UNVERIFIED_CANDIDATE_HF: str = ""

    def _assert_acknowledged(self) -> None:
        """Raise unless the caller acknowledged the unverified status."""
        if not getattr(self, "_acknowledged", False):
            raise UnverifiedAdapterError(
                f"{type(self).__name__} is marked UNVERIFIED. "
                f"{self.UNVERIFIED_NOTE}\n"
                f"Candidate HF path: huggingface.co/{self.UNVERIFIED_CANDIDATE_HF}\n"
                f"\n"
                f"To run anyway:\n"
                f"  * CLI:   pass --acknowledge-unverified\n"
                f"  * Code:  pass acknowledge_unverified=True to build()\n"
                f"\n"
                f"BEFORE PUBLISHING NUMBERS FROM THIS ADAPTER: confirm "
                f"with the paper authors that the candidate weights match "
                f"their published model, then update the adapter to drop "
                f"this safety rail and point at the verified-official "
                f"release path."
            )


__all__ = ["UnverifiedAdapterMixin"]
