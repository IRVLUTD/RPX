"""Deterministic, diversity-preferring row selection shared by the
acceptance-manifest and benchmark-plan builders."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from .canonical import semantic_key


def rank(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def row_identity(row: dict[str, Any]) -> str:
    return "\x1f".join(
        str(row.get(key, ""))
        for key in ("scene_id", "kind", "phase", "frame", "type", "question", "sample_id")
    )


def is_centered(row: dict[str, Any]) -> bool:
    """True if the answer bbox centroid is in the middle 50% of both axes.
    In-context rows already carry a precomputed answer_is_centered column
    (built with this exact rule at generation time); normal rows do not, so
    it is computed here from answer_bbox/img_w/img_h."""
    if row.get("answer_is_centered") is not None:
        return bool(row["answer_is_centered"])
    bbox = row.get("answer_bbox")
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / (2 * row["img_w"])
    cy = (y0 + y1) / (2 * row["img_h"])
    return 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75


class DeterministicSelector:
    """Greedy, seeded selection across many (cell -> quota) groups, sharing
    one running state so "prefer distinct scenes/frames" and "no semantic
    alias duplicate" hold across the whole output, not just within one cell.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.used_scene_frame: set[tuple[str, str]] = set()
        self.used_semantic_keys: set[tuple] = set()
        self.used_row_ids: set[str] = set()

    def _rank_key(self, row: dict[str, Any]) -> tuple:
        scene_frame = (str(row["scene_id"]), str(row["frame"]))
        # Rows whose (scene, frame) is already used sort after rows that
        # would introduce a new one -- "prefer distinct scenes/frames" as a
        # soft, deterministic tiebreak, not a hard requirement.
        reused = scene_frame in self.used_scene_frame
        return (reused, rank(self.seed, row_identity(row)))

    def select_cell(self, candidates: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=self._rank_key)
        picked: list[dict[str, Any]] = []
        for row in ordered:
            if len(picked) >= quota:
                break
            row_id = row_identity(row)
            if row_id in self.used_row_ids:
                continue
            key = semantic_key(row)
            if key in self.used_semantic_keys:
                continue
            picked.append(row)
            self.used_row_ids.add(row_id)
            self.used_semantic_keys.add(key)
            self.used_scene_frame.add((str(row["scene_id"]), str(row["frame"])))
        return picked

    def select(
        self,
        candidates_by_cell: dict[Any, list[dict[str, Any]]],
        quota_of: Callable[[Any], int],
    ) -> dict[Any, list[dict[str, Any]]]:
        """Process cells in a fixed, seed-independent order (sorted by cell
        key) so the shared used_* state is applied deterministically."""
        result: dict[Any, list[dict[str, Any]]] = {}
        for cell in sorted(candidates_by_cell, key=str):
            result[cell] = self.select_cell(candidates_by_cell[cell], quota_of(cell))
        return result
