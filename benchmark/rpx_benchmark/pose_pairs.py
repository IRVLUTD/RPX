"""On-the-fly deterministic pose-pair generation for RPX-RCPE.

Generates ~60K pairs across three complementary types — intra-phase,
cross-phase (Clutter↔Clean), and temporal chains — without writing
anything to disk.  Given the same seed and config the output is
bit-identical, so results are fully reproducible.

The generator produces lightweight sample dicts (frame indices + paths)
that plug directly into :class:`RPXDataset.from_dict`.  Actual pixel
data (RGB, depth) is loaded lazily by the dataset iterator, not here.

Usage
-----
    from rpx_benchmark.pose_pairs import PosePairGenerator, PairConfig

    gen = PosePairGenerator(
        extracted_root="/path/to/extracted",
        parquet_path="/path/to/frames_v1.parquet",
        split="easy",
    )
    manifest = gen.manifest()           # dict consumable by RPXDataset.from_dict
    dataset  = gen.as_dataset(batch_size=1)  # ready for BenchmarkRunner

    # Or iterate pairs without building the full manifest:
    for pair in gen.iter_pairs():
        print(pair["id"], pair["pair_type"], pair["rotation_bin"])
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tar extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_tar(
    tar_path: Path, member_name: str, out_path: Path,
) -> None:
    """Extract a single member from a tar to ``out_path``. Idempotent."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        f = tf.extractfile(member_name)
        if f is None:
            raise FileNotFoundError(
                f"{member_name} not found in {tar_path}"
            )
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as g:
            g.write(f.read())
        tmp.rename(out_path)

RPX_SEED = 5_062_026

# ─────────────────────────────────────────────────────────────────────────────
# Exclusions
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED_SCENE_IDS: Set[str] = {"scene58"}
"""Scenes excluded entirely — no ICP-optimized poses available."""

EXCLUDED_SCENE_PHASES: Set[Tuple[str, int]] = {
    ("scene83.jsom.garden.pot", 0),
    ("scene98.jsom.atrium", 0),
    ("scene100.jsom.atrium", 0),
    ("scene50.ecss.out.stairs", 2),
    ("scene72.ecsw.atriumStairs", 2),
}
"""(scene_id, phase) pairs where ICP optimization failed."""

VALID_PHASES: Tuple[int, ...] = (0, 2)
"""Clutter and Clean; Interaction is excluded from the RCPE protocol."""


def _mode_split(series: pd.Series) -> str | None:
    """Return the most common non-null split, or ``None`` if unspecified."""
    counts = series.dropna().value_counts()
    return str(counts.idxmax()) if not counts.empty else None

# ─────────────────────────────────────────────────────────────────────────────
# Rotation bins
# ─────────────────────────────────────────────────────────────────────────────

ROTATION_BINS: List[Tuple[float, float]] = [
    (0.0, 15.0),
    (15.0, 45.0),
    (45.0, 90.0),
    (90.0, 180.000001),
]
BIN_NAMES: List[str] = ["easy", "medium", "hard", "extreme"]


def _bin_index(rot_deg: float) -> int:
    """Return bin index for a rotation, or -1 if below minimum."""
    for i, (lo, hi) in enumerate(ROTATION_BINS):
        if lo <= rot_deg < hi:
            return i
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Pose math
# ─────────────────────────────────────────────────────────────────────────────

def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _load_pose_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an RPX camera pose as ``(rotation, translation)``.

    Published RPX snapshots store compact ``.npy`` vectors as
    ``[x, y, z, qx, qy, qz, qw]``.  Legacy extracted trees may still contain
    ``.npz`` files with named ``position`` and ``orientation`` arrays.
    """
    data = np.load(path)
    if path.suffix.lower() == ".npy":
        values = np.asarray(data, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError(f"camera pose {path} must have shape (7,), got {values.shape}")
        translation = values[:3]
        quaternion = values[3:]
    else:
        translation = np.asarray(data["position"], dtype=np.float64).reshape(3)
        quaternion = np.asarray(data["orientation"], dtype=np.float64).reshape(4)
    return _quat_xyzw_to_rot(quaternion), translation


def _rotation_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos_t = (np.trace(R_a.T @ R_b) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairConfig:
    """All knobs for pair generation.  Deterministic given ``seed``."""

    # Intra-phase
    intra_pairs_per_bin: int = 50
    frame_gap: int = 5
    # D6 retains every available t -> t+5 pair. The first rotation bin
    # therefore starts at zero instead of dropping low-motion pairs.
    min_rotation_deg: float = 0.0
    min_translation_m: float = 0.0

    # Cross-phase (Clutter ↔ Clean)
    cross_pairs_per_bin: int = 30
    cross_max_candidates: int = 5000

    # Temporal chains
    chain_count: int = 5
    chain_length: int = 10
    chain_stride: int = 5

    seed: int = RPX_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SeqPoses:
    """Loaded poses + frame indices for one (scene, phase)."""
    scene: str
    phase: int
    frame_idxs: List[int]
    rotations: List[np.ndarray]   # 3×3 each
    translations: List[np.ndarray]  # (3,) each


class PosePairGenerator:
    """Deterministic on-the-fly pair generator.

    Loads only poses (NPZ, lightweight) at construction time.
    Pixel data is never touched — that happens downstream in the
    :class:`RPXDataset` iterator.

    Parameters
    ----------
    extracted_root : Path
        Root of the extracted frames tree (contains ``scenes/…``).
    parquet_path : Path
        ``frames_v1.parquet`` with scene/phase/frame metadata.
    split : str
        ``"easy"`` | ``"medium"`` | ``"hard"``.
    config : PairConfig, optional
        Generation parameters.  Defaults are the standard benchmark
        settings (~60K pairs at full dataset scale).
    snapshot_root : Path, optional
        HF snapshot root (parent of ``manifest/`` and tar shards).
        When provided, cam_pose files are auto-extracted from tars
        if not already on disk.  If omitted, files must already be
        extracted under ``extracted_root``.
    """

    def __init__(
        self,
        extracted_root: Path | str,
        parquet_path: Path | str,
        split: str,
        config: PairConfig | None = None,
        snapshot_root: Path | str | None = None,
        repo_id: str | None = None,
    ) -> None:
        self.root = Path(extracted_root)
        self.split = split
        self.cfg = config or PairConfig()
        self._snapshot_root = Path(snapshot_root) if snapshot_root else None
        self._repo_id = repo_id

        self._sequences: Dict[Tuple[str, int], _SeqPoses] = {}
        self._frame_filenames: Dict[Tuple[str, int, int], str] = {}
        self._cross_scenes: List[str] = []
        self._load_poses(Path(parquet_path))

    # ── pose loading (one-time) ───────────────────────────────────────────

    def _load_poses(self, parquet_path: Path) -> None:
        df = pd.read_parquet(parquet_path)
        self._df_cache = df  # kept for ensure_pairs_extracted

        # Scene-wise split assignment
        scene_split = df.groupby("scene_id")["split"].agg(_mode_split).dropna()
        df = df.drop(columns=["split"]).merge(
            scene_split.rename("split"), left_on="scene_id", right_index=True
        )
        df = df[
            (df["split"].astype(str) == self.split)
            & df["has_cam_pose"].fillna(False).astype(bool)
        ]

        for (scene, phase), grp in df.groupby(["scene_id", "phase"], sort=True):
            phase = int(phase)
            if scene in EXCLUDED_SCENE_IDS:
                continue
            if (scene, phase) in EXCLUDED_SCENE_PHASES:
                continue
            if phase not in VALID_PHASES:
                continue

            grp = grp.sort_values("frame_idx").reset_index(drop=True)
            frame_idxs, rots, trans = [], [], []

            # Auto-extract cam_pose from tar shards if needed
            self._ensure_cam_pose_extracted(scene, phase, grp)

            for _, row in grp.iterrows():
                frame_filename = str(row["frame_filename"])
                stem = frame_filename.rsplit(".", 1)[0]
                pose_dir = self.root / "scenes" / scene / str(phase) / "cam_pose"
                npy_path = pose_dir / f"{stem}.npy"
                npz_path = pose_dir / f"{stem}.npz"
                pose_path = npy_path if npy_path.exists() else npz_path
                rotation, translation = _load_pose_file(pose_path)
                frame_idx = int(row["frame_idx"])
                frame_idxs.append(frame_idx)
                rots.append(rotation)
                trans.append(translation)
                self._frame_filenames[(str(scene), phase, frame_idx)] = frame_filename

            self._sequences[(scene, phase)] = _SeqPoses(
                scene=scene, phase=phase,
                frame_idxs=frame_idxs, rotations=rots, translations=trans,
            )

        # Scenes that have BOTH phase 0 and phase 2 (for cross-phase pairs)
        scenes_0 = {s for (s, p) in self._sequences if p == 0}
        scenes_2 = {s for (s, p) in self._sequences if p == 2}
        self._cross_scenes = sorted(scenes_0 & scenes_2)

        log.info(
            "loaded poses: %d sequences, %d cross-phase scenes, split=%s",
            len(self._sequences), len(self._cross_scenes), self.split,
        )

    # ── tar extraction ─────────────────────────────────────────────────────

    def _ensure_cam_pose_extracted(
        self, scene: str, phase: int, grp: pd.DataFrame,
    ) -> None:
        """Extract cam_pose NPZ files from tar shards if not on disk.

        If a tar shard is missing locally (partial HF cache), it is
        downloaded via ``huggingface_hub`` before extraction.
        """
        if self._snapshot_root is None:
            return
        if "shard_cam_pose" not in grp.columns:
            return

        for _, row in grp.iterrows():
            stem = str(row["frame_filename"]).rsplit(".", 1)[0]
            pose_dir = (
                self.root / "scenes" / scene / str(phase)
                / "cam_pose"
            )
            out_path = pose_dir / f"{stem}.npy"
            if out_path.exists() or (pose_dir / f"{stem}.npz").exists():
                continue
            shard_rel = str(row["shard_cam_pose"])
            tar_path = self._snapshot_root / shard_rel
            if not tar_path.exists():
                tar_path = self._download_shard(shard_rel)
                if tar_path is None:
                    raise FileNotFoundError(
                        f"cam_pose shard not found and download failed: {shard_rel}"
                    )
            member = f"cam_pose/{stem}.npy"
            _extract_from_tar(tar_path, member, out_path)

    def _download_shard(self, shard_rel: str) -> Optional[Path]:
        """Download a single tar shard from HuggingFace Hub.

        Returns the local path on success, None on failure.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            log.warning(
                "huggingface_hub not installed — cannot download shard %s. "
                "Install with: pip install 'rpx-benchmark[hub]'",
                shard_rel,
            )
            return None

        repo_id = self._repo_id or "IRVLUTD/rpx-benchmark"
        log.info("downloading shard %s from %s ...", shard_rel, repo_id)
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=shard_rel,
                repo_type="dataset",
            )
            return Path(local)
        except Exception as e:  # noqa: BLE001
            log.warning("shard download failed for %s: %s", shard_rel, e)
            return None

    def ensure_pairs_extracted(self, samples: Optional[List[Dict]] = None) -> None:
        """Extract RGB for all selected pairs so the dataset iterator works.

        Call after :meth:`pairs` or :meth:`manifest`.  Idempotent —
        skips files already on disk.
        """
        if self._snapshot_root is None:
            return
        if samples is None:
            samples = self.pairs()

        df = self._df_cache if hasattr(self, "_df_cache") else None
        if df is None:
            return

        # Build a lookup: (scene, phase, stem) → shard_rgb
        if not hasattr(self, "_shard_lookup"):
            lookup: Dict[Tuple[str, int, str], str] = {}
            for _, row in df.iterrows():
                s = str(row["scene_id"])
                p = int(row["phase"])
                stem = str(row["frame_filename"]).rsplit(".", 1)[0]
                if "shard_rgb" in row.index:
                    lookup[(s, p, stem)] = str(row["shard_rgb"])
            self._shard_lookup = lookup

        for s in samples:
            for path_key in ("rgb", "rgb_b"):
                rel = s.get(path_key, "")
                out_path = self.root / rel
                if out_path.exists():
                    continue
                parts = Path(rel).parts
                if len(parts) < 5:
                    continue
                scene, phase_s, _mod, fname = parts[1], parts[2], parts[3], parts[4]
                stem = fname.rsplit(".", 1)[0]
                shard = self._shard_lookup.get((scene, int(phase_s), stem))
                if shard is None:
                    continue
                tar_path = self._snapshot_root / shard
                if not tar_path.exists():
                    downloaded = self._download_shard(shard)
                    if downloaded is None:
                        continue
                    tar_path = downloaded
                member = f"rgb/{fname}"
                _extract_from_tar(tar_path, member, out_path)

    # ── pair generators ───────────────────────────────────────────────────

    def _intra_pairs(
        self, seq: _SeqPoses, rng: np.random.Generator,
    ) -> List[Dict[str, Any]]:
        """Stratified random intra-phase pairs for one sequence."""
        cfg = self.cfg
        n = len(seq.frame_idxs)
        # Collect candidates per bin
        bins: List[List[Tuple[int, int, float, float]]] = [[] for _ in BIN_NAMES]
        index_by_frame = {frame: index for index, frame in enumerate(seq.frame_idxs)}
        for i, frame_a in enumerate(seq.frame_idxs):
            frame_b = frame_a + cfg.frame_gap
            j = index_by_frame.get(frame_b)
            if j is None:
                continue
            rot = _rotation_deg(seq.rotations[i], seq.rotations[j])
            if rot < cfg.min_rotation_deg:
                continue
            t_m = float(np.linalg.norm(seq.translations[j] - seq.translations[i]))
            if t_m < cfg.min_translation_m:
                continue
            bi = _bin_index(rot)
            if bi >= 0:
                bins[bi].append((frame_a, frame_b, rot, t_m))

        out: List[Dict[str, Any]] = []
        for bi, name in enumerate(BIN_NAMES):
            cands = bins[bi]
            if not cands:
                continue
            k = min(len(cands), cfg.intra_pairs_per_bin)
            sel = rng.choice(len(cands), size=k, replace=False)
            for idx in sel:
                fa, fb, rot, t_m = cands[idx]
                out.append(self._make_entry(
                    seq.scene, seq.phase, fa, seq.phase, fb,
                    rot, t_m, "intra_phase", name,
                ))
        return out

    def _cross_pairs(
        self, scene: str, rng: np.random.Generator,
    ) -> List[Dict[str, Any]]:
        """Stratified random cross-phase pairs (phase 0 ↔ phase 2)."""
        cfg = self.cfg
        s0 = self._sequences[(scene, 0)]
        s2 = self._sequences[(scene, 2)]
        n0, n2 = len(s0.frame_idxs), len(s2.frame_idxs)
        total = n0 * n2

        # Subsample candidate index pairs for speed
        if total > cfg.cross_max_candidates:
            flat = rng.choice(total, size=cfg.cross_max_candidates, replace=False)
            ii, jj = flat // n2, flat % n2
        else:
            ii = np.repeat(np.arange(n0), n2)
            jj = np.tile(np.arange(n2), n0)

        bins: List[List[Tuple[int, int, float, float]]] = [[] for _ in BIN_NAMES]
        for idx in range(len(ii)):
            i, j = int(ii[idx]), int(jj[idx])
            rot = _rotation_deg(s0.rotations[i], s2.rotations[j])
            if rot < cfg.min_rotation_deg:
                continue
            t_m = float(np.linalg.norm(s2.translations[j] - s0.translations[i]))
            bi = _bin_index(rot)
            if bi >= 0:
                bins[bi].append((s0.frame_idxs[i], s2.frame_idxs[j], rot, t_m))

        out: List[Dict[str, Any]] = []
        for bi, name in enumerate(BIN_NAMES):
            cands = bins[bi]
            if not cands:
                continue
            k = min(len(cands), cfg.cross_pairs_per_bin)
            sel = rng.choice(len(cands), size=k, replace=False)
            for idx in sel:
                fa, fb, rot, t_m = cands[idx]
                out.append(self._make_entry(
                    scene, 0, fa, 2, fb,
                    rot, t_m, "cross_phase", name,
                ))
        return out

    def _temporal_chains(
        self, seq: _SeqPoses, rng: np.random.Generator,
    ) -> List[Dict[str, Any]]:
        """Ordered chains of consecutive pairs for drift evaluation."""
        cfg = self.cfg
        index_by_frame = {frame: index for index, frame in enumerate(seq.frame_idxs)}
        starts = [
            frame
            for frame in seq.frame_idxs
            if all(
                frame + step * cfg.chain_stride in index_by_frame
                for step in range(cfg.chain_length + 1)
            )
        ]
        if not starts:
            return []
        k = min(cfg.chain_count, len(starts))
        starts = [starts[index] for index in rng.choice(len(starts), size=k, replace=False)]

        out: List[Dict[str, Any]] = []
        for ci, start_frame in enumerate(starts):
            for pi in range(cfg.chain_length):
                frame_a = start_frame + pi * cfg.chain_stride
                frame_b = start_frame + (pi + 1) * cfg.chain_stride
                i = index_by_frame[frame_a]
                j = index_by_frame[frame_b]
                rot = _rotation_deg(seq.rotations[i], seq.rotations[j])
                t_m = float(np.linalg.norm(seq.translations[j] - seq.translations[i]))
                entry = self._make_entry(
                    seq.scene, seq.phase, seq.frame_idxs[i],
                    seq.phase, seq.frame_idxs[j],
                    rot, t_m, "temporal_chain", None,
                )
                # Override ID to include chain index (avoids duplicates
                # when two chains share a frame pair)
                sa = f"{seq.frame_idxs[i]:05d}"
                sb = f"{seq.frame_idxs[j]:05d}"
                entry["id"] = f"{seq.scene}__{seq.phase}__c{ci}_{pi}__{sa}__{sb}"
                entry["metadata"]["chain_id"] = (
                    f"{seq.scene}__{seq.phase}__chain{ci}"
                )
                entry["metadata"]["chain_position"] = pi
                out.append(entry)
        return out

    # ── entry builder ─────────────────────────────────────────────────────

    def _make_entry(
        self,
        scene: str,
        phase_a: int, frame_a: int,
        phase_b: int, frame_b: int,
        rot_deg: float, t_m: float,
        pair_type: str, rotation_bin: Optional[str],
    ) -> Dict[str, Any]:
        frame_filenames = getattr(self, "_frame_filenames", {})
        filename_a = frame_filenames.get(
            (str(scene), phase_a, frame_a), f"{frame_a:05d}.png"
        )
        filename_b = frame_filenames.get(
            (str(scene), phase_b, frame_b), f"{frame_b:05d}.png"
        )
        sa = Path(filename_a).stem
        sb = Path(filename_b).stem
        if pair_type == "cross_phase":
            pair_id = f"{scene}__cross__0_{sa}__2_{sb}"
        elif pair_type == "temporal_chain":
            pair_id = f"{scene}__{phase_a}__tc__{sa}__{sb}"
        else:
            pair_id = f"{scene}__{phase_a}__{sa}__{sb}"

        return {
            "id": pair_id,
            "scene_id": scene,
            "phase": phase_a,
            "phase_b": phase_b,
            "frame_idx": frame_a,
            "frame_idx_b": frame_b,
            "difficulty": None,  # filled by split
            "pair_type": pair_type,
            "rotation_bin": rotation_bin,
            "metadata": {
                "scene_id": scene,
                "phase_idx": phase_a,
                "phase_idx_b": phase_b,
                "frame": sa,
                "frame_b": sb,
                "pair_type": pair_type,
                "rotation_bin": rotation_bin,
                "rotation_deg_gt": rot_deg,
                "translation_m_gt": t_m,
            },
            "rgb": f"scenes/{scene}/{phase_a}/rgb/{filename_a}",
            "rgb_b": f"scenes/{scene}/{phase_b}/rgb/{filename_b}",
            "pose_a": self._pose_relative_path(scene, phase_a, sa),
            "pose_b": self._pose_relative_path(scene, phase_b, sb),
        }

    def _pose_relative_path(self, scene: str, phase: int, stem: str) -> str:
        """Return the published NPY path, retaining legacy NPZ compatibility."""
        relative_dir = Path("scenes") / scene / str(phase) / "cam_pose"
        root = getattr(self, "root", None)
        if root is not None and (root / relative_dir / f"{stem}.npz").exists():
            return str(relative_dir / f"{stem}.npz")
        return str(relative_dir / f"{stem}.npy")

    # ── public API ────────────────────────────────────────────────────────

    def iter_pairs(self) -> Iterator[Dict[str, Any]]:
        """Yield pair dicts in deterministic order.  Zero disk writes."""
        rng = np.random.default_rng(self.cfg.seed)

        for key in sorted(self._sequences):
            seq = self._sequences[key]
            for p in self._intra_pairs(seq, rng):
                p["difficulty"] = self.split
                yield p

        for scene in self._cross_scenes:
            for p in self._cross_pairs(scene, rng):
                p["difficulty"] = self.split
                yield p

        for key in sorted(self._sequences):
            seq = self._sequences[key]
            for p in self._temporal_chains(seq, rng):
                p["difficulty"] = self.split
                yield p

    def pairs(self) -> List[Dict[str, Any]]:
        """Materialise all pairs as a list (for manifest / stats)."""
        return list(self.iter_pairs())

    def manifest(self) -> Dict[str, Any]:
        """Build a manifest dict consumable by ``RPXDataset.from_dict``.

        No files are written — call ``json.dump`` yourself if you want
        to cache it, or pass directly to the dataset constructor.
        """
        samples = self.pairs()
        stats = self._compute_stats(samples)
        return {
            "task": "relative_camera_pose",
            "split": self.split,
            "root": str(self.root),
            "samples": samples,
            "_sampler": {
                "name": "stratified_v2",
                "version": "2.0",
                "rotation_bins_deg": [[lo, hi] for lo, hi in ROTATION_BINS],
                "valid_phases": list(VALID_PHASES),
                "seed": self.cfg.seed,
                "intra_pairs_per_bin": self.cfg.intra_pairs_per_bin,
                "cross_pairs_per_bin": self.cfg.cross_pairs_per_bin,
                "chain_count": self.cfg.chain_count,
                "chain_length": self.cfg.chain_length,
                "chain_stride": self.cfg.chain_stride,
                "frame_gap": self.cfg.frame_gap,
                "excluded_scenes": sorted(EXCLUDED_SCENE_IDS),
                "excluded_scene_phases": [
                    f"{s}/{p}" for s, p in sorted(EXCLUDED_SCENE_PHASES)
                ],
                **stats,
            },
        }

    def as_dataset(self, batch_size: int = 1):
        """Return an :class:`RPXDataset` ready for the runner.

        Automatically extracts RGB files from tar shards for the
        selected pairs (idempotent, skips existing files).
        """
        from .loader import RPXDataset  # noqa: PLC0415

        m = self.manifest()
        self.ensure_pairs_extracted(m["samples"])
        return RPXDataset.from_dict(m, batch_size=batch_size)

    @staticmethod
    def _compute_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_type = {"intra_phase": 0, "cross_phase": 0, "temporal_chain": 0}
        by_bin = {n: 0 for n in BIN_NAMES}
        scenes = set()
        for s in samples:
            by_type[s["pair_type"]] += 1
            if s.get("rotation_bin"):
                by_bin[s["rotation_bin"]] += 1
            scenes.add(s["scene_id"])
        return {
            "total_pairs": len(samples),
            "intra_pairs": by_type["intra_phase"],
            "cross_pairs": by_type["cross_phase"],
            "temporal_pairs": by_type["temporal_chain"],
            "pairs_by_bin": by_bin,
            "n_scenes": len(scenes),
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        m = self.manifest()
        s = m["_sampler"]
        lines = [
            f"RPX-RCPE Pair Generator — split={self.split}",
            f"  Scenes:           {s['n_scenes']}",
            f"  Intra-phase:      {s['intra_pairs']:,}",
            f"  Cross-phase:      {s['cross_pairs']:,}",
            f"  Temporal chains:  {s['temporal_pairs']:,}",
            f"  ──────────────────────",
            f"  TOTAL:            {s['total_pairs']:,}",
            f"  By bin: {s['pairs_by_bin']}",
            f"  Seed:             {s['seed']}",
        ]
        return "\n".join(lines)
