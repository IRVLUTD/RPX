#----------------------------------------------------------------------------------------------------
# Ego (GoPro) annotation — one command per scene, no sync required.
#
#   python -m maskgen_pipeline.annotate_ego --scene_dir /path/to/scene
#
# The ego camera and the D435 rig were recorded on independent clocks, so we
# do NOT attempt to time-sync them. We just uniformly extract 250 frames from
# the entire ego mp4 and save them into <scene>/ego/rgb/NNNNN.png — the same
# filename nomenclature the rig uses (rgb/, depth/, ...). The lab's existing
# interactive_gsam2 labeling tool then takes <scene>/ego as a scene dir.
#
# Work done while being at the Intelligent Robotics and Vision Lab at UT Dallas.
#----------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _resolve_ego_video(scene_dir: Path) -> Optional[Path]:
    """Find the ego mp4 in the scene dir. Looks for common names first."""
    for name in ("ego.mp4", "ego.MP4", "gopro.mp4", "GoPro-Ego.MP4", "ego_video.mp4"):
        cand = scene_dir / name
        if cand.is_file():
            return cand
    mp4s = sorted(scene_dir.glob("*.[mM][pP]4"))
    return mp4s[0] if mp4s else None


def _rig_mask_reference(scene_dir: Path) -> Optional[Path]:
    """Pick a representative rig mask folder to point the operator at for
    consistent object-ID conventions when labeling the ego frames."""
    for phase in ("1", "0", "2", "interaction"):
        cand = scene_dir / phase / "sam2" / "masks"
        if cand.is_dir():
            return cand
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract 250 frames from a scene's ego (GoPro) mp4 into <scene>/ego/rgb/."
    )
    ap.add_argument("--scene_dir", required=True, type=Path,
                    help="Scene directory (output lands at <scene_dir>/ego/rgb/).")
    ap.add_argument("--ego_video", type=Path, default=None,
                    help="Path to the ego mp4. Auto-detected in --scene_dir if omitted.")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="Frames to sample. Default: 250 (fixed for every ego video).")
    ap.add_argument("--auto_label", action="store_true",
                    help="Immediately launch interactive_gsam2 on the sampled ego/ folder.")
    args = ap.parse_args(argv)

    scene_dir: Path = args.scene_dir
    if not scene_dir.is_dir():
        print(f"error: scene dir not found: {scene_dir}", file=sys.stderr)
        return 2

    ego_video = args.ego_video or _resolve_ego_video(scene_dir)
    if ego_video is None or not Path(ego_video).is_file():
        print(f"error: no ego mp4 found in {scene_dir} (pass --ego_video).", file=sys.stderr)
        return 2

    print(f"[annotate-ego] scene     : {scene_dir}")
    print(f"[annotate-ego] ego video : {ego_video}")

    # ---- Step 1: EXTRACT 250 frames from the whole ego mp4 ----
    from .ego_sample import sample_ego_frames

    written = sample_ego_frames(scene_dir, ego_video, n_samples=args.n_samples)
    ego_dir = scene_dir / "ego"
    print(f"[annotate-ego] wrote {len(written)} frames → {ego_dir / 'rgb'}")

    # ---- Step 2: LABEL (hand off to the existing pipeline) ----
    rig_ref = _rig_mask_reference(scene_dir)
    print("\n" + "=" * 70)
    print("  NEXT STEP — label the objects in the ego frames")
    print("=" * 70)
    print("  The ego frames are now in a folder the lab's normal labeling tool")
    print("  understands. Give each object the SAME id it has in the rig view")
    print("  so scene-level object identity stays consistent across cameras.")
    print()
    print("  Run:")
    print(f"    python -m maskgen_pipeline.interactive_gsam2 --scene_dir {ego_dir}")
    if rig_ref is not None:
        print()
        print(f"  Reference (rig masks, any phase): {rig_ref}")
    print("=" * 70 + "\n")

    if args.auto_label:
        from .interactive_gsam2 import main as label_main
        print("[annotate-ego] launching interactive_gsam2 on the ego folder ...")
        label_main(str(ego_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
