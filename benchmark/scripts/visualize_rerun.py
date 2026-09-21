"""Stream a (scene, phase) of RPX into Rerun as a 6-panel time series.

Layout (2 rows × 3 columns):

    Cam Pose (3D)  |  Fisheye L     |  Fisheye R
    RGB            |  Depth (cividis) |  Masks

All panels share a `frame` sequence axis. The full T265 trajectory is logged
once as a static 3D path; the camera transform sweeps along it as you scrub.

By default the data is streamed straight into the viewer's memory — no
intermediate .rrd file is written, so the only disk usage is the HuggingFace
cache (~/.cache/huggingface/hub/) that holds the dataset itself. Pass --save
if you want a portable .rrd you can re-open or share.

Quick start
-----------
    # From the benchmark/ directory:
    make visualize             # stream only (no disk artifact)
    make visualize-save        # also write a portable .rrd
    make visualize-lite        # constrained: stride=3, jpeg=70
    make visualize-headless    # no viewer, build .rrd for later (CI / SSH)

    # Direct invocation:
    PYTHONPATH=. python scripts/visualize_rerun.py
    PYTHONPATH=. python scripts/visualize_rerun.py --scene scene23.jo.4f --phase 1

A saved .rrd can be reopened any time:
    rerun --memory-limit 2GB benchmark/site/rpx_*.rrd
"""

import argparse
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import rerun as rr
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional — fall back to a no-op iterator

    def tqdm(it, **_kw):
        return it


from rpx_benchmark.dataset_hub import download_for_task

# Depth display range. We log raw uint16 millimeters and ask Rerun to display
# them in mm directly (meter=1.0 means "1 unit of the value = 1 unit of depth"),
# so the hover tooltip reads e.g. "1420 mm" — matches the file format and
# avoids the meter-rounding artefact at typical D435 ranges.
DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 3000
_DEPTH_CMAP = matplotlib.colormaps["cividis"]

# Royal jewel-tone palette for instance masks. Each color is a deep, saturated
# tone (sapphire, emerald, ruby, amethyst, gold, etc.) — designed to look
# regal on a black background while remaining mutually distinguishable at
# small panel sizes. Mapping is deterministic: instance id N → palette[N % 16],
# so the same id always produces the same color across frames and phases
# (same property the Rerun hash gave, but with curated aesthetics).
_ROYAL_PALETTE = [
    (65, 105, 225),  # royal blue
    (218, 165, 32),  # goldenrod
    (199, 21, 133),  # medium violet red (royal magenta)
    (34, 139, 34),  # forest emerald
    (139, 0, 0),  # wine
    (138, 43, 226),  # blue violet (amethyst)
    (0, 128, 128),  # teal
    (255, 215, 0),  # pure gold
    (220, 20, 60),  # crimson
    (75, 0, 130),  # indigo
    (210, 105, 30),  # chocolate / bronze
    (147, 112, 219),  # medium purple
    (255, 140, 0),  # royal orange
    (30, 60, 175),  # sapphire
    (153, 50, 204),  # dark orchid
    (102, 0, 51),  # dark burgundy
]


def instance_annotations(max_label: int = 96):
    """AnnotationInfo list: bg=black; ids ≥ 1 cycle the royal jewel palette."""
    infos = [rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0))]
    for i in range(1, max_label + 1):
        infos.append(
            rr.AnnotationInfo(
                id=i,
                label=f"obj{i:02d}",
                color=_ROYAL_PALETTE[(i - 1) % len(_ROYAL_PALETTE)],
            )
        )
    return infos


def stretch_contrast(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile-stretch a uint8 image to use the full 0–255 range.

    T265 fisheye frames are uint8 but typically only occupy a narrow band
    (e.g. 30–140), so they look very dark in viewers that assume 0–255.
    """
    img = np.asarray(img)
    if img.size == 0:
        return img
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        return img
    out = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return out.astype(np.uint8)


def _equalize(values: np.ndarray) -> np.ndarray:
    """Rank-based histogram equalization → uniform on [0, 1].

    Spreads the depth distribution evenly so the colormap uses its full
    dynamic range. Same shape as input; output is float32 in [0, 1].
    """
    sorted_vals = np.sort(values)
    ranks = np.searchsorted(sorted_vals, values, side="right").astype(np.float32)
    return ranks / float(sorted_vals.size)


def colorize_depth_mm(depth_mm: np.ndarray, sharpen: bool = True) -> np.ndarray:
    """uint16 millimeter depth → cividis-colored uint8 RGB, sharpened.

    Sharpening (default on) uses per-frame histogram equalization, which
    maximises visible depth structure even when the scene's depth band is
    narrow (typical for tabletop walkarounds at 0.5–2 m). Pixels with
    depth==0 (invalid) are rendered black so they read as "missing data".

    Set ``sharpen=False`` to fall back to a fixed [DEPTH_MIN_MM, DEPTH_MAX_MM]
    linear stretch if you need cross-frame visual comparability (same physical
    distance → same color across frames).
    """
    d = depth_mm.astype(np.float32)
    valid = d > 0
    norm = np.zeros_like(d)
    if sharpen and valid.sum() > 100:
        # Equalize within the valid region only; black pixels stay black.
        norm[valid] = _equalize(d[valid])
    else:
        lo, hi = DEPTH_MIN_MM, DEPTH_MAX_MM
        norm[valid] = np.clip((d[valid] - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    rgba = _DEPTH_CMAP(norm)  # float64 (H, W, 4) in [0, 1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0  # invalid pixels → black
    return rgb


class TarReader:
    """Open a tar file once, keep the handle live, and serve members fast.

    Opening tarfile.open() per read scans the tar index every time; for
    250-frame phases that's ~250 redundant scans per modality. This wraps
    one open + a name→TarInfo dict so subsequent reads are O(1).
    """

    def __init__(self, tar_path: Path):
        self.tar_path = tar_path
        self._tf = tarfile.open(tar_path, "r")
        self._index = {m.name: m for m in self._tf.getmembers() if m.isfile()}

    def names(self) -> list[str]:
        return list(self._index.keys())

    def read(self, member_name: str) -> bytes:
        m = self._index.get(member_name)
        if m is None:
            raise FileNotFoundError(f"{member_name} not in {self.tar_path}")
        f = self._tf.extractfile(m)
        if f is None:
            raise FileNotFoundError(f"{member_name} in {self.tar_path} (no data)")
        return f.read()

    def close(self):
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def list_members(tar_path: Path):
    with tarfile.open(tar_path, "r") as tf:
        return [m.name for m in tf if m.isfile()]


def _opt_tar_reader(tar_path: Path | None) -> "TarReader | None":
    """TarReader(path) when path exists, else None."""
    return TarReader(tar_path) if tar_path else None


def find_tar(local_dir: Path, scene: str, phase: str, modality: str) -> Path | None:
    raw = local_dir / "scenes" / scene / phase / f"{modality}.tar"
    if raw.exists():
        return raw
    label_dir = local_dir / "scenes" / scene / phase / "labels" / modality
    if label_dir.exists():
        for v in sorted(label_dir.glob("v*.tar")):
            return v
    return None


def frame_id_of(member_name: str) -> str:
    """rgb/00042.png  →  '00042'  (also handles fisheye/left/00042.png)."""
    return Path(member_name).stem


def members_by_frame(names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(frame_id_of(n), []).append(n)
    return out


def load_image(reader: TarReader, member: str) -> np.ndarray:
    return np.array(Image.open(BytesIO(reader.read(member))))


def load_pose(reader: TarReader, member: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (translation[3], quaternion_xyzw[4]) from a cam_pose npz."""
    npz = np.load(BytesIO(reader.read(member)))
    t = np.asarray(npz["position"], dtype=np.float64)
    q = np.asarray(npz["orientation"], dtype=np.float64)
    return t, q


def main():
    from rpx_benchmark.cleanup import install_signal_cleanup
    install_signal_cleanup()

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n", 2)[0],
        epilog="""\
Examples:
  python scripts/visualize_rerun.py                          # default: stream into viewer (no .rrd)
  python scripts/visualize_rerun.py --save                   # also write a portable .rrd
  python scripts/visualize_rerun.py --lite                   # constrained: stride=3, jpeg=70
  python scripts/visualize_rerun.py --no-spawn --save        # headless: build .rrd only (CI / SSH)
  python scripts/visualize_rerun.py --scene <id> --phase 1   # specific scene/phase
""",
    )
    ap.add_argument(
        "--repo",
        default="anonymous/RPX",
        help="HuggingFace dataset repo (default: %(default)s)",
    )
    ap.add_argument("--scene", default=None, help="scene id (default: first available in cache)")
    ap.add_argument("--phase", default=None, help="phase 0|1|2 (default: first available)")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="cap frames logged (default: %(default)s = full phase)",
    )
    ap.add_argument(
        "--stride", type=int, default=1, help="log every Nth frame (1=all, 3≈80 frames/phase)"
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for RGB/fisheye, 1–100 (default: %(default)s)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="log raw uncompressed images (default: JPEG-compress RGB/fisheye)",
    )
    ap.add_argument(
        "--no-cividis",
        action="store_true",
        help="use raw DepthImage instead of cividis-colored Image",
    )
    ap.add_argument(
        "--keep-raw-depth",
        action="store_true",
        help="also log the raw DepthImage (meters readout) alongside cividis",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="also write a portable .rrd to benchmark/site/ (default: stream-only, no disk artifact)",
    )
    ap.add_argument(
        "--no-spawn", action="store_true", help="don't open the viewer (default: spawn)"
    )
    ap.add_argument(
        "--lite",
        action="store_true",
        help="resource-constrained preset: stride=3, jpeg=70 (Pi 4 / Jetson Nano)",
    )
    ap.add_argument(
        "--all-phases",
        action="store_true",
        help="show all 3 phases (clutter, interaction, clean) side-by-side",
    )
    args = ap.parse_args()

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "visualize_rerun — RPX → Rerun viewer",
        "6-panel time series per (scene, phase)",
    )
    cli_ux.config(vars(args))

    # --lite preset overrides the relevant knobs unless the user already set them.
    if args.lite:
        if args.stride == 1:
            args.stride = 3
        args.jpeg_quality = min(args.jpeg_quality, 70)

    JPEG = None if args.full else args.jpeg_quality

    print(f"== fetching segmentation/easy + depth + fisheye + cam_pose from {args.repo} ==")
    try:
        res = download_for_task(
            task="segmentation",
            split="easy",
            repo_id=args.repo,
            extra_modalities=["depth", "fisheye", "cam_pose"],
        )
    except Exception as e:  # noqa: BLE001 — surface the error cleanly
        sys.exit(
            f"\nFailed to download from HuggingFace ({args.repo}).\n"
            f"  Error: {type(e).__name__}: {e}\n\n"
            "Common causes:\n"
            "  • offline / no internet  → reconnect and retry\n"
            "  • repo is gated/private  → set HF_TOKEN\n"
            "  • disk full              → free ~4 GB in ~/.cache/huggingface/hub/\n"
        )
    local = Path(res.local_dir)
    print(f"  local_dir : {local}")

    # Pick scene/phase.
    candidates = [
        (p.parent.parent.name, p.parent.name) for p in sorted(local.glob("scenes/*/*/rgb.tar"))
    ]
    if args.scene:
        scene_phases = sorted({p for s, p in candidates if s == args.scene})
        if not scene_phases:
            sys.exit(
                f"scene {args.scene} not in cache; available scenes: "
                f"{sorted({s for s, _ in candidates})}"
            )
        scene = args.scene
    else:
        # First scene with at least one fully-modal phase.
        for s, p in candidates:
            if all(find_tar(local, s, p, m) for m in ("depth", "fisheye", "masks")):
                scene = s
                break
        else:
            sys.exit("no scene has all required modalities downloaded")
        scene_phases = sorted({p for s2, p in candidates if s2 == scene})

    # Decide which phase(s) to load.
    if args.all_phases:
        phases = scene_phases  # typically ["0", "1", "2"]
    elif args.phase is not None:
        if args.phase not in scene_phases:
            sys.exit(f"phase {args.phase} not in cache for {scene}; have {scene_phases}")
        phases = [args.phase]
    else:
        phases = [scene_phases[0]]
    print(f"  using     : scene={scene} phases={phases}")

    # Phase metadata: id, human label, trajectory color (3D), short tag (2D).
    PHASE_META = {
        "0": ("Clutter", (180, 50, 200), "P0"),
        "1": ("Interaction", (240, 130, 50), "P1"),
        "2": ("Clean", (60, 200, 220), "P2"),
    }
    multi = len(phases) > 1
    phase = phases[0]  # for backward-compat output paths / app id

    # Per-phase readers + frame indices. In single-phase mode this is a 1-elem dict;
    # in --all-phases it has one entry per phase loaded.
    phase_data: dict[str, dict] = {}
    for ph in phases:
        readers = {
            "rgb": TarReader(find_tar(local, scene, ph, "rgb")),
            "depth": _opt_tar_reader(find_tar(local, scene, ph, "depth")),
            "fisheye": _opt_tar_reader(find_tar(local, scene, ph, "fisheye")),
            "masks": _opt_tar_reader(find_tar(local, scene, ph, "masks")),
            "pose": _opt_tar_reader(find_tar(local, scene, ph, "cam_pose")),
        }
        idx = {k: members_by_frame(r.names()) if r else {} for k, r in readers.items()}
        phase_data[ph] = {"readers": readers, "idx": idx}

    # Frame ids: union across all loaded phases (rgb tar is canonical per phase).
    all_frame_ids = sorted({fid for d in phase_data.values() for fid in d["idx"]["rgb"]})
    if args.stride > 1:
        all_frame_ids = all_frame_ids[:: args.stride]
    if args.max_frames:
        all_frame_ids = all_frame_ids[: args.max_frames]
    frame_ids = all_frame_ids
    print(f"  frames    : {len(frame_ids)} (stride={args.stride}, phases={len(phases)})")
    print(f"  jpeg q    : {JPEG if JPEG else 'off (raw)'}; cividis depth: {not args.no_cividis}")

    # ── Rerun blueprint ──
    if multi:
        # All-phases comparison: each row is a phase, 6 panels per row.
        #   Cam Pose (3D, scoped to phase)  RGB  Depth (cividis)  Masks  Fisheye L  Fisheye R
        # Bottom: frame counter strip.
        def img_view(name, path):
            return rr.blueprint.Spatial2DView(name=name, origin=path)

        def row_for_phase(ph: str) -> rr.blueprint.Horizontal:
            label, _, tag = PHASE_META.get(ph, (f"P{ph}", (200, 200, 200), f"P{ph}"))
            return rr.blueprint.Horizontal(
                rr.blueprint.Spatial3DView(
                    name=f"{tag} · Cam Pose",
                    origin=f"world/p{ph}",
                ),
                img_view(f"{tag} · RGB · {label}", f"view/p{ph}/rgb"),
                img_view(f"{tag} · Depth (cividis)", f"view/p{ph}/depth_cividis"),
                img_view(f"{tag} · Masks", f"view/p{ph}/masks"),
                img_view(f"{tag} · Fisheye L", f"view/p{ph}/fisheye_left"),
                img_view(f"{tag} · Fisheye R", f"view/p{ph}/fisheye_right"),
            )

        bp_layout = rr.blueprint.Vertical(
            *[row_for_phase(ph) for ph in phases],
            rr.blueprint.TextLogView(name="Frame", origin="status"),
            row_shares=[6] * len(phases) + [1],
        )
    else:
        # Single-phase 2×3 grid (original layout).
        masks_cell = rr.blueprint.Vertical(
            rr.blueprint.Spatial2DView(name="Masks", origin=f"view/p{phase}/masks"),
            rr.blueprint.TextLogView(name="Frame", origin="status"),
            row_shares=[6, 1],
        )
        bp_layout = rr.blueprint.Vertical(
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial3DView(name="Cam Pose", origin="world"),
                rr.blueprint.Spatial2DView(name="Fisheye L", origin=f"view/p{phase}/fisheye_left"),
                rr.blueprint.Spatial2DView(name="Fisheye R", origin=f"view/p{phase}/fisheye_right"),
            ),
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial2DView(name="RGB", origin=f"view/p{phase}/rgb"),
                rr.blueprint.Spatial2DView(name="Depth (mm)", origin=f"view/p{phase}/depth"),
                masks_cell,
            ),
        )

    bp = rr.blueprint.Blueprint(
        bp_layout,
        rr.blueprint.BlueprintPanel(state="collapsed"),
        rr.blueprint.SelectionPanel(state="collapsed"),
        rr.blueprint.TimePanel(state="expanded"),
    )

    spawn = not args.no_spawn
    rr.init(f"rpx · {scene}/{phase}", default_blueprint=bp)

    if spawn:
        # rr.init's spawn=True doesn't expose --hide-welcome-screen, so spawn
        # the viewer ourselves with the right flags, then connect over gRPC.
        # This stops the example-gallery overlay from blocking the canvas.
        import shutil
        import subprocess

        rerun_bin = shutil.which("rerun")
        if rerun_bin is None:
            sys.exit("'rerun' CLI not found on PATH — install with: pip install rerun-sdk")
        # Reuse an existing viewer on :9876 if one is already running; else spawn.
        try:
            import socket

            with socket.create_connection(("127.0.0.1", 9876), timeout=0.5):
                pass
        except OSError:
            subprocess.Popen(
                [rerun_bin, "--memory-limit", "2GB", "--port", "9876", "--hide-welcome-screen"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            import time

            time.sleep(1.5)  # give the viewer a moment to bind the port
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
    elif not args.save:
        # --no-spawn AND --no-save: connect to whatever viewer is on :9876,
        # otherwise data has nowhere to go and the user sees an empty viewer.
        try:
            rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        except Exception as e:  # noqa: BLE001
            sys.exit(
                f"--no-spawn was passed but no viewer at 127.0.0.1:9876 ({e}).\n"
                "Either start `rerun` first, drop --no-spawn, or pass --save."
            )
    # Force the new blueprint to override any per-app override the viewer
    # has cached from a previous session
    # (~/.local/share/rerun/blueprints/<app_id>-*.rbl).
    rr.send_blueprint(bp, make_active=True, make_default=True)

    out_rrd = Path(__file__).resolve().parent.parent / "site" / f"rpx_{scene}_{phase}.rrd"
    if args.save:
        out_rrd.parent.mkdir(exist_ok=True)
        rr.save(str(out_rrd), default_blueprint=bp)
        print(f"  saving    : {out_rrd}")

    # Static annotation context per phase: bg=black + vivid palette for instances.
    for ph in phases:
        rr.log(f"view/p{ph}/masks", rr.AnnotationContext(instance_annotations()), static=True)

    # Per-phase static trajectory line + Pinhole frustum.
    for ph in phases:
        pose_r = phase_data[ph]["readers"]["pose"]
        pose_idx = phase_data[ph]["idx"]["pose"]
        if not pose_r:
            continue
        poses = []
        for fid in frame_ids:
            if fid in pose_idx:
                t, q = load_pose(pose_r, pose_idx[fid][0])
                poses.append(t)
        if not poses:
            continue
        traj = np.stack(poses)
        _, color, _ = PHASE_META.get(ph, (f"P{ph}", (200, 80, 200), f"P{ph}"))
        rr.log(
            f"world/p{ph}/trajectory",
            rr.LineStrips3D([traj], colors=[color], radii=0.005),
            static=True,
        )
        rr.log(
            f"world/p{ph}/trajectory_points",
            rr.Points3D(traj, colors=[color], radii=0.008),
            static=True,
        )
        rr.log(
            f"world/p{ph}/camera",
            rr.Pinhole(
                focal_length=617.0,
                width=640,
                height=480,
                principal_point=[320.0, 240.0],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
            static=True,
        )
    print(f"  poses     : logged static trajectories for {len(phases)} phase(s)")

    fisheye_jpeg = max(40, JPEG - 15) if JPEG else None
    total_frames = len(frame_ids)

    for i, fid in enumerate(tqdm(frame_ids, desc="logging frames", unit="fr")):
        rr.set_time("frame", sequence=int(fid))
        rr.log("status", rr.TextLog(f"Frame {i + 1} / {total_frames}  (id={fid})"))

        for ph in phases:
            d = phase_data[ph]
            readers, idx = d["readers"], d["idx"]
            base = f"view/p{ph}"
            world = f"world/p{ph}"

            # Pose: per-frame Transform3D under the phase's camera path.
            if readers["pose"] and fid in idx["pose"]:
                t, q = load_pose(readers["pose"], idx["pose"][fid][0])
                rr.log(
                    f"{world}/camera",
                    rr.Transform3D(
                        translation=t,
                        rotation=rr.Quaternion(xyzw=q),
                        axis_length=0.1,
                    ),
                )

            # RGB.
            if fid in idx["rgb"]:
                rgb = load_image(readers["rgb"], idx["rgb"][fid][0])
                rgb_arch = rr.Image(rgb).compress(jpeg_quality=JPEG) if JPEG else rr.Image(rgb)
                rr.log(f"{base}/rgb", rgb_arch)

            # Depth (uint16 mm → meters via meter=1000, Viridis ≈ cividis).
            if readers["depth"] and fid in idx["depth"]:
                depth = load_image(readers["depth"], idx["depth"][fid][0])
                # meter=1.0 → tooltip displays raw mm values (e.g. "1420 mm"),
                # which matches the on-disk uint16 format. depth_range is in
                # the same unit (mm).
                rr.log(
                    f"{base}/depth",
                    rr.DepthImage(
                        depth,
                        meter=1.0,
                        colormap=rr.components.Colormap.Viridis,
                        depth_range=(DEPTH_MIN_MM, DEPTH_MAX_MM),
                    ),
                )
                # Always log a cividis-colored RGB Image for the per-phase
                # all-phases layout (which displays Depth (cividis)). For
                # single-phase, this only matters with --keep-raw-depth.
                if multi or args.keep_raw_depth:
                    rr.log(f"{base}/depth_cividis", rr.Image(colorize_depth_mm(depth)))

            # Masks.
            if readers["masks"] and fid in idx["masks"]:
                mask = load_image(readers["masks"], idx["masks"][fid][0])
                rr.log(f"{base}/masks", rr.SegmentationImage(mask.astype(np.uint16)))

            # Fisheye stereo: log in both single- and all-phases modes.
            if readers["fisheye"] and fid in idx["fisheye"]:
                for fm in sorted(idx["fisheye"][fid]):
                    lr = (
                        "left"
                        if "left" in fm.lower() or fm.endswith("0.png")
                        else ("right" if "right" in fm.lower() or fm.endswith("1.png") else "img")
                    )
                    fimg = stretch_contrast(load_image(readers["fisheye"], fm))
                    fimg_arch = (
                        rr.Image(fimg).compress(jpeg_quality=fisheye_jpeg)
                        if fisheye_jpeg
                        else rr.Image(fimg)
                    )
                    rr.log(f"{base}/fisheye_{lr}", fimg_arch)

    # Close every tar handle we opened.
    for d in phase_data.values():
        for r in d["readers"].values():
            if r is not None:
                r.close()

    print(f"  done      : logged {len(frame_ids)} frames")
    if args.save:
        try:
            sz = out_rrd.stat().st_size
            print(f"  rrd size  : {sz / 1e6:.1f} MB")
        except FileNotFoundError:
            pass
        print(f"  reopen    : rerun --memory-limit 2GB {out_rrd}")


if __name__ == "__main__":
    main()
