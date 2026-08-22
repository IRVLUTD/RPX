"""Tests for ego (egocentric/GoPro) scene support across the dataset hub:
recipes, scanner, packer, manifest split assignment, and the
``ego_layout`` capture-arrangement adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.ego_layout import (
    EXCLUDED_META_FILES,
    arrange_ego_scene,
    prepare_ego_layout,
    resolve_mos_scene_id,
)
from rpx_benchmark.dataset_hub.manifest import build_frame_manifest, read_frame_manifest
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.recipes import (
    EGO_TASK_RECIPES,
    SceneType,
    all_recipe_names,
    resolve_recipe,
)
from rpx_benchmark.dataset_hub.scanner import scan_capture_root


@pytest.fixture
def ego_mock(tmp_path: Path) -> Path:
    return generate_mock(
        tmp_path / "ds",
        MockSpec(
            multi_object_scenes=3,
            single_object_scenes=2,
            phases_per_multi=3,
            frames_per_phase=4,
            ego_scenes=2,  # scene1 + scene2 get ego siblings; scene3 doesn't
        ),
    )


# --------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------- #


def test_ego_recipes_exist():
    assert "ego_segmentation" in EGO_TASK_RECIPES
    assert "ego_object_tracking" in EGO_TASK_RECIPES
    assert "ego_vqa" in EGO_TASK_RECIPES
    for rec in EGO_TASK_RECIPES.values():
        assert rec.scene_type is SceneType.EGO


def test_resolve_recipe_ego():
    rec = resolve_recipe("ego_segmentation", scene_type=SceneType.EGO)
    assert rec.name == "ego_segmentation"
    # Unscoped lookup also finds it (no name collision with mos/sos tables).
    assert resolve_recipe("ego_segmentation").scene_type is SceneType.EGO


def test_all_recipe_names_includes_ego():
    names = set(all_recipe_names())
    assert "ego_segmentation" in names
    assert "ego_object_tracking" in names
    assert "ego_vqa" in names


def test_ego_vqa_recipe_reserved_slot():
    """Mirrors mos's vqa reserved slot — same rgb->vqa+questionnaire shape,
    same "not yet wired" status, just scoped to ego scenes."""
    rec = EGO_TASK_RECIPES["ego_vqa"]
    assert "vqa" in rec.labels
    assert "questionnaire" in rec.labels
    assert "v1.x" in rec.notes


# --------------------------------------------------------------------- #
# ego_layout — scene_id resolution
# --------------------------------------------------------------------- #


def test_resolve_mos_scene_id_returns_egos_own_bare_form(tmp_path: Path):
    # The canonical scene_id is ego's own bare/padded name — verified to
    # match the live IRVLUTD/RPX repo's convention (bare scene001..scene100)
    # — NOT whatever the local mos/ directory happens to be named (which
    # can carry a location suffix that's a local/staging naming quirk).
    # mos_root is only used to validate a sibling exists by scene number.
    mos_root = tmp_path / "mos"
    (mos_root / "scene20.su.checkerboard").mkdir(parents=True)
    (mos_root / "scene7.library.fountain").mkdir(parents=True)
    assert resolve_mos_scene_id(mos_root, "scene020") == "scene020"
    assert resolve_mos_scene_id(mos_root, "scene007") == "scene007"


def test_resolve_mos_scene_id_no_match_returns_none(tmp_path: Path):
    mos_root = tmp_path / "mos"
    (mos_root / "scene1.foo").mkdir(parents=True)
    assert resolve_mos_scene_id(mos_root, "scene099") is None


def test_resolve_mos_scene_id_ambiguous_returns_none(tmp_path: Path):
    # Two dirs both starting with "scene1" at the number-boundary — e.g.
    # "scene1.foo" and "scene1.bar" both parse as scene number 1 twice;
    # ambiguity should refuse to guess rather than pick one silently.
    mos_root = tmp_path / "mos"
    (mos_root / "scene1.foo").mkdir(parents=True)
    (mos_root / "scene1.bar").mkdir(parents=True)
    assert resolve_mos_scene_id(mos_root, "scene001") is None


# --------------------------------------------------------------------- #
# ego_layout — arrangement + exclusions
# --------------------------------------------------------------------- #


def _make_raw_ego_scene(root: Path, name: str) -> Path:
    """A minimal raw ego/scene<N>/ego/ capture, including the files that
    should be excluded from the arranged tree."""
    d = root / name / "ego"
    (d / "rgb").mkdir(parents=True)
    (d / "rgb" / "00000.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    sam2 = d / "sam2"
    (sam2 / "masks").mkdir(parents=True)
    (sam2 / "masks" / "00000.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (sam2 / "masks_verified").mkdir()
    (sam2 / "masks_verified" / "00000.png").write_bytes(b"fake")
    (sam2 / "mask_to_object.json").write_text("{}", encoding="utf-8")
    (sam2 / "verified_masks.txt").write_text("00000\n", encoding="utf-8")
    (sam2 / "ref_frame.txt").write_text("00000\n", encoding="utf-8")
    (sam2 / "mask_to_object_multi.json").write_text("{}", encoding="utf-8")
    return d


def test_arrange_ego_scene_excludes_pipeline_internal_files(tmp_path: Path):
    ego_scene_src = _make_raw_ego_scene(tmp_path / "raw_ego", "scene001")
    dst_root = tmp_path / "DATA" / "ego"
    dst_phase = arrange_ego_scene(ego_scene_src, dst_root, "scene1.mock", mode="symlink")

    assert (dst_phase / "rgb" / "00000.png").exists()
    assert (dst_phase / "sam2" / "masks" / "00000.png").exists()
    assert (dst_phase / "sam2" / "mask_to_object.json").exists()
    assert (dst_phase / "sam2" / "verified_masks.txt").exists()

    for excluded in EXCLUDED_META_FILES:
        assert not (dst_phase / "sam2" / excluded).exists()
    # masks_verified/ excluded by default (no MOS precedent).
    assert not (dst_phase / "sam2" / "masks_verified").exists()


def test_arrange_ego_scene_include_masks_verified_opt_in(tmp_path: Path):
    ego_scene_src = _make_raw_ego_scene(tmp_path / "raw_ego", "scene001")
    dst_root = tmp_path / "DATA" / "ego"
    dst_phase = arrange_ego_scene(
        ego_scene_src, dst_root, "scene1.mock", include_masks_verified=True, mode="symlink"
    )
    assert (dst_phase / "sam2" / "masks_verified" / "00000.png").exists()


def test_arrange_ego_scene_symlink_mode_does_not_copy_bytes(tmp_path: Path):
    ego_scene_src = _make_raw_ego_scene(tmp_path / "raw_ego", "scene001")
    dst_root = tmp_path / "DATA" / "ego"
    dst_phase = arrange_ego_scene(ego_scene_src, dst_root, "scene1.mock", mode="symlink")
    assert (dst_phase / "rgb").is_symlink()


def test_prepare_ego_layout_end_to_end(tmp_path: Path):
    raw_ego_root = tmp_path / "raw_ego"
    _make_raw_ego_scene(raw_ego_root, "scene001")
    _make_raw_ego_scene(raw_ego_root, "scene099")  # no mos sibling on purpose

    mos_root = tmp_path / "mos"
    (mos_root / "scene1.mock.loc").mkdir(parents=True)

    dst = tmp_path / "DATA"
    report = prepare_ego_layout(raw_ego_root, mos_root, dst)

    ok_ids = {r.ego_scene_dir: r.scene_id for r in report.ok}
    assert ok_ids == {"scene001": "scene001"}  # bare form, not mos_root's suffixed name

    skipped = {r.ego_scene_dir: r.status for r in report.skipped}
    assert skipped["scene099"] == "no_mos_sibling"


# --------------------------------------------------------------------- #
# scanner + packer, via the mock tree
# --------------------------------------------------------------------- #


def test_scanner_recognises_ego_scene_type(ego_mock: Path):
    res = scan_capture_root(ego_mock)
    ego_scenes = res.by_type(SceneType.EGO)
    assert len(ego_scenes) == 2
    assert all(s.scene_type is SceneType.EGO for s in ego_scenes)
    ids = {s.scene_id for s in ego_scenes}
    assert ids == {"scene1", "scene2"}  # mock's mos naming has no location suffix


def test_ego_scene_has_one_phase(ego_mock: Path):
    res = scan_capture_root(ego_mock)
    ego_scene = res.by_type(SceneType.EGO)[0]
    assert len(ego_scene.phases) == 1
    assert ego_scene.phases[0].phase_index == 0


def test_packer_writes_ego_scenes_under_ego_root(tmp_path: Path, ego_mock: Path):
    scan = scan_capture_root(ego_mock)
    plan = PackPlan(src_root=ego_mock, staging_root=tmp_path / "stage")
    result = pack_capture_tree(plan, scan)

    ego_shards = [s for s in result.shards if s.repo_path.startswith("ego/")]
    assert ego_shards, "expected at least one ego/ shard"
    for s in ego_shards:
        assert s.repo_path.startswith(f"ego/{s.scene_id}/0/")
    # mos/sos shard roots are unaffected (still "scenes/" and "objects/").
    assert any(s.repo_path.startswith("scenes/") for s in result.shards)
    assert any(s.repo_path.startswith("objects/") for s in result.shards)


# --------------------------------------------------------------------- #
# manifest — split assignment
# --------------------------------------------------------------------- #


def test_manifest_assigns_split_to_ego_scene_matching_mos_sibling(tmp_path: Path, ego_mock: Path):
    scan = scan_capture_root(ego_mock)
    plan = PackPlan(src_root=ego_mock, staging_root=tmp_path / "stage")
    pack = pack_capture_tree(plan, scan)

    splits = {"scene1": "easy", "scene2": "hard"}
    paths = build_frame_manifest(scan, pack, tmp_path / "stage", splits=splits)
    table = read_frame_manifest(paths.parquet_path)

    rows = list(
        zip(table["scene_id"].to_pylist(), table["scene_type"].to_pylist(),
            table["split"].to_pylist())
    )
    ego_splits = {sid: sp for sid, st, sp in rows if st == SceneType.EGO.value}
    assert ego_splits.get("scene1") == "easy"
    assert ego_splits.get("scene2") == "hard"

    # SOS still gets None — unaffected by the ego change.
    sos_splits = {sid: sp for sid, st, sp in rows if st == SceneType.SINGLE_OBJECT.value}
    assert all(sp is None for sp in sos_splits.values())
