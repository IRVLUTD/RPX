"""Tests 1-2 (spec §13): catalog ID joins, dotted source_catalog_id stays a
string. Uses the real official catalog (hf_hub_download caches locally after
the first call in this environment, so this does not re-download on every
test run)."""
import pytest

import sos_catalog as sc


@pytest.fixture(scope="module")
def catalog():
    return sc.load_catalog()


def test_catalog_has_70_published_objects(catalog):
    # Measured fact (see report): exactly 70 objects are officially
    # published with real HF assets, out of 220 in the wider local catalog.
    assert len(catalog) == 70


def test_dotted_source_catalog_id_is_a_string_not_a_float(catalog):
    obj = catalog["baking_tray.1"]
    assert obj.source_catalog_id == "7.1"
    assert isinstance(obj.source_catalog_id, str)
    # the classic failure mode this test guards against: float("7.1") == 7.1,
    # and a naive numeric cast would silently collapse "7.10"/"7.1" or lose
    # the distinction between category id and instance suffix entirely.
    assert obj.source_catalog_id != 7.1


def test_by_source_catalog_id_join_is_the_real_scene_join_key(catalog):
    by_scid = sc.by_source_catalog_id(catalog)
    assert by_scid["7.1"].object_id == "baking_tray.1"
    assert by_scid["1"].object_id == "air_duster_can"
    # object_id (folder-name form) is NOT the join key -- confirmed in the
    # report against real mask_to_object.json data.
    assert "baking_tray.1" not in by_scid


def test_by_source_catalog_id_detects_ambiguity():
    fake = {
        "a": sc.CatalogObject(1, "a", "5", None, None, "a", "a", "objects/a/", "objects_meta/a/questionnaire.json", {}, {}),
        "b": sc.CatalogObject(2, "b", "5", None, None, "b", "b", "objects/b/", "objects_meta/b/questionnaire.json", {}, {}),
    }
    with pytest.raises(sc.AmbiguousMappingError):
        sc.by_source_catalog_id(fake)


def test_normalize_value_minimal_deterministic():
    assert sc.normalize_value("  Plastic/Metal  ") == "plastic/metal"
    assert sc.normalize_value("Red") == sc.normalize_value("red")
    assert sc.normalize_value("a   b") == "a b"
    # deliberately NOT split on slash or ampersand -- see module docstring
    assert sc.normalize_value("plastic/metal") == "plastic/metal"


def test_join_scene_mask_resolves_published_and_unpublished_alike(catalog):
    by_scid = sc.by_source_catalog_id(catalog)
    mapping = {5: {"name": "air_duster_can", "oid": "1"}, 6: {"name": "mystery_obj", "oid": "999999"}}
    published = sc.join_scene_mask(mapping, 5, by_scid)
    assert published.object_id == "air_duster_can"
    assert published.global_object_id == 1
    unpublished = sc.join_scene_mask(mapping, 6, by_scid)
    assert unpublished.object_id is None
    assert unpublished.global_object_id is None
    assert unpublished.source_catalog_id == "999999"  # local identity still recorded


def test_load_mos_mask_map_source_catalog_id_is_string():
    df = sc.load_mos_mask_map()
    assert df["source_catalog_id"].dtype == object
    assert (df["source_catalog_id"].str.contains(r"\.")).any()


def test_verify_against_mos_manifest_agrees_on_real_row():
    df = sc.load_mos_mask_map()
    assert sc.verify_against_mos_manifest(df, "scene001", 0, 4, "1") is True
    assert sc.verify_against_mos_manifest(df, "scene001", 0, 4, "999") is False
    assert sc.verify_against_mos_manifest(df, "scene001", 0, 99999, "1") is None
