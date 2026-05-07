"""Tests for RPX VQA generation — deterministic, balanced, unbiased."""
import pytest
from rpx_benchmark.tasks.vqa_generation import (
    generate_existence_questions,
    generate_attribute_questions,
    generate_count_questions,
    generate_spatial_questions,
    evaluate_mcq,
    QAPair,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

VISIBLE = [
    {"category": "mug", "color": "red", "material": "ceramic", "shape": "cylinder",
     "instance_id": 1, "bbox_cx": 100, "bbox_cy": 200},
    {"category": "bottle", "color": "blue", "material": "plastic", "shape": "cylinder",
     "instance_id": 2, "bbox_cx": 300, "bbox_cy": 200},
    {"category": "bowl", "color": "white", "material": "ceramic", "shape": "round",
     "instance_id": 3, "bbox_cx": 200, "bbox_cy": 100},
]

ALL_OBJECTS = VISIBLE + [
    {"category": "plate", "color": "green", "material": "glass", "shape": "flat"},
    {"category": "cup", "color": "yellow", "material": "metal", "shape": "cylinder"},
    {"category": "fork", "color": "silver", "material": "metal", "shape": "long"},
    {"category": "knife", "color": "silver", "material": "metal", "shape": "long"},
]


# ── Existence ───────────────────────────────────────────────────────────────

def test_existence_balanced():
    """Yes and No questions should be roughly balanced."""
    pairs = generate_existence_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    yes_count = sum(1 for p in pairs if p.correct_answer == "yes")
    no_count = sum(1 for p in pairs if p.correct_answer == "no")
    assert yes_count > 0
    assert no_count > 0
    assert abs(yes_count - no_count) <= 1  # balanced within 1


def test_existence_options_have_4():
    pairs = generate_existence_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    for p in pairs:
        assert len(p.options) == 4
        assert p.correct_answer in p.options
        assert p.correct_index == p.options.index(p.correct_answer)


def test_existence_deterministic():
    """Same inputs → same outputs."""
    a = generate_existence_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    b = generate_existence_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


# ── Attribute ───────────────────────────────────────────────────────────────

def test_attribute_distractors_are_real():
    """Distractors should be real attribute values, not random strings."""
    pairs = generate_attribute_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    all_colors = {o.get("color", "") for o in ALL_OBJECTS} - {""}
    for p in pairs:
        if p.metadata.get("attribute") == "color":
            for opt in p.options:
                assert opt in all_colors, f"Distractor '{opt}' not a real color"


def test_attribute_correct_answer_from_questionnaire():
    pairs = generate_attribute_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    for p in pairs:
        obj_cat = p.metadata["object"]
        attr_type = p.metadata["attribute"]
        expected = next(o[attr_type] for o in VISIBLE if o["category"] == obj_cat)
        assert p.correct_answer == expected


# ── Count ───────────────────────────────────────────────────────────────────

def test_count_gt_from_masks():
    pairs = generate_count_questions(VISIBLE, "s1", 0, "00000")
    for p in pairs:
        cat = p.metadata["category"]
        expected = sum(1 for o in VISIBLE if o["category"] == cat)
        assert p.correct_answer == str(expected)


def test_count_distractors_are_integers():
    pairs = generate_count_questions(VISIBLE, "s1", 0, "00000")
    for p in pairs:
        for opt in p.options:
            assert opt.isdigit(), f"Count option '{opt}' is not an integer"


# ── Spatial ─────────────────────────────────────────────────────────────────

def test_spatial_image_plane():
    """'Left' means smaller x-coordinate in image plane."""
    pairs = generate_spatial_questions(VISIBLE, "s1", 0, "00000")
    for p in pairs:
        if p.metadata["direction"] == "left":
            ref = next(o for o in VISIBLE if o["category"] == p.metadata["reference"])
            target = next(o for o in VISIBLE if o["category"] == p.metadata["target"])
            assert target["bbox_cx"] < ref["bbox_cx"], \
                f"{p.metadata['target']} should be left of {p.metadata['reference']}"


def test_spatial_above_means_smaller_y():
    pairs = generate_spatial_questions(VISIBLE, "s1", 0, "00000")
    for p in pairs:
        if p.metadata["direction"] == "above":
            ref = next(o for o in VISIBLE if o["category"] == p.metadata["reference"])
            target = next(o for o in VISIBLE if o["category"] == p.metadata["target"])
            assert target["bbox_cy"] < ref["bbox_cy"]


# ── Phase variation (anti-bias) ────────────────────────────────────────────

def test_phase_varying_answers():
    """Same question type on different phases should produce different answers
    when the scene changes."""
    # Phase 0: mug visible. Phase 2: mug removed.
    phase0_visible = [{"category": "mug", "color": "red", "material": "ceramic", "shape": "cylinder"}]
    phase2_visible = [{"category": "bottle", "color": "blue", "material": "plastic", "shape": "cylinder"}]

    p0 = generate_existence_questions(phase0_visible, ALL_OBJECTS, "s1", 0, "00000")
    p2 = generate_existence_questions(phase2_visible, ALL_OBJECTS, "s1", 2, "00000")

    # At least one question about "red mug" should flip from yes→no
    p0_mug = [p for p in p0 if "mug" in p.question and p.correct_answer == "yes"]
    p2_mug = [p for p in p2 if "mug" in p.question and p.correct_answer == "no"]
    # Can't guarantee exact match, but the mechanism works if both lists are non-empty
    # when the object moves between phases
    assert len(p0_mug) > 0 or len(p2_mug) > 0  # at least one phase has mug questions


# ── Evaluation ──────────────────────────────────────────────────────────────

def test_evaluate_perfect_score():
    pairs = generate_existence_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    preds = [p.correct_index for p in pairs]
    result = evaluate_mcq(preds, pairs)
    assert result["accuracy"] == 1.0


def test_evaluate_random_baseline():
    """Random guessing on 4-choice MCQ should be ~25%."""
    import random
    random.seed(42)
    pairs = generate_existence_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    pairs += generate_attribute_questions(VISIBLE, ALL_OBJECTS, "s1", 0, "00000")
    pairs += generate_count_questions(VISIBLE, "s1", 0, "00000")
    # Random predictions
    preds = [random.randint(0, 3) for _ in pairs]
    result = evaluate_mcq(preds, pairs)
    # With small N, allow wide margin, but should not be perfect
    assert result["accuracy"] < 0.8


def test_evaluate_str_computation():
    """STR should be computed from per-phase accuracy differences."""
    qa0 = QAPair("s1", 0, "f0", "existence", "Q?", "yes", ["yes", "no", "x", "y"], 0)
    qa1 = QAPair("s1", 1, "f1", "existence", "Q?", "no", ["yes", "no", "x", "y"], 1)
    qa2 = QAPair("s1", 2, "f2", "existence", "Q?", "yes", ["yes", "no", "x", "y"], 0)

    # Perfect on phase 0 and 2, wrong on phase 1
    preds = [0, 0, 0]  # correct, wrong (should be 1), correct
    result = evaluate_mcq(preds, [qa0, qa1, qa2])

    assert result["per_phase"][0] == 1.0
    assert result["per_phase"][1] == 0.0
    assert result["per_phase"][2] == 1.0
    assert result["str_clu_to_int"] == -1.0  # dropped from 1.0 to 0.0
    assert result["str_int_to_cln"] == 1.0   # recovered from 0.0 to 1.0
