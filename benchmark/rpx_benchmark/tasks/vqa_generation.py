"""RPX VQA ground-truth generation.

Generates deterministic Q/A pairs from annotations. No human labeling needed.
No LLM. No fuzzy matching. MCQ with 4 options, random baseline = 25%.

Design principles (Karpathy/Girshick/Fox/He):
  - Every question has ONE correct answer derivable from masks + questionnaire.
  - Every question maps to a robot operation.
  - Phase variation is the anti-bias mechanism — same question, different answer.
  - MCQ distractors come from real attributes of real objects in the dataset.
  - Option order is shuffled per question with a fixed seed for reproducibility.

Question types:
  Q1. Existence   — "Is there a red mug?"        → from visibility mask
  Q2. Attribute   — "What color is the mug?"      → from questionnaire
  Q3. Count       — "How many cups are visible?"   → from mask instance count
  Q4. Spatial     — "What is left of the bowl?"    → from bbox centroids (image plane)

Evaluation: MCQ accuracy. Per-phase breakdown → feeds STR.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── Question templates ──────────────────────────────────────────────────────
# Each template maps to a robot operation. Multiple phrasings per type
# to reduce template memorization, but all are simple imperative/interrogative.

EXISTENCE_TEMPLATES = [
    "Is there a {attr} {category} in this scene?",
    "Can you see a {attr} {category}?",
    "Is a {attr} {category} present?",
]

ATTRIBUTE_TEMPLATES = {
    "color": [
        "What color is the {category}?",
        "What is the color of the {category}?",
    ],
    "material": [
        "What material is the {category} made of?",
        "What is the material of the {category}?",
    ],
    "shape": [
        "What shape is the {category}?",
        "What is the shape of the {category}?",
    ],
}

COUNT_TEMPLATES = [
    "How many {category} objects are visible?",
    "How many {category} can you see?",
    "Count the {category} in this scene.",
]

SPATIAL_TEMPLATES = [
    "What object is to the {direction} of the {object}?",
    "What is {direction} of the {object}?",
]

SPATIAL_DIRECTIONS = ["left", "right", "above", "below"]


@dataclass
class QAPair:
    """One question-answer pair with MCQ options."""

    scene_id: str
    phase: int
    frame_id: str
    question_type: str  # existence | attribute | count | spatial
    question: str
    correct_answer: str
    options: List[str]  # 4 options, correct answer included, shuffled
    correct_index: int  # index of correct answer in options
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "phase": self.phase,
            "frame_id": self.frame_id,
            "question_type": self.question_type,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "options": self.options,
            "correct_index": self.correct_index,
            "metadata": self.metadata,
        }


def _deterministic_seed(scene_id: str, phase: int, frame_id: str, q: str) -> int:
    """Reproducible seed per question for option shuffling."""
    h = hashlib.md5(f"{scene_id}/{phase}/{frame_id}/{q}".encode()).hexdigest()
    return int(h[:8], 16)


def _shuffle_options(correct: str, distractors: List[str], seed: int) -> Tuple[List[str], int]:
    """Shuffle 4 options with deterministic seed. Return (options, correct_index)."""
    opts = [correct] + distractors[:3]
    rng = random.Random(seed)
    rng.shuffle(opts)
    return opts, opts.index(correct)


def _pick_template(templates: list, seed: int) -> str:
    """Pick a template deterministically."""
    return templates[seed % len(templates)]


# ── Generators ──────────────────────────────────────────────────────────────


def generate_existence_questions(
    visible_objects: List[Dict],  # [{"category": "mug", "color": "red", ...}, ...]
    all_objects_in_dataset: List[Dict],  # for generating "no" distractors
    scene_id: str,
    phase: int,
    frame_id: str,
) -> List[QAPair]:
    """Generate balanced existence questions: 50% yes, 50% no.

    'Yes' questions ask about objects visible in this frame.
    'No' questions ask about objects NOT visible (from other scenes in the dataset).
    This prevents a model from scoring well by always answering 'yes'.
    """
    pairs = []
    visible_set = {(o["category"], o.get("color", "")) for o in visible_objects}

    # Yes questions — pick up to 3 visible objects
    for obj in visible_objects[:3]:
        attr = obj.get("color", obj.get("material", ""))
        if not attr:
            continue
        seed = _deterministic_seed(scene_id, phase, frame_id, f"exist_yes_{obj['category']}")
        q = _pick_template(EXISTENCE_TEMPLATES, seed).format(attr=attr, category=obj["category"])
        opts, idx = _shuffle_options("yes", ["no", "not sure", "partially visible"], seed)
        pairs.append(
            QAPair(
                scene_id=scene_id,
                phase=phase,
                frame_id=frame_id,
                question_type="existence",
                question=q,
                correct_answer="yes",
                options=opts,
                correct_index=idx,
                metadata={"object": obj["category"], "expected": True},
            )
        )

    # No questions — ask about objects absent from this frame
    absent = [
        o for o in all_objects_in_dataset if (o["category"], o.get("color", "")) not in visible_set
    ]
    for obj in absent[: len(pairs)]:  # match count for 50/50 balance
        attr = obj.get("color", obj.get("material", ""))
        if not attr:
            continue
        seed = _deterministic_seed(scene_id, phase, frame_id, f"exist_no_{obj['category']}")
        q = _pick_template(EXISTENCE_TEMPLATES, seed).format(attr=attr, category=obj["category"])
        opts, idx = _shuffle_options("no", ["yes", "not sure", "partially visible"], seed)
        pairs.append(
            QAPair(
                scene_id=scene_id,
                phase=phase,
                frame_id=frame_id,
                question_type="existence",
                question=q,
                correct_answer="no",
                options=opts,
                correct_index=idx,
                metadata={"object": obj["category"], "expected": False},
            )
        )

    return pairs


def generate_attribute_questions(
    visible_objects: List[Dict],
    all_objects_in_dataset: List[Dict],  # for distractors
    scene_id: str,
    phase: int,
    frame_id: str,
) -> List[QAPair]:
    """Generate attribute questions with distractors from real objects."""
    pairs = []
    # Collect all attribute values by type for distractor pool
    attr_pool: Dict[str, List[str]] = {"color": [], "material": [], "shape": []}
    for o in all_objects_in_dataset:
        for attr_type in attr_pool:
            v = o.get(attr_type, "")
            if v and v not in attr_pool[attr_type]:
                attr_pool[attr_type].append(v)

    for obj in visible_objects[:5]:
        for attr_type, templates in ATTRIBUTE_TEMPLATES.items():
            gt_val = obj.get(attr_type, "")
            if not gt_val:
                continue
            # Get 3 distractors from real attribute values
            pool = [v for v in attr_pool.get(attr_type, []) if v != gt_val]
            if len(pool) < 3:
                continue

            seed = _deterministic_seed(
                scene_id, phase, frame_id, f"attr_{attr_type}_{obj['category']}"
            )
            rng = random.Random(seed)
            distractors = rng.sample(pool, 3)
            q = _pick_template(templates, seed).format(category=obj["category"])
            opts, idx = _shuffle_options(gt_val, distractors, seed)
            pairs.append(
                QAPair(
                    scene_id=scene_id,
                    phase=phase,
                    frame_id=frame_id,
                    question_type="attribute",
                    question=q,
                    correct_answer=gt_val,
                    options=opts,
                    correct_index=idx,
                    metadata={"object": obj["category"], "attribute": attr_type},
                )
            )

    return pairs


def generate_count_questions(
    visible_objects: List[Dict],
    scene_id: str,
    phase: int,
    frame_id: str,
) -> List[QAPair]:
    """Generate counting questions. GT from mask instance count."""
    pairs = []
    # Count by category
    counts: Dict[str, int] = {}
    for obj in visible_objects:
        counts[obj["category"]] = counts.get(obj["category"], 0) + 1

    # Ask about categories with ≥1 instance
    for cat, count in counts.items():
        seed = _deterministic_seed(scene_id, phase, frame_id, f"count_{cat}")
        q = _pick_template(COUNT_TEMPLATES, seed).format(category=cat)
        gt = str(count)
        # Distractors: nearby integers, always including 0
        distractor_pool = [str(i) for i in range(0, max(count + 4, 6)) if str(i) != gt]
        rng = random.Random(seed)
        distractors = rng.sample(distractor_pool, min(3, len(distractor_pool)))
        if len(distractors) < 3:
            continue
        opts, idx = _shuffle_options(gt, distractors, seed)
        pairs.append(
            QAPair(
                scene_id=scene_id,
                phase=phase,
                frame_id=frame_id,
                question_type="count",
                question=q,
                correct_answer=gt,
                options=opts,
                correct_index=idx,
                metadata={"category": cat, "count": count},
            )
        )

    return pairs


def generate_spatial_questions(
    visible_objects: List[Dict],  # must include "bbox_cx", "bbox_cy"
    scene_id: str,
    phase: int,
    frame_id: str,
) -> List[QAPair]:
    """Generate spatial questions. Relations defined in image plane.

    "Left" = smaller x-coordinate. "Above" = smaller y-coordinate.
    This matches how VLMs process images and avoids viewpoint ambiguity.
    """
    pairs = []
    if len(visible_objects) < 2:
        return pairs

    for ref_obj in visible_objects[:3]:
        ref_cx = ref_obj.get("bbox_cx")
        ref_cy = ref_obj.get("bbox_cy")
        if ref_cx is None or ref_cy is None:
            continue

        for direction in SPATIAL_DIRECTIONS:
            # Find the nearest object in that direction
            best = None
            best_dist = float("inf")
            for other in visible_objects:
                if other["category"] == ref_obj["category"] and other.get(
                    "instance_id"
                ) == ref_obj.get("instance_id"):
                    continue
                ox, oy = other.get("bbox_cx"), other.get("bbox_cy")
                if ox is None or oy is None:
                    continue

                # Check direction
                if direction == "left" and ox >= ref_cx:
                    continue
                if direction == "right" and ox <= ref_cx:
                    continue
                if direction == "above" and oy >= ref_cy:
                    continue
                if direction == "below" and oy <= ref_cy:
                    continue

                dist = abs(ox - ref_cx) + abs(oy - ref_cy)
                if dist < best_dist:
                    best_dist = dist
                    best = other

            if best is None:
                continue

            seed = _deterministic_seed(
                scene_id, phase, frame_id, f"spatial_{direction}_{ref_obj['category']}"
            )
            q = _pick_template(SPATIAL_TEMPLATES, seed).format(
                direction=direction, object=ref_obj["category"]
            )
            gt = best["category"]

            # Distractors: other visible objects
            others = [
                o["category"]
                for o in visible_objects
                if o["category"] != gt and o["category"] != ref_obj["category"]
            ]
            others = list(set(others))
            if len(others) < 2:
                others += ["nothing", "unknown"]
            rng = random.Random(seed)
            distractors = rng.sample(others, min(3, len(others)))
            while len(distractors) < 3:
                distractors.append("nothing")
            opts, idx = _shuffle_options(gt, distractors[:3], seed)

            pairs.append(
                QAPair(
                    scene_id=scene_id,
                    phase=phase,
                    frame_id=frame_id,
                    question_type="spatial",
                    question=q,
                    correct_answer=gt,
                    options=opts,
                    correct_index=idx,
                    metadata={
                        "reference": ref_obj["category"],
                        "direction": direction,
                        "target": best["category"],
                    },
                )
            )
            break  # one spatial question per reference object per direction

    return pairs


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_mcq(predictions: List[int], ground_truth: List[QAPair]) -> Dict:
    """Evaluate MCQ predictions. Returns accuracy overall and per type/phase.

    Args:
        predictions: list of predicted option indices (0-3)
        ground_truth: list of QAPair objects

    Returns:
        dict with overall accuracy + per-type + per-phase breakdowns
    """
    assert len(predictions) == len(ground_truth)

    correct = 0
    by_type: Dict[str, List[bool]] = {}
    by_phase: Dict[int, List[bool]] = {}

    for pred, qa in zip(predictions, ground_truth, strict=False):
        hit = pred == qa.correct_index
        correct += hit

        by_type.setdefault(qa.question_type, []).append(hit)
        by_phase.setdefault(qa.phase, []).append(hit)

    n = len(predictions)
    result = {
        "accuracy": correct / n if n > 0 else 0.0,
        "n_questions": n,
        "per_type": {t: sum(v) / len(v) for t, v in by_type.items()},
        "per_phase": {p: sum(v) / len(v) for p, v in by_phase.items()},
    }

    # STR for VQA = phase accuracy difference
    phases = sorted(by_phase.keys())
    if len(phases) >= 2:
        phase_acc = {p: sum(v) / len(v) for p, v in by_phase.items()}
        if 0 in phase_acc and 1 in phase_acc:
            result["str_clu_to_int"] = phase_acc[1] - phase_acc[0]
        if 1 in phase_acc and 2 in phase_acc:
            result["str_int_to_cln"] = phase_acc[2] - phase_acc[1]

    return result
