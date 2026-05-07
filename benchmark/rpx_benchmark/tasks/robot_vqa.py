"""RPX Robot-Grounded VQA — tasks a robot actually needs.

Every question maps to a real robot operation. Every answer feeds an action.
No generic VQA. No "what is the weather like?"

Six task types, each grounded in a specific robot pipeline stage:

T1. IN-CONTEXT RECOGNITION (the killer feature)
    Robot has a reference photo of target object (from single-object scene).
    VLM must identify whether that object appears in the cluttered scene.
    Uses: paired single-object / multi-object scenes.
    This is exactly what DE-ViT, NIDS-Net, and real robot "object registration" do.

T2. ATTRIBUTE GROUNDING
    Robot receives "pick the red mug" — must distinguish red mug from blue mug.
    VLM must ground natural-language attributes to specific objects.
    Uses: FewSOL questionnaire attributes.

T3. SPATIAL GROUNDING
    Robot receives "pick the object left of the bowl."
    VLM must resolve spatial relations in the image plane.
    Uses: 360° walkaround (relations change per viewpoint).

T4. STATE VERIFICATION
    Robot asks "did I succeed? Is the target still on the table?"
    VLM compares two frames: before and after manipulation.
    Uses: clutter → clean phase transition (unique to RPX).

T5. COUNTING
    Robot asks "how many objects remain to pack?"
    VLM counts visible instances of a category.
    Uses: instance masks for GT.

T6. INTERACTION UNDERSTANDING
    Learning-from-human pipeline asks "what object is the human touching?"
    VLM must identify the object being manipulated during interaction phase.
    Uses: interaction phase with human hands in frame.

Format: 4-choice MCQ for T1-T5. Open-ended for T6 (scored by exact category match).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RobotQA:
    """One robot-grounded QA pair."""
    scene_id: str
    phase: int
    frame_id: str
    task_type: str  # T1-T6
    robot_operation: str  # what robot pipeline stage this tests
    question: str
    correct_answer: str
    options: List[str]  # 4 options for MCQ; empty for open-ended
    correct_index: int  # -1 for open-ended
    reference_image: Optional[str] = None  # path to single-object reference (T1)
    comparison_frame: Optional[str] = None  # path to before/after frame (T4)
    metadata: Dict = field(default_factory=dict)


def _seed(scene_id: str, phase: int, frame_id: str, tag: str) -> int:
    h = hashlib.md5(f"{scene_id}/{phase}/{frame_id}/{tag}".encode()).hexdigest()
    return int(h[:8], 16)


def _shuffle(correct: str, distractors: List[str], seed: int) -> Tuple[List[str], int]:
    opts = [correct] + distractors[:3]
    random.Random(seed).shuffle(opts)
    return opts, opts.index(correct)


# ── T1: In-Context Recognition ─────────────────────────────────────────────

def generate_incontext_recognition(
    visible_objects: List[Dict],
    single_object_refs: Dict[str, str],  # {category: path_to_reference_image}
    absent_categories: List[str],  # categories NOT in this frame
    scene_id: str, phase: int, frame_id: str,
) -> List[RobotQA]:
    """T1: 'Here is a photo of the target. Is it in this scene?'

    This is the object registration task every real robot does.
    The single-object scene image IS the reference photo.
    """
    pairs = []
    visible_cats = {o["category"] for o in visible_objects}

    # Positive: objects that ARE visible and have a reference image
    for cat in list(visible_cats)[:3]:
        if cat not in single_object_refs:
            continue
        s = _seed(scene_id, phase, frame_id, f"t1_yes_{cat}")
        q = "The reference image shows an object. Is this exact object present in the current scene?"
        opts, idx = _shuffle("yes, it is visible", [
            "no, it is not in this scene",
            "a similar but different object is present",
            "cannot determine from this viewpoint",
        ], s)
        pairs.append(RobotQA(
            scene_id=scene_id, phase=phase, frame_id=frame_id,
            task_type="T1_incontext", robot_operation="object_registration",
            question=q, correct_answer="yes, it is visible",
            options=opts, correct_index=idx,
            reference_image=single_object_refs[cat],
            metadata={"category": cat, "present": True},
        ))

    # Negative: objects NOT in this frame but that have reference images
    neg_count = 0
    for cat in absent_categories:
        if cat not in single_object_refs or neg_count >= len(pairs):
            continue
        s = _seed(scene_id, phase, frame_id, f"t1_no_{cat}")
        q = "The reference image shows an object. Is this exact object present in the current scene?"
        opts, idx = _shuffle("no, it is not in this scene", [
            "yes, it is visible",
            "a similar but different object is present",
            "partially occluded but present",
        ], s)
        pairs.append(RobotQA(
            scene_id=scene_id, phase=phase, frame_id=frame_id,
            task_type="T1_incontext", robot_operation="object_registration",
            question=q, correct_answer="no, it is not in this scene",
            options=opts, correct_index=idx,
            reference_image=single_object_refs[cat],
            metadata={"category": cat, "present": False},
        ))
        neg_count += 1

    return pairs


# ── T2: Attribute Grounding ────────────────────────────────────────────────

def generate_attribute_grounding(
    visible_objects: List[Dict],
    all_objects: List[Dict],
    scene_id: str, phase: int, frame_id: str,
) -> List[RobotQA]:
    """T2: 'Pick the red mug' — which object matches this description?

    Tests whether the VLM can distinguish objects by attributes.
    Only generates questions when multiple objects of the same category
    are visible (otherwise there's no disambiguation needed).
    """
    pairs = []
    # Find categories with multiple visible instances
    from collections import Counter
    cat_counts = Counter(o["category"] for o in visible_objects)
    ambiguous_cats = [c for c, n in cat_counts.items() if n >= 2]

    for cat in ambiguous_cats:
        instances = [o for o in visible_objects if o["category"] == cat]
        for obj in instances[:2]:
            color = obj.get("color", "")
            if not color:
                continue
            s = _seed(scene_id, phase, frame_id, f"t2_{cat}_{color}")
            q = f"A robot is instructed to pick the {color} {cat}. Which object should it grasp?"
            gt = f"the {color} {cat}"
            other_colors = [o.get("color", "?") for o in instances if o.get("color") != color]
            distractors = [f"the {c} {cat}" for c in other_colors[:2]]
            distractors.append(f"any {cat}")
            while len(distractors) < 3:
                distractors.append("none of the visible objects")
            opts, idx = _shuffle(gt, distractors[:3], s)
            pairs.append(RobotQA(
                scene_id=scene_id, phase=phase, frame_id=frame_id,
                task_type="T2_attribute", robot_operation="language_grounding",
                question=q, correct_answer=gt,
                options=opts, correct_index=idx,
                metadata={"category": cat, "target_color": color},
            ))

    return pairs


# ── T3: Spatial Grounding ──────────────────────────────────────────────────

def generate_spatial_grounding(
    visible_objects: List[Dict],  # must include bbox_cx, bbox_cy
    scene_id: str, phase: int, frame_id: str,
) -> List[RobotQA]:
    """T3: 'Pick the object to the left of the bowl.'

    Relations in image plane. Left = smaller x. Above = smaller y.
    """
    pairs = []
    if len(visible_objects) < 2:
        return pairs

    directions = {"left": (-1, 0), "right": (1, 0), "above": (0, -1), "below": (0, 1)}

    for ref in visible_objects[:3]:
        cx, cy = ref.get("bbox_cx"), ref.get("bbox_cy")
        if cx is None:
            continue

        for direction, (dx_sign, dy_sign) in directions.items():
            # Find nearest object in that direction
            best, best_dist = None, float("inf")
            for other in visible_objects:
                if other is ref:
                    continue
                ox, oy = other.get("bbox_cx"), other.get("bbox_cy")
                if ox is None:
                    continue
                # Check direction
                if dx_sign < 0 and ox >= cx:
                    continue
                if dx_sign > 0 and ox <= cx:
                    continue
                if dy_sign < 0 and oy >= cy:
                    continue
                if dy_sign > 0 and oy <= cy:
                    continue
                d = abs(ox - cx) + abs(oy - cy)
                if d < best_dist:
                    best, best_dist = other, d

            if best is None:
                continue

            s = _seed(scene_id, phase, frame_id, f"t3_{direction}_{ref['category']}")
            q = f"A robot must pick the object {direction} of the {ref['category']}. What should it pick?"
            gt = best["category"]
            others = list({o["category"] for o in visible_objects
                          if o["category"] != gt and o["category"] != ref["category"]})
            others += ["nothing — no object in that direction"]
            rng = random.Random(s)
            distractors = rng.sample(others, min(3, len(others)))
            while len(distractors) < 3:
                distractors.append("cannot determine")
            opts, idx = _shuffle(gt, distractors[:3], s)
            pairs.append(RobotQA(
                scene_id=scene_id, phase=phase, frame_id=frame_id,
                task_type="T3_spatial", robot_operation="spatial_command",
                question=q, correct_answer=gt,
                options=opts, correct_index=idx,
                metadata={"reference": ref["category"], "direction": direction},
            ))
            break  # one per ref per direction

    return pairs


# ── T4: State Verification ─────────────────────────────────────────────────

def generate_state_verification(
    clutter_objects: List[Dict],  # objects visible in clutter phase
    clean_objects: List[Dict],    # objects visible in clean phase
    scene_id: str, frame_id: str,
) -> List[RobotQA]:
    """T4: 'Did the manipulation succeed? What changed?'

    Compares clutter and clean phases of the same scene.
    This is what a robot does after every pick-and-place: verify.
    UNIQUE TO RPX — no other benchmark can generate this.
    """
    pairs = []
    clutter_cats = {o["category"] for o in clutter_objects}
    clean_cats = {o["category"] for o in clean_objects}

    removed = clutter_cats - clean_cats
    added = clean_cats - clutter_cats
    remained = clutter_cats & clean_cats

    # Question: what was removed?
    if removed:
        removed_obj = list(removed)[0]
        s = _seed(scene_id, 0, frame_id, f"t4_removed_{removed_obj}")
        q = "Comparing the before and after images: which object was removed from the scene?"
        distractors = list(remained)[:3]
        while len(distractors) < 3:
            distractors.append("nothing was removed")
        opts, idx = _shuffle(removed_obj, distractors[:3], s)
        pairs.append(RobotQA(
            scene_id=scene_id, phase=0, frame_id=frame_id,
            task_type="T4_state", robot_operation="task_verification",
            question=q, correct_answer=removed_obj,
            options=opts, correct_index=idx,
            metadata={"change_type": "removed", "object": removed_obj},
        ))

    # Question: what was added?
    if added:
        added_obj = list(added)[0]
        s = _seed(scene_id, 2, frame_id, f"t4_added_{added_obj}")
        q = "Comparing the before and after images: which object was added to the scene?"
        distractors = list(remained)[:3]
        while len(distractors) < 3:
            distractors.append("nothing was added")
        opts, idx = _shuffle(added_obj, distractors[:3], s)
        pairs.append(RobotQA(
            scene_id=scene_id, phase=2, frame_id=frame_id,
            task_type="T4_state", robot_operation="task_verification",
            question=q, correct_answer=added_obj,
            options=opts, correct_index=idx,
            metadata={"change_type": "added", "object": added_obj},
        ))

    # Question: did the count change?
    if len(clutter_cats) != len(clean_cats):
        s = _seed(scene_id, 0, frame_id, "t4_count")
        delta = len(clean_cats) - len(clutter_cats)
        q = "After manipulation, did the number of visible objects change?"
        if delta < 0:
            gt = f"decreased by {abs(delta)}"
        elif delta > 0:
            gt = f"increased by {delta}"
        else:
            gt = "stayed the same"
        opts, idx = _shuffle(gt, [
            "stayed the same" if delta != 0 else "decreased by 1",
            f"increased by {abs(delta) + 1}" if delta <= 0 else f"decreased by {delta + 1}",
            "cannot determine",
        ], s)
        pairs.append(RobotQA(
            scene_id=scene_id, phase=0, frame_id=frame_id,
            task_type="T4_state", robot_operation="task_verification",
            question=q, correct_answer=gt,
            options=opts, correct_index=idx,
            metadata={"change_type": "count", "delta": delta},
        ))

    return pairs


# ── T5: Counting ───────────────────────────────────────────────────────────

def generate_counting(
    visible_objects: List[Dict],
    scene_id: str, phase: int, frame_id: str,
) -> List[RobotQA]:
    """T5: 'How many objects remain to pack?'"""
    pairs = []
    from collections import Counter
    counts = Counter(o["category"] for o in visible_objects)

    for cat, count in list(counts.items())[:3]:
        s = _seed(scene_id, phase, frame_id, f"t5_{cat}")
        q = f"A robot is packing all {cat} objects. How many does it need to pick?"
        gt = str(count)
        pool = [str(i) for i in range(max(count + 3, 5)) if str(i) != gt]
        distractors = random.Random(s).sample(pool, min(3, len(pool)))
        while len(distractors) < 3:
            distractors.append("0")
        opts, idx = _shuffle(gt, distractors[:3], s)
        pairs.append(RobotQA(
            scene_id=scene_id, phase=phase, frame_id=frame_id,
            task_type="T5_counting", robot_operation="task_progress",
            question=q, correct_answer=gt,
            options=opts, correct_index=idx,
            metadata={"category": cat, "count": count},
        ))

    return pairs


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_robot_vqa(
    predictions: List[int],
    ground_truth: List[RobotQA],
) -> Dict:
    """MCQ accuracy with breakdowns by task type, phase, and robot operation."""
    assert len(predictions) == len(ground_truth)

    by_type: Dict[str, List[bool]] = {}
    by_phase: Dict[int, List[bool]] = {}
    by_operation: Dict[str, List[bool]] = {}

    for pred, qa in zip(predictions, ground_truth):
        hit = (pred == qa.correct_index)
        by_type.setdefault(qa.task_type, []).append(hit)
        by_phase.setdefault(qa.phase, []).append(hit)
        by_operation.setdefault(qa.robot_operation, []).append(hit)

    n = len(predictions)
    correct = sum(1 for p, q in zip(predictions, ground_truth) if p == q.correct_index)

    result = {
        "accuracy": correct / n if n else 0.0,
        "n_questions": n,
        "per_task_type": {t: sum(v) / len(v) for t, v in by_type.items()},
        "per_phase": {p: sum(v) / len(v) for p, v in by_phase.items()},
        "per_robot_operation": {o: sum(v) / len(v) for o, v in by_operation.items()},
    }

    # STR
    pa = {p: sum(v) / len(v) for p, v in by_phase.items()}
    if 0 in pa and 1 in pa:
        result["str_clu_to_int"] = pa[1] - pa[0]
    if 1 in pa and 2 in pa:
        result["str_int_to_cln"] = pa[2] - pa[1]

    return result
