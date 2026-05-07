"""RPX Robot VQA v2 — three context modes, all GT from recorded data.

Every question can be asked in three ways:
  - image_only:    [reference photo] + [scene frame]
  - image_text:    [reference photo] + text description + [scene frame]
  - text_only:     text description only + [scene frame]

The ground truth is IDENTICAL across modes — it comes from masks +
questionnaire + bounding boxes, never from any VLM or LLM.

The model being evaluated receives the prompt in one mode and must
produce the answer. Comparing accuracy across modes reveals which
prompting strategy is most robust for robot deployment.

GT sources:
  - Instance masks (SAM2 + human refinement) → visibility, bbox, count
  - FewSOL questionnaire (MTurk) → category, color, material, shape
  - Bounding box centroids (from masks) → spatial relations
  - Phase diff (clutter vs clean) → state changes
  - Single-object scenes → reference images
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RobotVQA:
    """One question with all three context modes.

    The model is evaluated in ONE mode at a time. The GT is the same
    regardless of mode.
    """
    scene_id: str
    phase: int
    frame_id: str
    task: str                    # recognition | attribute | spatial | state | counting

    # The question in each context mode
    image_only_prompt: str       # what to show/ask with reference image only
    image_text_prompt: str       # what to show/ask with reference image + text
    text_only_prompt: str        # what to ask with text only

    reference_image: Optional[str]  # path to single-object reference (None for text-only tasks)
    scene_frame: str              # path to the scene frame being evaluated

    correct_answer: str
    options: List[str]           # 4-choice MCQ
    correct_index: int

    robot_operation: str         # what pipeline stage this tests
    metadata: Dict = field(default_factory=dict)


def _seed(s: str, p: int, f: str, tag: str) -> int:
    return int(hashlib.md5(f"{s}/{p}/{f}/{tag}".encode()).hexdigest()[:8], 16)


def _shuffle(correct: str, distractors: List[str], seed: int) -> Tuple[List[str], int]:
    opts = [correct] + distractors[:3]
    random.Random(seed).shuffle(opts)
    return opts, opts.index(correct)


# ─── Recognition: "Is this object in the scene?" ──────────────────────────

def gen_recognition(
    visible: List[Dict],           # objects visible in this frame
    ref_images: Dict[str, str],    # {category: path} from single-object scenes
    absent_cats: List[str],        # categories NOT visible
    questionnaire: Dict[str, Dict],  # {category: {color, material, shape}}
    scene_id: str, phase: int, frame_id: str, scene_frame: str,
) -> List[RobotVQA]:
    """Object recognition in three modes.

    Same GT (yes/no from mask visibility), different prompts.
    """
    pairs = []
    vis_cats = {o["category"] for o in visible}

    # Positive examples
    for cat in sorted(vis_cats)[:3]:
        if cat not in ref_images:
            continue
        attrs = questionnaire.get(cat, {})
        color = attrs.get("color", "")
        material = attrs.get("material", "")
        desc = f"{color} {material} {cat}".strip()

        s = _seed(scene_id, phase, frame_id, f"rec_yes_{cat}")
        opts, idx = _shuffle("yes", ["no", "similar but different object", "cannot determine"], s)

        pairs.append(RobotVQA(
            scene_id=scene_id, phase=phase, frame_id=frame_id,
            task="recognition",
            image_only_prompt="Is the object shown in the reference image present in this scene?",
            image_text_prompt=f"Is the {desc} shown in the reference image present in this scene?",
            text_only_prompt=f"Is there a {desc} in this scene?",
            reference_image=ref_images[cat],
            scene_frame=scene_frame,
            correct_answer="yes", options=opts, correct_index=idx,
            robot_operation="object_registration",
            metadata={"category": cat, "present": True},
        ))

    # Negative examples (balanced 50/50)
    n_neg = 0
    for cat in absent_cats:
        if cat not in ref_images or n_neg >= len([p for p in pairs if p.metadata.get("present")]):
            continue
        attrs = questionnaire.get(cat, {})
        color = attrs.get("color", "")
        material = attrs.get("material", "")
        desc = f"{color} {material} {cat}".strip()

        s = _seed(scene_id, phase, frame_id, f"rec_no_{cat}")
        opts, idx = _shuffle("no", ["yes", "partially visible", "cannot determine"], s)

        pairs.append(RobotVQA(
            scene_id=scene_id, phase=phase, frame_id=frame_id,
            task="recognition",
            image_only_prompt="Is the object shown in the reference image present in this scene?",
            image_text_prompt=f"Is the {desc} shown in the reference image present in this scene?",
            text_only_prompt=f"Is there a {desc} in this scene?",
            reference_image=ref_images[cat],
            scene_frame=scene_frame,
            correct_answer="no", options=opts, correct_index=idx,
            robot_operation="object_registration",
            metadata={"category": cat, "present": False},
        ))
        n_neg += 1

    return pairs


# ─── Attribute grounding: "Pick the RED mug" ──────────────────────────────

def gen_attribute(
    visible: List[Dict],
    ref_images: Dict[str, str],
    questionnaire: Dict[str, Dict],
    all_objects: List[Dict],
    scene_id: str, phase: int, frame_id: str, scene_frame: str,
) -> List[RobotVQA]:
    """Attribute grounding — only fires when disambiguation is needed."""
    pairs = []
    from collections import Counter
    cat_counts = Counter(o["category"] for o in visible)

    for cat, count in cat_counts.items():
        if count < 2:
            continue  # no disambiguation needed
        instances = [o for o in visible if o["category"] == cat]

        for obj in instances[:2]:
            color = obj.get("color", "")
            if not color:
                continue
            other_colors = [o.get("color", "?") for o in instances if o.get("color") != color]

            s = _seed(scene_id, phase, frame_id, f"attr_{cat}_{color}")
            gt = f"the {color} {cat}"
            distractors = [f"the {c} {cat}" for c in other_colors[:2]] + [f"any {cat}"]
            while len(distractors) < 3:
                distractors.append("none of these")
            opts, idx = _shuffle(gt, distractors[:3], s)

            ref_path = ref_images.get(cat)
            pairs.append(RobotVQA(
                scene_id=scene_id, phase=phase, frame_id=frame_id,
                task="attribute",
                image_only_prompt=f"Which of the visible {cat} objects matches the reference image?",
                image_text_prompt=f"Which object is the {color} {cat}?",
                text_only_prompt=f"A robot must pick the {color} {cat}. Which object should it grasp?",
                reference_image=ref_path,
                scene_frame=scene_frame,
                correct_answer=gt, options=opts, correct_index=idx,
                robot_operation="language_grounding",
                metadata={"category": cat, "target_color": color},
            ))

    return pairs


# ─── Spatial grounding: "Pick the object LEFT of the bowl" ────────────────

def gen_spatial(
    visible: List[Dict],  # must have bbox_cx, bbox_cy
    scene_id: str, phase: int, frame_id: str, scene_frame: str,
) -> List[RobotVQA]:
    """Spatial grounding — image-plane relations only. Text-only task."""
    pairs = []
    if len(visible) < 2:
        return pairs

    for ref in visible[:3]:
        cx, cy = ref.get("bbox_cx"), ref.get("bbox_cy")
        if cx is None:
            continue

        for direction, check in [
            ("left", lambda o: o.get("bbox_cx", cx) < cx),
            ("right", lambda o: o.get("bbox_cx", cx) > cx),
            ("above", lambda o: o.get("bbox_cy", cy) < cy),
            ("below", lambda o: o.get("bbox_cy", cy) > cy),
        ]:
            candidates = [o for o in visible if o is not ref and check(o) and o.get("bbox_cx") is not None]
            if not candidates:
                continue
            # Nearest
            target = min(candidates, key=lambda o: abs(o["bbox_cx"] - cx) + abs(o["bbox_cy"] - cy))
            gt = target["category"]

            s = _seed(scene_id, phase, frame_id, f"sp_{direction}_{ref['category']}")
            others = list({o["category"] for o in visible if o["category"] not in (gt, ref["category"])})
            others += ["nothing"]
            distractors = random.Random(s).sample(others, min(3, len(others)))
            while len(distractors) < 3:
                distractors.append("cannot determine")
            opts, idx = _shuffle(gt, distractors[:3], s)

            pairs.append(RobotVQA(
                scene_id=scene_id, phase=phase, frame_id=frame_id,
                task="spatial",
                image_only_prompt=f"What object is {direction} of the {ref['category']}?",
                image_text_prompt=f"What object is {direction} of the {ref['category']}?",
                text_only_prompt=f"A robot must pick the object {direction} of the {ref['category']}. What should it pick?",
                reference_image=None,
                scene_frame=scene_frame,
                correct_answer=gt, options=opts, correct_index=idx,
                robot_operation="spatial_command",
                metadata={"reference": ref["category"], "direction": direction},
            ))
            break  # one per direction per ref

    return pairs


# ─── State verification: "Did the pick succeed?" ─────────────────────────

def gen_state_verification(
    clutter_objects: List[Dict],
    clean_objects: List[Dict],
    scene_id: str, frame_id: str,
    clutter_frame: str, clean_frame: str,
) -> List[RobotVQA]:
    """State verification — compare two phases. Uses TWO frames as input."""
    pairs = []
    clu_cats = {o["category"] for o in clutter_objects}
    cln_cats = {o["category"] for o in clean_objects}
    removed = clu_cats - cln_cats
    remained = clu_cats & cln_cats

    for obj_cat in list(removed)[:2]:
        s = _seed(scene_id, 0, frame_id, f"state_{obj_cat}")
        distractors = list(remained)[:3]
        while len(distractors) < 3:
            distractors.append("nothing changed")
        opts, idx = _shuffle(obj_cat, distractors[:3], s)

        pairs.append(RobotVQA(
            scene_id=scene_id, phase=0, frame_id=frame_id,
            task="state",
            image_only_prompt="Compare these two images. Which object was removed?",
            image_text_prompt="Compare the before and after images. Which object was removed from the scene?",
            text_only_prompt="After the robot's manipulation, which object is no longer on the table?",
            reference_image=clutter_frame,  # "before" image
            scene_frame=clean_frame,        # "after" image
            correct_answer=obj_cat, options=opts, correct_index=idx,
            robot_operation="task_verification",
            metadata={"change_type": "removed", "object": obj_cat},
        ))

    return pairs


# ─── Counting: "How many left to pack?" ──────────────────────────────────

def gen_counting(
    visible: List[Dict],
    scene_id: str, phase: int, frame_id: str, scene_frame: str,
) -> List[RobotVQA]:
    """Counting — text-only task, GT from instance masks."""
    pairs = []
    from collections import Counter
    counts = Counter(o["category"] for o in visible)

    for cat, count in list(counts.items())[:3]:
        s = _seed(scene_id, phase, frame_id, f"cnt_{cat}")
        gt = str(count)
        pool = [str(i) for i in range(max(count + 3, 5)) if str(i) != gt]
        distractors = random.Random(s).sample(pool, min(3, len(pool)))
        while len(distractors) < 3:
            distractors.append("0")
        opts, idx = _shuffle(gt, distractors[:3], s)

        pairs.append(RobotVQA(
            scene_id=scene_id, phase=phase, frame_id=frame_id,
            task="counting",
            image_only_prompt=f"How many {cat} objects are in this scene?",
            image_text_prompt=f"How many {cat} objects are visible in this scene?",
            text_only_prompt=f"A robot is packing all {cat} objects. How many does it need to pick?",
            reference_image=None,
            scene_frame=scene_frame,
            correct_answer=gt, options=opts, correct_index=idx,
            robot_operation="task_progress",
            metadata={"category": cat, "count": count},
        ))

    return pairs


# ─── Evaluation ──────────────────────────────────────────────────────────

CONTEXT_MODES = ("image_only", "image_text", "text_only")


def evaluate(
    predictions: Dict[str, List[int]],  # {mode: [pred_indices]}
    ground_truth: List[RobotVQA],
) -> Dict:
    """Evaluate predictions across context modes.

    Args:
        predictions: {mode_name: list of predicted option indices}
                     e.g. {"image_only": [0,1,3,...], "text_only": [2,0,1,...]}
        ground_truth: list of RobotVQA

    Returns:
        Per-mode accuracy + per-task + per-phase + cross-mode comparison
    """
    results = {}
    for mode, preds in predictions.items():
        assert len(preds) == len(ground_truth), f"{mode}: {len(preds)} preds vs {len(ground_truth)} GT"

        by_task, by_phase = {}, {}
        correct = 0
        for pred, qa in zip(preds, ground_truth):
            hit = (pred == qa.correct_index)
            correct += hit
            by_task.setdefault(qa.task, []).append(hit)
            by_phase.setdefault(qa.phase, []).append(hit)

        n = len(preds)
        pa = {p: sum(v) / len(v) for p, v in by_phase.items()}
        mode_result = {
            "accuracy": correct / n if n else 0.0,
            "per_task": {t: sum(v) / len(v) for t, v in by_task.items()},
            "per_phase": pa,
        }
        if 0 in pa and 1 in pa:
            mode_result["str_clu_to_int"] = pa[1] - pa[0]
        if 1 in pa and 2 in pa:
            mode_result["str_int_to_cln"] = pa[2] - pa[1]
        results[mode] = mode_result

    # Cross-mode comparison: which mode is most robust?
    if len(results) > 1:
        results["mode_comparison"] = {
            mode: r["accuracy"] for mode, r in results.items() if mode != "mode_comparison"
        }

    return results
