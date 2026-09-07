"""Q11: empirically quantify how many EXISTING rows would become ambiguous
if the 4 slash/ampersand compound material values (air_duster_can
'plastic/metal', handy_pot 'metal & plastic', power_drill 'plastic & steel',
remote_controller 'plastic & rubber') were split into atomic tokens
(['plastic','metal'] etc.) instead of kept as one opaque string.

Restricted to the 7 KEEP_STAGED scenes still on local disk -- restaging the
other 93 to get real per-frame object lists would violate the no-rerun
constraint (see Q13's determination in the report). Disclosed as a partial,
not full-dataset, empirical measurement; extrapolated cautiously.
"""
import glob
import json
import os
import sys
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))
from generate_spatial_gt import load_mapping, imread_mask, MIN_MASK_AREA_PX  # noqa: E402
from lib import object_attrs, load_fewsol_lookup  # noqa: E402
import sos_catalog as sc  # noqa: E402

STAGE_ROOT = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged"
SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
KEEP_STAGED = ["scene001", "scene004", "scene010", "scene016", "scene046", "scene069", "scene098"]

COMPOUNDS = {
    "plastic/metal": ["plastic", "metal"],
    "metal & plastic": ["metal", "plastic"],
    "plastic & steel": ["plastic", "steel"],
    "plastic & rubber": ["plastic", "rubber"],
}


def main():
    lookup = load_fewsol_lookup(SOS_ROOT)
    n_frames_checked = 0
    n_facts_checked = 0
    n_would_break = 0
    examples = []

    for scene in KEEP_STAGED:
        phases = [(str(p), None) for p in (0, 1, 2)] + [("ego", None)]
        for phase_dir, _ in phases:
            root = f"{STAGE_ROOT}/{scene}/{phase_dir}"
            map_path = f"{root}/sam2/mask_to_object.json"
            if not os.path.exists(map_path):
                continue
            mapping = load_mapping(map_path)
            for mask_path in glob.glob(f"{root}/sam2/masks/*.png"):
                mask = imread_mask(mask_path)
                import numpy as np
                present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping
                               and int((mask == v).sum()) >= MIN_MASK_AREA_PX]
                attrs = {}
                for oid in present_ids:
                    a = object_attrs(mapping[oid]["oid"], lookup)
                    if a:
                        attrs[oid] = a
                if len(attrs) < 2:
                    continue
                n_frames_checked += 1

                # material single-attr uniqueness under CURRENT (unsplit) semantics
                owners = {}
                for oid, a in attrs.items():
                    for v in a["material"]:
                        owners.setdefault(v, []).append(oid)
                for val, ov in owners.items():
                    if val not in COMPOUNDS or len(ov) != 1:
                        continue
                    n_facts_checked += 1
                    target_oid = ov[0]
                    atoms = COMPOUNDS[val]
                    # would splitting introduce a NEW owner for any atom?
                    new_owners = set()
                    for oid2, a2 in attrs.items():
                        if oid2 == target_oid:
                            continue
                        for v2 in a2["material"]:
                            v2_atoms = COMPOUNDS.get(v2, [v2])
                            if set(atoms) & set(v2_atoms):
                                new_owners.add(oid2)
                    if new_owners:
                        n_would_break += 1
                        if len(examples) < 8:
                            examples.append({
                                "scene": scene, "phase": phase_dir, "frame": os.path.basename(mask_path),
                                "compound": val, "would_be_ambiguous_with": [mapping[o]["name"] for o in new_owners],
                            })

    print(f"KEEP_STAGED scenes checked: {len(KEEP_STAGED)}")
    print(f"frames with >=2 attributable objects: {n_frames_checked}")
    print(f"currently-unique material facts involving a compound value: {n_facts_checked}")
    print(f"of those, would become AMBIGUOUS if split into atoms: {n_would_break} "
          f"({100*n_would_break/max(n_facts_checked,1):.1f}%)")
    print("\nexamples:")
    for e in examples:
        print(" ", e)
    print("\nNOTE: this is a 7-scene empirical sample, not a full-dataset measurement -- "
          "see report Q13 for why the other 93 scenes' per-frame object lists are not "
          "recoverable without restaging (a real, disclosed limitation, not an omission).")


if __name__ == "__main__":
    main()
