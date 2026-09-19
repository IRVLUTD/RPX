"""Canonical IDs must never silently merge objects or erase later-frame IDs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def mapper():
    path = Path(__file__).parents[2] / 'data/mask_annotation/visual_grounding_gt/mask_to_object.py'
    spec = importlib.util.spec_from_file_location('mask_to_object', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_mapping_requires_complete_unique_assignments():
    module = mapper()
    objects = [{'id': 'a', 'name': 'A'}, {'id': 'b', 'name': 'B'}]
    with pytest.raises(ValueError, match='must be assigned'):
        module.build_canonical_remap([1, 9], {1: objects[0]}, objects)
    with pytest.raises(ValueError, match='different scene object'):
        module.build_canonical_remap([1, 9], {1: objects[0], 9: objects[0]}, objects)
    remap = module.build_canonical_remap([1, 9], {1: objects[1], 9: objects[0]}, objects)
    original = np.array([[0, 1, 9]], dtype=np.uint16)
    np.testing.assert_array_equal(module.remap_mask(original, remap), [[0, 2, 1]])
    np.testing.assert_array_equal(original, [[0, 1, 9]])
    with pytest.raises(ValueError, match='Unmapped'):
        module.remap_mask(np.array([[1, 9, 42]], np.uint16), remap)
