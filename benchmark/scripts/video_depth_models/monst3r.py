"""Skeleton adapter for MonST3R (Video Depth roster entry ``monst3r``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model monst3r`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``zhang2024monst3r``.
Install hint: ``pip install monst3r (or `pip install git+https://github.com/Junyi42/monst3r`)``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("monst3r")
