"""Skeleton adapter for VGGT-Ω (Video Depth roster entry ``vggt-omega``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model vggt-omega`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``vggt_omega``.
Install hint: ``pip install vggt (or upstream Meta repo); Ω = the large variant``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("vggt-omega")
