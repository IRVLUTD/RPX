#!/usr/bin/env python3
"""Render the explicit module inventory without hiding modules via __all__."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
from pathlib import Path
from typing import get_type_hints

import pdoc.doc
import pdoc.render


def render(output: Path, templates: Path, modules: list[str]) -> None:
    imported = [importlib.import_module(name) for name in modules]
    # Generated dataclass constructors carry inherited string annotations in
    # the subclass's namespace. Resolve them from the class/MRO, where Python
    # knows the original globals. This affects the isolated docs process only.
    seen = set()
    for module in imported:
        for cls in vars(module).values():
            if (
                not inspect.isclass(cls)
                or not dataclasses.is_dataclass(cls)
                or cls in seen
            ):
                continue
            seen.add(cls)
            if not cls.__module__.startswith("rpx_benchmark.tasks."):
                continue
            hints = get_type_hints(cls)
            sig = inspect.signature(cls.__init__)
            cls.__init__.__signature__ = sig.replace(
                parameters=[
                    p.replace(annotation=hints.get(p.name, p.annotation))
                    for p in sig.parameters.values()
                ],
            )
    pdoc.render.configure(template_directory=templates)
    docs = {name: pdoc.doc.Module.from_name(name) for name in modules}
    output.mkdir(parents=True, exist_ok=True)
    for name, module in docs.items():
        target = output / (name.replace(".", "/") + ".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pdoc.render.html_module(module, docs))
    (output / "search.js").write_text(pdoc.render.search_index(docs))
    print(f"Rendered {len(docs)} API modules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()
    render(args.output, args.templates, args.modules)
