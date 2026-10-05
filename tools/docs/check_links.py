#!/usr/bin/env python3
"""Reject missing pages, images, downloads and anchors in generated toolkit docs."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE_PATH = "/RPX/toolkit-docs/"


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("id", "name"):
            if values.get(key):
                self.ids.add(values[key])
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key])


def check(root: Path) -> None:
    root = root.resolve()
    pages = {p.resolve(): Page(p.read_text()) for p in root.rglob("*.html")}
    errors = []
    checked = 0
    for source, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                if url.netloc != "irvlutd.github.io" or not url.path.startswith(
                    SITE_PATH
                ):
                    continue
                target = root / unquote(url.path[len(SITE_PATH) :])
            elif url.path.startswith(SITE_PATH):
                target = root / unquote(url.path[len(SITE_PATH) :])
            elif url.path.startswith("/"):
                errors.append(
                    f"{source.relative_to(root)}: unexpected site-root link {link}"
                )
                continue
            else:
                target = source.parent / unquote(url.path) if url.path else source
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if not target.is_file():
                errors.append(f"{source.relative_to(root)}: missing target {link}")
            elif url.fragment and target.suffix == ".html":
                fragment = unquote(url.fragment)
                if target in pages and fragment not in pages[target].ids:
                    errors.append(f"{source.relative_to(root)}: missing anchor {link}")
    if errors:
        raise SystemExit(
            "Broken documentation links:\n" + "\n".join(sorted(set(errors)))
        )
    print(
        f"Verified {checked} internal links and anchors across {len(pages)} HTML pages."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    check(parser.parse_args().root)
