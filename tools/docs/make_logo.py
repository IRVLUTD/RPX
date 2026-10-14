#!/usr/bin/env python3
"""Draw the RPX logo as SVG: three sensor nodes feeding a camera lens, and the wordmark.

The wordmark is set in Inter SemiBold and converted to outlines, so the SVG needs no
font to render. Colours are the RPX palette (indigo, pink, orange).

    python tools/docs/make_logo.py Inter-SemiBold.ttf logo.svg            # icon + wordmark
    python tools/docs/make_logo.py Inter-SemiBold.ttf logo-dark.svg --dark
    python tools/docs/make_logo.py Inter-SemiBold.ttf favicon.svg --icon-only
"""

from __future__ import annotations

import argparse

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

INDIGO, PINK, ORANGE = "#4F46E5", "#DB2777", "#C2410C"
WORDMARK = {"light": "#5148D5", "dark": "#C7D2FE"}
LENS = (46.0, 26.0)  # the drawing uses a 52-unit-tall grid


def wordmark(font_file: str, text: str, x: float, baseline: float, cap_height: float):
    """SVG path for ``text`` with its cap height scaled to ``cap_height``; returns (path, right edge)."""
    font = TTFont(font_file)
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    em = font["head"].unitsPerEm
    scale = cap_height / (font["OS/2"].sCapHeight or 0.7 * em)
    pen, advance = SVGPathPen(glyphs), 0.0
    for char in text:
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + advance * scale, baseline)))
        advance += glyphs[name].width - 0.02 * em  # slightly tighter than the font's default
    return pen.getCommands(), x + advance * scale


def icon() -> str:
    cx, cy = LENS
    return f"""
  <defs>
    <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{INDIGO}"/><stop offset="0.5" stop-color="{PINK}"/><stop offset="1" stop-color="{ORANGE}"/>
    </linearGradient>
    <radialGradient id="pupil" cx="0.42" cy="0.40" r="0.62">
      <stop offset="0" stop-color="#3B2F7A"/><stop offset="1" stop-color="#14112E"/>
    </radialGradient>
  </defs>
  <g stroke-linecap="round">
    <path d="M19 12.5 L37.5 25.5" stroke="{INDIGO}" stroke-width="2.2" fill="none"/>
    <path d="M15.5 26 L37 26" stroke="{PINK}" stroke-width="2.2" fill="none"/>
    <path d="M19 40 L37.5 27.5" stroke="{ORANGE}" stroke-width="2.2" fill="none"/>
    <circle cx="19" cy="12.5" r="3.3" fill="{INDIGO}"/>
    <circle cx="15.5" cy="26" r="3.3" fill="{PINK}"/>
    <circle cx="19" cy="40" r="3.3" fill="{ORANGE}"/>
    <circle cx="{cx}" cy="{cy}" r="11.2" fill="url(#ring)"/>
    <circle cx="{cx}" cy="{cy}" r="8.1" fill="#FFFFFF"/>
    <circle cx="{cx}" cy="{cy}" r="6.9" fill="url(#pupil)"/>
    <circle cx="{cx}" cy="{cy}" r="6.9" fill="none" stroke="#6D5BD0" stroke-width="0.7" opacity="0.8"/>
    <circle cx="{cx + 2.4}" cy="{cy - 2.6}" r="1.75" fill="#FFFFFF"/>
    <circle cx="{cx - 2.3}" cy="{cy + 2.6}" r="0.75" fill="#FFFFFF" opacity="0.85"/>
  </g>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("font", help="Inter SemiBold (TTF or WOFF)")
    parser.add_argument("out", help="SVG to write")
    parser.add_argument("--dark", action="store_true", help="light wordmark for dark backgrounds")
    parser.add_argument("--icon-only", action="store_true", help="the icon alone, square (favicon)")
    args = parser.parse_args()
    if args.icon_only:
        body, view_box = icon(), "8 0 52 52"
    else:
        path, right = wordmark(args.font, "RPX", 70.0, 41.0, 31.0)
        colour = WORDMARK["dark" if args.dark else "light"]
        body = icon() + f'\n  <path d="{path}" fill="{colour}"/>'
        view_box = f"0 0 {round(right + 6)} 52"
    with open(args.out, "w") as handle:
        handle.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" role="img" '
            f'aria-label="RPX">{body}\n</svg>\n'
        )


if __name__ == "__main__":
    main()
