#!/usr/bin/env python3
"""Assemble the deliverable: the frozen eight-page main paper (golden/RPX-19.pdf)
followed by the compiled appendix (appendix.pdf), then verify that pages 1-8
render pixel-identically to the golden master.
    python3 tools/assemble_final.py [out.pdf]"""
import os, sys
import fitz                      # PyMuPDF, used only for the render check
from pypdf import PdfReader, PdfWriter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, 'golden', 'RPX-19.pdf'); APP = os.path.join(ROOT, 'appendix.pdf')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'RPX_final_with_appendix.pdf')
w = PdfWriter(); w.append(PdfReader(GOLD)); w.append(PdfReader(APP))   # page objects copied as-is
with open(OUT, 'wb') as f: w.write(f)
G, F = fitz.open(GOLD), fitz.open(OUT); bad = 0
for i in range(G.page_count):
    a = G[i].get_pixmap(dpi=200, alpha=False).samples; b = F[i].get_pixmap(dpi=200, alpha=False).samples
    n = sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else -1
    same = G[i].read_contents() == F[i].read_contents()
    print(f'page {i + 1}: {"0" if a == b else "DIFFERENT"} changed pixels; content stream identical: {same}')
    bad += (a != b) or not same
print(f'{F.page_count} pages ({G.page_count} main + {F.page_count - G.page_count} appendix); main-paper check', 'PASSED' if not bad else 'FAILED')
sys.exit(1 if bad else 0)
