#!/usr/bin/env python3
"""Fit the SOS catalogue column blocks to the compiled layout (greedy, largest block
that stays in its column). Run from the project root: python3 tools/catalogue/fit_catalogue.py [tex cmd]"""
import os, re, subprocess, sys
import fitz
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = sys.argv[1:] or ['tectonic', '--keep-logs', 'appendix.tex']
N = 70
def build(sizes):
    sp = []; k = 0
    for z in sizes[:-1]: k += z; sp.append(str(k))
    subprocess.run([sys.executable, 'tools/catalogue/build_catalogue.py', ','.join(sp) or '70'], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(TEX, cwd=ROOT, capture_output=True)
def cols():
    d = fitz.open(os.path.join(ROOT, 'appendix.pdf')); out = []
    for i, p in enumerate(d):
        for b in sorted(p.get_text('blocks'), key=lambda b: (b[0] > p.rect.width / 2, b[1])):
            if re.match(r'^TABLE [IVXL]+(:| \(continued\))', b[4].strip()) and ('Canonical SOS' in b[4] or 'continued' in b[4]):
                num = b[4].split()[1].rstrip(':')
                out.append((num, 2 * i + (1 if b[0] > p.rect.width / 2 else 0)))
    num = [n for n, c in out if True]
    last = out[-1][0] if out else None
    return [c for n, c in out if n == last]
def overfull():
    return 'Overfull \\vbox' in open(os.path.join(ROOT, 'appendix.log'), errors='ignore').read()
def chunk(r, c=14):
    return [c] * (r // c) + ([r % c] if r % c else [])
fixed = []
while sum(fixed) < N:
    rem = N - sum(fixed); b = len(fixed)
    def trial(m):
        build(fixed + [m] + chunk(rem - m))
        return cols(), overfull()
    c0, _ = trial(1); expect = c0[b] if b == 0 else prev + 1
    lo, hi = 1, rem
    while lo < hi:
        mid = (lo + hi + 1) // 2; c, of = trial(mid)
        if len(c) > b and c[b] == expect and not of: lo = mid
        else: hi = mid - 1
    fixed.append(lo); build(fixed + chunk(N - sum(fixed))); prev = cols()[b]
    print('block', b, 'size', lo, flush=True)
build(fixed); print('sizes', fixed)
