#!/usr/bin/env python3
"""Fit block sizes of the column-block tables to the compiled page layout.

Run from the project root after any change that moves the appendix text:
    python3 tools/fit_block_tables.py [tex-command]
Writes tools/block_splits.json. Each block receives the largest number of row
groups that fits in its column without being pushed to the next column."""
import json, os, re, subprocess, sys
import fitz
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = sys.argv[1:] or ['tectonic', '--keep-logs', 'appendix.tex']
SPLITS = os.path.join(ROOT, 'tools', 'block_splits.json')
# (split key, printed table number, row groups, builder)
TABLES = [('desirability', 'XIII', 15, 'tools/protocol/build_protocol_tables.py'),
          ('t1', 'XIV', 10, 'tools/raw_metrics/build_raw_tables.py'),
          ('t2', 'XV', 10, 'tools/raw_metrics/build_raw_tables.py'),
          ('t3', 'XVI', 11, 'tools/raw_metrics/build_raw_tables.py'),
          ('t4', 'XVII', 10, 'tools/raw_metrics/build_raw_tables.py'),
          ('t5', 'XVIII', 12, 'tools/raw_metrics/build_raw_tables.py'),
          ('t6', 'XIX', 12, 'tools/raw_metrics/build_raw_tables.py')]
PDF = os.path.join(ROOT, 'appendix.pdf'); LOG = os.path.join(ROOT, 'appendix.log')

def build(splits):
    json.dump(splits, open(SPLITS, 'w'), indent=1)
    for b in sorted({t[3] for t in TABLES}):
        subprocess.run([sys.executable, b], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(TEX, cwd=ROOT, capture_output=True)

def block_columns(num):
    """Column index (2*page + col) of every block header of table `num`, in order."""
    d = fitz.open(PDF); out = []
    pat = re.compile(r'^TABLE ' + num + r'(:| \(continued\))')
    for i, p in enumerate(d):
        for b in sorted(p.get_text('blocks'), key=lambda b: (b[0] > p.rect.width / 2, b[1])):
            if pat.match(b[4].strip()): out.append(2 * i + (1 if b[0] > p.rect.width / 2 else 0))
    return out

def overfull():
    return 'Overfull \\vbox' in open(LOG, errors='ignore').read()

splits = {k: [1] * n for k, _, n, _ in TABLES}
for key, num, n, _ in TABLES:
    fixed = []
    while sum(fixed) < n:
        rem = n - sum(fixed); b = len(fixed)
        def trial(m):
            splits[key] = fixed + [m] + [1] * (rem - m); build(splits)
            return block_columns(num), overfull()
        cols, _ = trial(1)
        expect = cols[b] if b == 0 else block_columns_prev + 1
        lo, hi = 1, rem
        while lo < hi:                      # largest m that stays in the expected column
            mid = (lo + hi + 1) // 2
            cols, of = trial(mid)
            if len(cols) > b and cols[b] == expect and not of: lo = mid
            else: hi = mid - 1
        fixed.append(lo)
        splits[key] = fixed + [1] * (n - sum(fixed)); build(splits)
        block_columns_prev = block_columns(num)[b]
    splits[key] = fixed
    print(key, fixed, flush=True)
build(splits)
print('final', splits, 'overfull:', overfull())
