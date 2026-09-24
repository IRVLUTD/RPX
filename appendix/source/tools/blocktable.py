"""Shared writer for column-block tables in the two-column appendix.

A table is written as a sequence of unbreakable column blocks (minipages):
the first carries the caption, every later block starts with
'TABLE N (continued)' and repeats the column heads. Block sizes (number of
row groups per block) come from tools/block_splits.json, which
tools/fit_block_tables.py fits to the page layout. Column widths are fixed,
so replacing placeholder values does not change the fit."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, 'tools', 'block_splits.json')

def load_splits():
    return json.load(open(SPLITS)) if os.path.exists(SPLITS) else {}

def write(key, label, caption, colspec, headrows, groups, sep, font, out_path, note=None):
    sizes = load_splits().get(key) or [len(groups)]
    if sum(sizes) != len(groups):            # stale split: fall back to one block
        sizes = [len(groups)]
    L = [f'% Generated block table ({key}); block sizes from tools/block_splits.json -- do not edit by hand.']
    k = 0
    for b, n in enumerate(sizes):
        L.append('\\par\\noindent\\begin{minipage}[t]{\\columnwidth}\\centering' + font)
        if b == 0:
            L.append(f'{{\\normalsize\\captionof{{table}}{{{caption}}}\\label{{{label}}}}}')
        else:
            L.append(f'{{\\footnotesize TABLE~\\ref{{{label}}} (continued)}}\\par\\vspace{{3pt}}')
        L.append(f'\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{headrows}\n\\midrule')
        chunk = groups[k:k + n]; k += n
        for gi, g in enumerate(chunk):
            L.append(g + (f' \\\\{sep}' if gi < len(chunk) - 1 else ' \\\\'))
        L.append('\\bottomrule\n\\end{tabular}')
        if note and b == len(sizes) - 1: L.append(note)
        L.append('\\end{minipage}\\par\\medskip')
    open(out_path, 'w').write('\n'.join(L) + '\n')
