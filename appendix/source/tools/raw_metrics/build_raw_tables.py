#!/usr/bin/env python3
"""Build the appendix per-model raw-metric tables (T1-T6).

Fill values in tools/raw_metrics/<task>.csv (one row per model x view; leave a
cell empty while its value is pending -> printed as an em dash), then run
    python3 tools/raw_metrics/build_raw_tables.py
from the project root and recompile. Values are printed exactly as written in
the CSV, so format them there (e.g. 0.123, 45.6). Column widths are fixed, so
filling values does not change the layout. SPLITS gives, per task, the model
indices at which a table continues in the next column."""
import csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
V3 = ['Clutter', 'Interaction', 'Clean']
V4 = ['Clutter', 'Interaction', 'Clean', 'Ego']
ABBR = {'Clutter': 'Clu.', 'Interaction': 'Int.', 'Clean': 'Clean', 'Ego': 'Ego'}
VLM = ['Cosmos Reason2 8B', 'Cosmos Reason2 2B', 'Qwen3-VL 8B', 'Qwen3-VL 2B', 'InternVL 3.5 14B',
       'InternVL 3.5 1B', 'PaliGemma 2 10B', 'PaliGemma 2 3B', 'DeepSeek-VL2 4.5B', 'DeepSeek-VL2 1.0B',
       'Florence 2 0.77B', 'Florence 2 0.23B']
TASKS = [
 ('t1', 'tab:raw-t1', 'T1 image depth: native metrics per phase.',
  [('AbsRel', '$\\downarrow$'), ('RMSE', '$\\downarrow$'), ('SILog', '$\\downarrow$'), ('$\\delta_1$', '$\\uparrow$')], V3,
  ['ZipDepth', 'HyDen', 'FE2E', 'DA3 Metric-L', 'UniDepth V2', 'MoGe-2 ViT-L', 'Lotus-2', 'Depth Pro',
   'Metric3D V2', 'DA-V2 Large']),
 ('t2', 'tab:raw-t2', 'T2 video depth: native metrics per phase.',
  [('AbsRel', '$\\downarrow$'), ('RMSE', '$\\downarrow$'), ('SILog', '$\\downarrow$'), ('$\\delta_1$', '$\\uparrow$'),
   ('TGM', '$\\downarrow$'), ('TGSE', '$\\downarrow$')], V3,
  ['DA3 (video)', 'Video-DA', 'ViGeo', 'VGGT', 'MonST3R', 'GemDepth', 'DepthCrafter', 'DVD v1.1',
   'ChronoDepth', 'RollingDepth']),
 ('t3', 'tab:raw-t3', 'T3 multi-object tracking: native metrics per view and phase.',
  [('HOTA', '$\\uparrow$'), ('MOTA', '$\\uparrow$'), ('IDSW', '$\\downarrow$')], V4,
  ['SAM 2 Long', 'DAM4SAM', 'EdgeTAM', 'SAM 2++', 'SAM 2', 'Cutie', 'SAM 3.1 (bbox)', 'MiTS', 'XMem',
   'SAM 3.1 (text)', 'GSAM2 (text)']),
 ('t4', 'tab:raw-t4', 'T4 relative camera pose: native metrics for the Clutter and Clean pair sets.',
  [('AUC@5$^\\circ$', '$\\uparrow$'), ('AUC@10$^\\circ$', '$\\uparrow$'), ('AUC@20$^\\circ$', '$\\uparrow$')], ['Clutter', 'Clean'],
  ['DA3', 'VGGT', 'Reloc3R', 'Pi3X', 'MonST3R', 'MUSt3R', 'Fast3R', 'CUT3R', 'MASt3R', 'DUSt3R']),
 ('t5', 'tab:raw-t5', 'T5 regular VQA: native grounding metrics per view and phase (Center: Center-in-GT).',
  [('Acc@0.5', '$\\uparrow$'), ('GIoU', '$\\uparrow$'), ('Center', '$\\uparrow$')], V4, VLM),
 ('t6', 'tab:raw-t6', 'T6 in-context VQA: native grounding metrics per view and phase (Center: Center-in-GT).',
  [('Acc@0.5', '$\\uparrow$'), ('GIoU', '$\\uparrow$'), ('Center', '$\\uparrow$')], V4, VLM),
]
# model index at which each table continues in a new column block (tuned to the page flow)
NOTE = {}   # the working-copy note now lives once in Appendix A (footnote)
# fixed widths (model column, metric column) so that filled values do not reflow the layout
WIDTH = {'t1': ('2.3cm', '1.05cm'), 't2': ('2.0cm', '0.8cm'), 't3': ('2.3cm', '1.25cm'),
         't4': ('2.3cm', '1.25cm'), 't5': ('2.65cm', '1.45cm'), 't6': ('2.65cm', '1.45cm')}   # temporary working-copy note under the first table

def template(key, metrics, views, models):
    path = os.path.join(HERE, f'{key}.csv')
    if os.path.exists(path): return
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'view'] + [strip(m) for m, _ in metrics])
        for mo in models:
            for v in views: w.writerow([mo, v] + [''] * len(metrics))

def strip(s): return s.replace('$\\delta_1$', 'delta1').replace('$^\\circ$', 'deg')

def load(key):
    rows = {}
    with open(os.path.join(HERE, f'{key}.csv')) as f:
        r = csv.reader(f); next(r)
        for row in r: rows[(row[0], row[1])] = row[2:]
    return rows

def cell(v): return v.strip() if v and v.strip() else '---'

import sys
sys.path.insert(0, os.path.dirname(HERE))
import blocktable

def stack(vals, align):
    return '\\begin{tabular}[c]{@{}' + align + '@{}}' + '\\\\'.join(vals) + '\\end{tabular}'

for key, label, cap, metrics, views, models in TASKS:
    template(key, metrics, views, models)
    data = load(key)
    nm = len(metrics)
    mw, cw = WIDTH[key]
    colspec = (f'@{{}}>{{\\raggedright\\arraybackslash}}m{{{mw}}}@{{\\hspace{{3pt}}}}l'
               + f'>{{\\centering\\arraybackslash}}m{{{cw}}}' * nm + '@{}')
    headrows = ('\\multicolumn{1}{@{}l}{Model} & View & ' + ' & '.join(f'\\makebox[\\linewidth][c]{{{m}}}' for m, _ in metrics) + ' \\\\ '
                + ' & & ' + ' & '.join(a for _, a in metrics) + ' \\\\')
    groups = []
    for mo in models:
        vals = [data.get((mo, v), [''] * nm) for v in views]
        cells = [stack([ABBR[v] for v in views], 'l')] + [stack([cell(vals[vi][mi]) for vi in range(len(views))], 'c') for mi in range(nm)]
        groups.append(f'{mo} & ' + ' & '.join(cells))
    blocktable.write(key, label, cap, colspec, headrows, groups, '\\rawsep', '\\rawfont',
                     os.path.join(ROOT, 'generated', f'raw_{key}_table.tex'))
print('built', ', '.join(f'raw_{t[0]}_table.tex' for t in TASKS))
