# Source-side generators (not needed to compile the paper)

The paper compiles from the committed `generated/*.tex`, `figures/*.tex` and
`assets/` alone. These scripts regenerate those files:

- `raw_metrics/build_raw_tables.py`: fill `raw_metrics/t1.csv` … `t6.csv`
  (one row per model × view; empty cell = em dash) and run it from the project
  root to rewrite `generated/raw_t1_table.tex` … `raw_t6_table.tex`.
  `raw_metrics/splits.txt` sets where each table continues in the next column.
- `catalogue/build_catalogue.py 12,32,51`: rewrites `generated/sos_catalogue.tex`
  from `catalogue/catalogue_canonical.json` (paper-only spelling normalization).
- `esd/build_esd27.py <data dir> <project root>`: rewrites
  `generated/esd_feature_table.tex`, `figures/fig_esd_heatmap.tex` and
  `assets/images/appendix/esd_heatmap_27x300.png` from the released 300×27 ESD
  feature matrix and split file.
