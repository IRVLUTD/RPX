# RPX supplementary appendix: build

The eight-page main paper is frozen: `golden/RPX-19.pdf` is used as-is and is never recompiled.

1. Compile the appendix: `tectonic appendix.tex` (or `xelatex`/`pdflatex` + bibtex-free run) → `appendix.pdf`.
   `appendix.tex` uses the main paper's preamble, starts numbering at Fig. 5 / Table VIII / Eq. (4),
   and resolves the four main-paper cross-references via `main_paper_refs.tex`.
2. Assemble: `python3 tools/assemble_final.py` → `RPX_final_with_appendix.pdf`, with a
   200-dpi pixel check of pages 1–8 against the golden master.

Author inputs are listed in CAMERA_READY_TODO.md. After editing text that shifts the appendix layout,
re-fit the column-block tables with `python3 tools/fit_block_tables.py`.
