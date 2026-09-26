# RPX supplementary appendix: build

The eight-page main paper is frozen: `golden/RPX-19.pdf` is used as-is and is never recompiled.

1. Compile the appendix with `tectonic appendix.tex` to produce `appendix.pdf`.
2. Run `python3 tools/assemble_final.py` to create `RPX_final_with_appendix.pdf`. The assembler performs a 200-dpi pixel comparison of pages 1–8 against the golden paper.
3. After edits that alter pagination, refit the column tables with `python3 tools/fit_block_tables.py` before compiling again.

`appendix.tex` preserves the main paper's numbering by starting at Fig. 5, Table VIII, and Eq. (4), and resolves the four main-paper references through `main_paper_refs.tex`.

Normalized raw inputs for T1–T6 are stored under `tools/raw_metrics/`. The T5/T6 inputs contain the five attribute question types used by the paper. The recovered VQA and text-prompt tracking CSVs retain model identifiers, contexts, views, counts, and unrounded native metrics needed to regenerate the tables.
