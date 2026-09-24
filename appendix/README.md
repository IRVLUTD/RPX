# RPX supplementary appendix

- `RPX_final_with_appendix.pdf`: the eight-page main paper (RPX-19, unchanged) followed by Appendices A–E (35 pages).
- `RPX_appendix_only.pdf`: the appendix alone.
- `source/`: self-contained appendix LaTeX source and assets. See `source/README_BUILD.md`.
  - Build the appendix: `tectonic appendix.tex`.
  - Assemble with the frozen main paper: `python3 tools/assemble_final.py` (includes a pixel check of pages 1–8).
- `CHANGES.md`: latest changes. `CAMERA_READY_TODO.md`: remaining author inputs (Fig. 5 asset, raw T1–T6 metrics, VQA desirability windows, VLM decoding fields).
