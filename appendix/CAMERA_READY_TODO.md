# Camera-ready TODO

1. **Replace Fig. 5** with the genuine mask-annotation workflow asset: `figures/fig5_annotation_workflow.pdf`, 7.0 in × about 1.6 in.
2. **Fill the final raw metrics for T1–T6** in `tools/raw_metrics/t1.csv` … `t6.csv`, then run `python3 tools/raw_metrics/build_raw_tables.py`.
3. **Fill the desirability windows** for Acc@0.5, GIoU and Center-in-GT in `tools/protocol/desirability_windows.csv`.
4. **Fill the configuration fields** in `tools/protocol/model_config.csv`: the two VLM decoding fields and the T4 frame gap. Then run `python3 tools/protocol/build_protocol_tables.py`.
5. **Site 15 photo (optional).** The package only has a 560 × 225 reference image; it prints at about 180 dpi. Its full-resolution JPEG is blank. Drop a full photo in and rerun `python3 tools/web_assets/build_web_assets.py <png_assets>`.
6. **Recompile and QA.**
   - Run `tectonic appendix.tex`.
   - If text moved, run `python3 tools/fit_block_tables.py` and `python3 tools/catalogue/fit_catalogue.py`.
7. **Reassemble.** Run `python3 tools/assemble_final.py` and confirm "main-paper check PASSED".
