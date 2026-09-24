# v8 changes (appendix only; pages 1–8 = RPX-19, 0 changed pixels)

1. **New Fig. 13: SOS modalities.** Replaces the earlier RGB+mask example. It shows one released frame (250/500) of object 49, the power drill:
   - RGB and metric depth (D435);
   - the instance mask;
   - the T265 fisheye stereo pair (left and right);
   - the T265 camera pose, as a top-view trajectory of all 500 frames with the heading at frame 250;
   - the questionnaire, with all released annotator responses next to the canonical entry.

   Everything is taken directly from the released shards.
2. **MOS statement.** Added to Appendix E.A, the Fig. 13 caption and Appendix D.D: the exocentric MOS captures record the same RGB-D, mask, stereo and pose modalities (Ego: RGB and masks), MOS objects inherit the questionnaire through their global identity, and the MOS figures show only the RGB phase images.
3. **SOS coverage corrected.** Measured from the released poses of all 70 objects: there is no turntable. The hand-held rig circles each object 1.7–3.1 times (median 2.6 loops) at about 0.5 m over its 500 frames. Appendix A, E.A and Table XXII previously said "one 360° turntable rotation"; they now use this wording.
4. **Layout.** The SOS catalogue blocks were refitted, and the final page is balanced across both columns. The appendix is 28 pages; the combined PDF is 36.

Empty values remain as before.
