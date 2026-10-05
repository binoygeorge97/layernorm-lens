# Plan: P-I decisions applied, the trim code, `prereg/p1.md`

Author-approved queue of 5 Oct 2026; branch `quadrotor-sim`. Nothing runs on quadrotor
training data before the tag `prereg-p1`.

1. **Draft.** Apply the author's decisions on D1–D18:
   - D3: confirm the thrust range is symmetric about hover thrust, and list every box
     range;
   - D9: a positive Spearman correlation, with tie handling;
   - D10: E1b's definition was searched for and not found, so the fallback
     S_out/S_block1 is used;
   - D15: G2 at 2 of 4 cells, stated as a planning gate;
   - D16: (a) "no stabilising gain", (b) trims inside the box, (c) variation across
     trims;
   - the provenance sentence.
2. **Code** (tests on synthetic plants, plus the quadrotor plant for the trims):
   - `p1_hover.py`:
     - `true_y_jacobians` at any (x, u);
     - "no stabilising gain" (the ARE fails, or its gain does not stabilise the
       surrogate's own (A, B)), and H1's per-model flag;
     - `sample_trims` and `check_trims`;
     - `trim_check`: per-trim errors and the error in the surrogate's variation
       relative to hover, against lens distance.
   - `lens/models.py`: `trace(..., upto=j)`, the head applied after block j.
   - `lens/analysis.py`: `attenuation_S`, prediction 4's S_out/S_block1 at block 1's z*
     along a direction.
   - `run.py`:
     - stage `trims` (ungated: the simulator only, no surrogate), which samples 200
       trims with a fixed seed and writes `results/p1/trims/` with a SHA-256 manifest
       before any training;
     - `hover` loads and verifies the trims and adds the trim check;
     - stage `predictions` (gated), which applies every pre-registered rule
       (predictions 1–5, H1, G2) to the stage outputs.
3. **`prereg/p1.md`**, in the style of `prereg/r6.md`, committed alone and pushed. Then
   stop for the author's tag.
