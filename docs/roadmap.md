# Roadmap

What is planned and in what order. Reasons and results go in `decisions.md`; this file only tracks what is next. Updated 2026-10-10.

## Now

1. **S2 proxy check and the weights-EMA test** (D-024; running). S2 is the 1.75M-parameter model: three seeds each of train_v2, train_v3b and train_v1 at 3.5k steps, scored with the subset measures. Each run that kept a moving average of its weights (EMA) is also scored with those averaged weights. Results by about midnight on 2026-10-10.
2. **Time the coupled synthesis path** (D-023) on a quiet machine, against the harmonic path. Then speed up the open-glottis pole search (fewer continuation steps, or a cached table), so coupled rendering costs about the same as uncoupled.
3. **Conformer swap** (D-021 item 1) on S2, if S2 passes as a proxy.
4. **v3c, glottal coupling** (D-023), with a decision on whether to bundle the back-rounded-vowel coverage fix: more lip rounding and protrusion in the tube, and a lower cascade F2 floor of about 500 Hz × size.

## Training recipe experiments (deferred)

Each would run on S2 and train_v2 against the three v2 seeds. The cheap screen is synthetic validation loss and F1–F3 error; anything that passes then gets the subset real-voice measures with 2–3 seeds. Optimiser and schedule choices tend to carry over to larger models. The best learning rate does not: it changes with model width.

| Experiment | Why |
|---|---|
| Shorter warmup (about 250–300 steps) | 1000 steps is 29 % of a 3.5k run |
| Learning-rate sweep (×0.5, ×2, ×3) | 1e-3 has never been tuned |
| Decay length about 20 % of the run (now 43 %) | the WSD literature finds about 20 % enough, so runs get cheaper |
| Muon for 2-D weights (GRU, linear), AdamW for the rest | built into torch 2.13; strong results on transformers, and relevant to the Conformer |
| Adan / AdanW | gains reported in vision and NLP papers; three betas to tune; not in torch |
| Schedule-free AdamW, retested with seeds | D-018 was a single-run comparison: better F1, worse f0 RPA |
| Cosine instead of linear warmup | usually small once the warmup length is right |
| `torch.compile`, channels-last | speed only; uncertain on ROCm under Windows |

## Later

- D-021: longer and mixed crop lengths; a per-formant observability label; vocal-tract-length outputs.
- A source head (Rd or open quotient; D-023 plan item 3).
- A mixture-of-hypotheses formant head for assignment swaps (D-021, gap (a)).
- `external/tcn-formants` in the architecture bake-off.
- Bayesian vocal-tract-length estimation for the labeller (`voice-dataset-labeler`, branch `feat_bayesian_vtl`).
- AMR through one long-running ffmpeg process per data-loader worker. Needed only if a model trains faster than the loader's 134–155 s/s.

## Housekeeping

- Merge `m3-resynth` into main. Delete the `reports/analysis` junction in the main checkout first: it points at this branch's tracked copy.
- Update the labeller submodule pin (upstream main is 9 commits ahead and changes `extractors/tvlp.py`).
- Each chain under `scripts/analysis/chains/` runs from a copy, so a running queue isn't affected when its committed file changes.
