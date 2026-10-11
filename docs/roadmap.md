# Roadmap

What is planned and in what order. Reasons and results go in `decisions.md`; this file only tracks what is next. Updated 2026-10-11.

## Now

Done (D-024): S2 failed as a proxy for real-voice comparisons (seed spread as large as the effects; only invariance reproduces); weights EMA rejected (off by default, code kept); the Conformer's Hillenbrand collapse traced to short inputs and fixed by mixed crop lengths, at a cost in women's and children's F1 for both architectures.

1. **Training speed** (below, item 3). Done 2026-10-11: AMR streaming (loader with codecs 97 → 829 s/s at fixed crops, 139 → 1036 mixed; 4 workers give about 900–1100 s/s, so 4–6 workers suffice for S2); validation cached on the GPU. Parked: `torch.compile` (no Triton on ROCm under Windows; hand-captured HIP graphs abort in hipBLASLt, and rocBLAS makes a step 4× slower; eager S2 already keeps the GPU about 95 % busy), so kernel fusion waits for Linux (WSL2 or a rented GPU). Scheduling (benchmarks 2026-10-11, S2, 500 steps, alternated): on a quiet machine 4 and 16 workers and either priority all give 106–117 s/s, with jumps between about 106 and 116 within a configuration (the main thread's core placement: it runs on the two CPPC-best cores, cores 6 and 7, and GPU use follows its utility). Under CPU load, above-normal priority gains 0–4 % (24 spinning processes: 70 against 68–70 s/s; the CPU closed phase alongside: 78.5 / 90.8 against 75.0 / 89.8), while the load itself costs 15–36 %: keep measures off the CPU during training rather than raise priority. 4 workers for S2. The 81–84 s/s plateau seen late morning was transient. Closed phase on the GPU (`add_model.py`, `VA_DEVICE=cuda`): fp32, batched by frame count, MIOpen fast find mode, 1.9 min, identical to the CPU. Next: the GPU eval suite.
2. **Coupled synthesis cost**: done 2026-10-11 (D-023). On a quiet machine, 40 tube clips: uncoupled 0.234 → 0.166 and coupled 0.860 → 0.241 CPU s per s of audio (the tube chain matrix vectorised over sections; `open_phase` without LF root solves). Existing sets regenerate bit-identically. Remaining coupled cost: closed-glottis poles (about 14 s per 40 clips, shared with uncoupled), the time-domain voice path and the open-glottis search (about 11 s). Fixed 40-step continuation jumps to a neighbouring root for about 1 pole in 2000 (against 160 steps); adaptive continuation was tried and dropped (1.4–2x faster at equal reliability).
3. **Training speed** (timeboxed to about a day; a quiet-machine benchmark before and after each change). Run-to-run noise means comparisons need full-size runs or many seeds (D-024), so throughput sets how much can be tested. CPU and GPU share one power budget on this machine: while a CPU measure ran, the GPU clock fell from about 2.6 to 2.1 GHz and S2 slowed from about 100 to 86 s/s. S2's training loop is limited by the CPU thread that queues its many small GPU operations; CPU boost off slowed S2 to 78–81 s/s, so keep boost on and lower everything else's CPU use. Recipe unchanged (existing seeds stay comparable), in order:
   - AMR through persistent ffmpeg processes per loader worker: no process launches per clip. Launches serialise on Windows (about 6.5 AMR round trips/s across 16 processes), hung twice on 2026-10-11, and cause CPU bursts that dip the GPU;
   - `torch.compile`, ideally with graph capture: a 30-minute feasibility check on ROCm under Windows first;
   - a GPU eval suite: one command for a list of checkpoints, running the paired Hillenbrand / PTDB-TUG measures, the invariance probe and the closed phase on the GPU in one process, with audio cached at 16 kHz and clips batched by length. For one new model on a quiet machine (2026-10-11): paired 1.7 min and probe 0.6 min on the GPU, closed phase 5.2 min on the CPU; about 20 min per S2 run when the closed phase shared the machine with training. Measures then run between training runs, not alongside them;
   - scheduling: training process at above-normal priority (loader workers stay normal); fewer loader workers (10–12) now that the loader has headroom; the hidden core-parking setting (`CPMINCORES`), since Windows parks the second CCD and the main thread may share a core with a worker. Pin the main thread only if these help, using AMD's preferred-core order (CPPC). On this machine (HWiNFO, 2026-10-11), fastest first: cores 7, 8, 5, 4, 1, 6, 3, 2 (first CCD), then 11, 12, 13, 15, 10, 16, 9, 14 (HWiNFO numbers cores from 1; Windows counts logical processors from 0, two per core).
   Recipe changes (need fresh baseline seeds): larger batches, combined with the learning-rate sweep below so the re-baselining happens once.
4. **Crop length and the Conformer** (D-021 item 1; D-024). On S2, one seed each, read with care (S2 is noisy on real voices; synthetic validation is not):
   - a long-weighted mix (about two-thirds of frames in crops of 194 frames or more) on the Conformer at 3.5k steps: does women's and children's F1 recover while the short-input fix holds?
   - if that is ambiguous: mixed crops at about 9.3k steps (the same long-crop exposure as 3.5k fixed steps) against a fixed-200 run at the same steps;
   - then full size: the Conformer with the chosen crop mix against the full-size v2 seeds, with seeds of its own.
5. **v3c, glottal coupling** (D-023), with a decision on whether to bundle the back-rounded-vowel coverage fix: more lip rounding and protrusion in the tube, and a lower cascade F2 floor of about 500 Hz × size.

## Training recipe experiments (deferred)

Each would run on S2 and train_v2 against the three v2 seeds. The cheap screen is synthetic validation loss and F1–F3 error; anything that passes then gets the subset real-voice measures with 2–3 seeds. Optimiser and schedule choices tend to carry over to larger models. The best learning rate does not: it changes with model width.

| Experiment | Why |
|---|---|
| Shorter warmup (about 250–300 steps) | 1000 steps is 29 % of a 3.5k run |
| Larger batch (64), learning rate and steps adjusted | halves the per-step CPU cost per second of audio for small models |
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
- A learnable front end, much later: train with the fixed VQT, then unfreeze it. Learn a few parameters per filter (gain, bandwidth, small centre shifts; as in SincNet or LEAF) rather than free kernels, so the frequency axis keeps its meaning: the f0 comb and the heat-map output axes assume the VQT's bin frequencies. The cached validation log-power would then be recomputed at each eval.
- Bayesian vocal-tract-length estimation for the labeller (`voice-dataset-labeler`, branch `feat_bayesian_vtl`).
- Rented GPUs, if iteration time becomes the limit. Only code and synthetic data would leave this machine; the evaluation sets stay local, with checkpoints measured here. First check the loader under Linux (WSL, or a one-hour rental): process start-up there takes milliseconds, not about 120 ms as on Windows, so AMR may not be a bottleneck there. Then run one machine's worth of known runs (the three S2 v2 seeds) to check that results agree across platforms (CUDA there, ROCm here).

## Housekeeping

- Merge `m3-resynth` into main. Delete the `reports/analysis` junction in the main checkout first: it points at this branch's tracked copy.
- Update the labeller submodule pin (upstream main is 9 commits ahead and changes `extractors/tvlp.py`).
- Each chain under `scripts/analysis/chains/` runs from a copy, so a running queue isn't affected when its committed file changes.
