# Roadmap

What is planned and in what order. Reasons and results go in `decisions.md`; this file only tracks what is next. Updated 2026-10-17.

## Now

*State on 2026-10-17 evening. Results and reasoning are in D-025 to D-027 (decisions.md).*

- **Teacher v3** (D-025): conformer_v2r2 (train_v2 + 30 % resynthesised VocalSet and Zeroth-Korean) beat conformer_v2 on all three full-size seeds and replicated on semi-private (logged use). One model: `conformer_v2r2_s1_d16k`; for labelling, the per-frame mean of seeds 1 and 2 (in log frequency).
- **Interaction investigation** (D-026): the women's F1 overread is mostly source–tract interaction; the high-f0 F3 underread is mostly identifiability (sparse harmonics plus tilt), which is what D-027 is for. The return-phase source option is built, off by default.
- **Observability** (D-027): the label, its training-time channel model (validated), train_v2's store and the head are built. Running: `obs_chain.sh` (fresh baselines mixr_conformer_s0–2, since the replica check failed, then obs_conformer_s0–2; `decide_obs_s2.py`: gate / output / fix), decision about 21:30 on 2026-10-17.

## Next, in order

1. **Replica check** (minutes of GPU, after the obs chain): two 50-step runs of the same command on the same code. Tells whether the 7.8 % bandwidth-loss difference against mix_conformer_s0's log is GPU nondeterminism or a code change since 69ccf9a, which matters for older arms paired against mix_conformer and sets the noise floor for item 3.
2. **D-027 decision.** "gate" or "output": the head goes into a full-size teacher, either teacher v3's seeds 1 and 2 retrained with it (about 13 h of GPU) or M4's first teacher (to decide then). "fix": model the failing part better first.
3. **Training recipe screen** on S2 (table below): each arm three seeds on train_v2 against the same-seed baseline, about 2 h of GPU per arm, about a day in all. Rules pre-registered before each arm. Cheaper runs pay off in M4, which retrains several times. Alongside, on the CPU: M3's analysis items (item 5).
4. **Interactive arm on teacher v3's data** (agreed to follow the observability head; pre-register first).
5. **Close M3** (plan §12): Iseli–Alwan validation with predicted F / B by f0 band; the women's and children's F1 offset (model against Hillenbrand's LPC reference); training-pool corpora (VCTK and LibriTTS-R not downloaded; manifests beyond VocalSet). Then M3 acceptance on the gold set with the hard suite, which is a milestone acceptance: the lockbox opens then, once.
6. **M4** (plan §12). Its code (differentiable renderer and reconstruction loss, equivariance pairs, candidate generator with Viterbi on teacher emissions) needs no GPU and can be written during items 3–5. With observability ("gate"), the head decides which frames re-tracking trusts.

## Speed and machine notes

AMR streams through one long-lived ffmpeg per bitrate per loader worker (loader 829–1036 s/s); 4 workers for S2; validation cached on the GPU; closed phase on the GPU (`add_model.py`, `VA_DEVICE=cuda`, identical to CPU); `torch.compile` and HIP graph capture parked (no Triton on ROCm under Windows; capture aborts in hipBLASLt); priority opt-in (`VA_PRIORITY`), no measurable gain; under CPU load S2 loses 15–36 %, so keep measures off the CPU during training. Open: a batched GPU eval suite for the paired measures and the probe (closed phase done); profiling the training step needs ablation timing, since the ROCm profiler's GPU times are unreliable here. Machine notes: CPU boost on; CPPC core order 7, 8, 5, 4, 1, 6, 3, 2, then 11, 12, 13, 15, 10, 16, 9, 14 (HWiNFO numbering from 1); the main training thread runs on cores 6–7 (Windows numbering) and GPU use follows it.

## Training recipe experiments (next item 3)

Each runs on S2 and train_v2 against three same-seed baselines. The cheap screen is synthetic validation loss and F1–F3 error; anything that passes then gets the subset real-voice measures with 2–3 seeds. Optimiser and schedule choices tend to carry over to larger models. The best learning rate does not: it changes with model width.

| Experiment | Why |
|---|---|
| Shorter warmup (about 250–300 steps) | 1000 steps is 29 % of a 3.5k run |
| Larger batch (64), learning rate and steps adjusted | halves the per-step CPU cost per second of audio for small models |
| Learning-rate sweep (×0.5, ×2, ×3) | 1e-3 has never been tuned |
| Decay length about 20 % of the run (now 43 %) | the WSD literature finds about 20 % enough, so runs get cheaper |
| Muon for 2-D weights (GRU, linear), AdamW for the rest | built into torch 2.13; strong results on transformers, and relevant to the Conformer |
| Adan / AdanW | gains reported in vision and NLP papers; three betas to tune; not in torch, on PyPI |
| Schedule-free AdamW, retested with seeds | D-018 was a single-run comparison: better F1, worse f0 RPA |
| Cosine instead of linear warmup | usually small once the warmup length is right |
| `torch.compile`, channels-last | speed only; uncertain on ROCm under Windows |

## Later

- D-021: independent (joint) per-formant outputs (the chained heads let a wrong F1 pull F2 with it; GPU held for now); vocal-tract-length outputs. With the observability label in place (D-027), add voiced clips with little or no formant evidence (a few per cent): unfiltered glottal sources, formants too wide to see, F1 between sparse harmonics at high f0, a single missing formant, so the label says 'unobservable' instead of forcing a guess (given a bare source by mistake on 2026-10-11, the teachers and Praat read formants 100–500 Hz off).
- The return-phase source (D-026, built, off by default) as an M4 source-variety arm.
- Labeller: the median of all three teacher v3 seeds was best on F4 (Hillenbrand 94.2 Hz against 96.3 for the mean of seeds 1–2; D-025), so a hybrid (mean of 1–2 for F1–F3, median of three for F4) is worth a proper check before M4's re-tracking.
- A source head (Rd or open quotient; D-023 plan item 3).
- A mixture-of-hypotheses formant head for assignment swaps (D-021, gap (a)).
- `external/tcn-formants` in the architecture bake-off.
- A learnable front end, much later: train with the fixed VQT, then unfreeze it. Learn a few parameters per filter (gain, bandwidth, small centre shifts; as in SincNet or LEAF) rather than free kernels, so the frequency axis keeps its meaning: the f0 comb and the heat-map output axes assume the VQT's bin frequencies. The cached validation log-power would then be recomputed at each eval.
- Bayesian vocal-tract-length estimation for the labeller (`voice-dataset-labeler`, branch `feat_bayesian_vtl`).
- Rented GPUs, if iteration time becomes the limit. Only code and synthetic data would leave this machine; the evaluation sets stay local, with checkpoints measured here. First check the loader under Linux (WSL, or a one-hour rental): process start-up there takes milliseconds, not about 120 ms as on Windows, so AMR may not be a bottleneck there. Then run one machine's worth of known runs (the three S2 v2 seeds) to check that results agree across platforms (CUDA there, ROCm here).

## Housekeeping

- Update the labeller submodule pin (upstream main is 9 commits ahead and changes `extractors/tvlp.py`).
- Each chain under `scripts/analysis/chains/` runs from a copy, so a running queue isn't affected when its committed file changes.
