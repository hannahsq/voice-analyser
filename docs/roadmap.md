# Roadmap

What is planned and in what order. Reasons and results go in `decisions.md`; this file only tracks what is next. Updated 2026-10-11.

## Now

*Handover, 2026-10-11 evening. Results and reasoning are in D-023 and D-024 (decisions.md); this section is the state of play.*

**In flight**
- `v2c` chain (`scripts/analysis/chains/v2c_chain.sh`, launched from the scratch copy; log lines "v2c: ..." in `runs/orchestrator.log`). `data/synth/train_v2c` is generated (train_v2's settings and seed 2, plus glottal coupling on every tube clip). Three S2 mixed-crop Conformer seeds (`runs/small/v2c_conformer_s0-2`) train against `mix_conformer_s0-2`; each is measured in fp32 (paired vs v2_s0 and vs the same-seed mixed Conformer in `reports/analysis/paired/fp32/`, closed phase on the GPU), then the invariance probe and `closed_phase/cp_small_v2c.txt`. Done when "v2c: done" appears (about 22:30).
- Saarbrücken download (`data/downloads/saarbruecken/`: healthy, hypofunctional dysphonia, recurrent laryngeal nerve paresis, functional dysphonia, phonasthenia; CC BY 4.0). The files are being saved with a trailing carriage return in their names (the URL list had CRLF endings): rename them once complete.

**Next, in order**
1. *Read v2c against D-024's pre-registered rule, on dev speakers* (`vocal_analysis/eval/tiers.py`): women's and men's closed-phase F1 bias (the frames carry PTDB-TUG speaker ids; ARCTIC is dev), the correlation of F1 error with the EGG contact quotient, Hillenbrand F1 (dev speakers from the cached predictions, which carry speaker ids) and the probe. Thresholds on the women's bias change: 15 Hz or more, coupling is a large part of the problem; 5–15 Hz, three more seeds per arm; under 5 Hz, not the main cause. Then confirm on the semi-private speakers (log the use in D-024). Rerun `ensemble.py`; record the result in D-024.
2. *Switch the measure scripts to dev by default*: `m2_paired_sub.py` reads `runs/manifests_sub_dev` (made by `va tiers --dev-copy`), `cp_cmp.py`, `ensemble.py`, `voice_quality.py` filter frames by tier; an explicit option for semi-private, logged. Register OPENGLOT, Saarbrücken and MOCHA-TIMIT as evaluation corpora for `va leakage`; write the Saarbrücken manifest and its 50 / 50 semi-private / lockbox split (rule in D-024).
3. *If coupling helps*: full-size Conformer and GRU on train_v2c with seeds against the full-size v2 seeds (dev tier), then M3 (below). *If not*: record the M2 shortfall and move to M3 anyway (decided 2026-10-11).
4. *Disk*: done 2026-10-11 (192 GB free). train_v1 and train_v3a deleted, train_v2 and train_v3b kept as `flat/` only; all regenerate bit-identically (`docs/synth_sets/`). Delete train_v2c's shards (about 17 GB) once the v2c chain is done.

**Settled 2026-10-11** (details in D-024): S2 is no proxy for small real-voice effects; EMA off; mixed crop lengths (8 geometric lengths 32–400, several crops per clip) fix the Conformer's short-input failure, the long-weighted mix doesn't help and is dropped; measures run in fp32; four lines of evidence (coupling probes, real-voice voice quality, OPENGLOT II with coupling, OPENGLOT III without) point to source–tract coupling through the glottal gap as the cause of the F1 overread every model shares; evaluation tiers in place; coupled synthesis costs 0.24 CPU s per s of audio (uncoupled 0.17).

**Speed** (done, parked, open): AMR streams through one long-lived ffmpeg per bitrate per loader worker (loader 829–1036 s/s); 4 workers for S2; validation cached on the GPU; closed phase on the GPU (`add_model.py`, `VA_DEVICE=cuda`, identical to CPU); `torch.compile` and HIP graph capture parked (no Triton on ROCm under Windows; capture aborts in hipBLASLt); priority opt-in (`VA_PRIORITY`), no measurable gain; under CPU load S2 loses 15–36 %, so keep measures off the CPU during training. Open: a batched GPU eval suite for the paired measures and the probe (closed phase done); profiling the training step needs ablation timing, since the ROCm profiler's GPU times are unreliable here. Machine notes: CPU boost on; CPPC core order 7, 8, 5, 4, 1, 6, 3, 2, then 11, 12, 13, 15, 10, 16, 9, 14 (HWiNFO numbering from 1); the main training thread runs on cores 6–7 (Windows numbering) and GPU use follows it.

**Then**: M3 (`docs/voice-analyser-mk2-plan.md` §12). Only its first item (real-audio training pools, manifests, leakage) is a prerequisite for most of M4; M4's candidate tracker feeds M3's labeller. Resynthesis should add coupling (it otherwise reinforces the closed-glottis picture), and M4's reconstruction loss would train the overread in unless coupling-aware.

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

- D-021: a per-formant observability label, with independent per-formant outputs (the chained heads let a wrong F1 pull F2 with it); vocal-tract-length outputs. With observability, add voiced clips with little or no formant evidence (a few per cent): unfiltered glottal sources, formants too wide to see, F1 between sparse harmonics at high f0, a single missing formant. Their label then says 'unobservable' instead of forcing a guess (given a bare source by mistake on 2026-10-11, the teachers and Praat read formants 100–500 Hz off). Define observability physically (how far the resonance lifts the harmonics above the source envelope) so synthesis can label it per frame. Mixed crop lengths: done (D-024).
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
