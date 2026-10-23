# D-034 stage 1: student fidelity to Parakeet layer 5

*Updated 2026-10-24, 03:40: student v2 finished (step 12,000, 03:22). Raw outputs for every checkpoint are in `fid_v2_*.txt`; the head stage (the bias test) is running.*

**What this measures.** How closely the student (32 M parameters, our encoder architecture at 10 ms) reproduces Parakeet's layer-5 features on audio it was not trained on. R² is averaged over the 1,024 dimensions, on per-dimension standardised targets. Variance-weighted R² and mean cosine track it closely (see `fid_breakdown.py`).

This is not the bias test. The pre-registered decision comes from six heads trained on the frozen student (and on control R), scored on the real-voice gold sets.

## The two runs

| | Real audio in training | Synthetic |
|---|---|---|
| **v1** (stopped at step 2,800) | 30k crops: Korean (Zeroth, Common Voice) and VocalSet singing | 30k |
| **v2** (finished) | 45k crops: English adults (LibriTTS-R), English children and adults (speechocean762), Icelandic children (Samrómur), Korean, VocalSet, Italian / Portuguese / Polish (MLS) | 45k |

v1 was stopped because, at step 2000, it copied Parakeet well only on the kinds of audio it had heard. That made the blended restart a decision on fidelity alone, taken before any bias result (D-034 amendment).

## Fidelity by population

| Held-out population | v1, step 2000 | v2, step 2000 | v2, step 4000 | v2, step 6000 | **v2, step 12000** |
|---|---|---|---|---|---|
| English adults (LibriTTS-R dev-clean) | 0.21 | 0.48 | 0.56 | 0.59 | **0.63** |
| English children ≤ 12 (speechocean762 test) | 0.23 | 0.38 | 0.46 | 0.51 | **0.56** |
| English adults ≥ 19 (speechocean762 test) | 0.24 | 0.37 | 0.45 | 0.50 | **0.55** |
| Icelandic children ≤ 12 (Samrómur test) | 0.24 | 0.35 | 0.43 | 0.48 | **0.53** |
| Icelandic teens 13–17 (Samrómur test) | 0.23 | 0.33 | 0.41 | 0.46 | **0.51** |
| Italian (MLS dev) | 0.28 | 0.37 | 0.45 | 0.50 | **0.54** |
| Portuguese (MLS dev) | 0.26 | 0.37 | 0.44 | 0.49 | **0.54** |
| Polish (MLS dev) | 0.26 | 0.35 | 0.43 | 0.48 | **0.53** |
| Korean (Zeroth val) | 0.42 | 0.39 | 0.48 | 0.52 | **0.57** |
| Singing (VocalSet val) | 0.30 | 0.31 | 0.36 | 0.41 | **0.47** |
| Synthetic (train_v2 validation) | 0.47 | 0.50 | 0.54 | 0.59 | **0.64** |
| Synthetic (train_v2s validation) | 0.42 | 0.45 | 0.50 | 0.55 | **0.61** |

- **Step 12,000 (final)**: every speech population is within 0.51–0.63, singing 0.47, synthetic 0.61–0.64. Gains slowed only in the last 1–2k steps, as the learning rate reached zero, so a longer run would probably still help (the case for stage 1b).
- **Every set climbed at every checkpoint.** Step 4000 added +0.07 to +0.09 on every speech population and +0.05 on singing; step 6000 another +0.03 to +0.05 on all of them. Korean (0.52) is well above v1's 0.42, so the dip from its smaller share is gone.
- **At step 2000, the coverage gap had already closed.** Every speech population was within 0.33–0.48, against v1's 0.21–0.42, and English adults went from worst to best. Korean dipped (0.42 → 0.39) as its share of the real crops fell from 75% to 22%.
- **Synthetic keeps improving** (0.50 → 0.54 on train_v2) at the same synthetic share. This is where the formant heads train.
- **Children and teens are still the hardest speech** (0.51–0.56 at step 12,000). They're still a small share of the mix.
- *Caveat:* v2's held-out speech sets are new speakers from corpora it trains on, so part of the gain is coverage of those corpora. The real-voice gold sets (Hillenbrand, PTDB-TUG) are separate corpora, tested only through the heads.

## Singing by VocalSet task and technique (small groups)

R² per dimension. The v1 → v2 comparison is at step 2000. **Bold:** the paired bootstrap's 95% interval for v2 − v1 excludes 0 on all three metrics (R² per dimension, variance-weighted R², mean cosine). *Italic:* only cosine's does. The bootstrap uses 2,000 resamples of crops, on the same 512 VocalSet val crops for both models (`fid_paired.py`). The last column is v2 at step 4000 (point estimates).

| Task | n | v1 | v2 | v2 − v1, R² [95%] | v2, step 4000 | v2, step 6000 | v2, step 12000 |
|---|---|---|---|---|---|---|---|
| Excerpts (sung words) | 26 | 0.34 | **0.40** | +0.06 [+0.04, +0.08] | 0.46 | 0.51 | 0.56 |
| Arpeggios | 136 | 0.25 | *0.28* | +0.03 [−0.01, +0.07] | 0.33 | 0.38 | 0.44 |
| Scales | 220 | 0.32 | 0.32 | [−0.03, +0.02] | 0.38 | 0.41 | 0.49 |
| Long tones | 130 | 0.25 | 0.25 | [−0.01, +0.02] | 0.30 | 0.35 | 0.40 |

| Technique | n | v1 | v2 | v2 − v1, R² [95%] | v2, step 4000 | v2, step 6000 | v2, step 12000 |
|---|---|---|---|---|---|---|---|
| Forte | 24 | 0.26 | **0.31** | +0.05 [+0.02, +0.09] | 0.36 | 0.40 | 0.45 |
| Trill | 20 | 0.27 | **0.31** | +0.04 [+0.00, +0.08] | 0.35 | 0.37 | 0.42 |
| Vibrato | 35 | 0.35 | *0.38* | +0.03 [−0.00, +0.07] | 0.45 | 0.48 | 0.54 |
| Fast forte | 27 | 0.33 | *0.35* | +0.02 [−0.00, +0.06] | 0.39 | 0.48 | 0.54 |
| Breathy | 35 | 0.22 | *0.25* | +0.03 [−0.02, +0.07] | 0.31 | 0.35 | 0.43 |
| Pianissimo | 18 | 0.21 | *0.24* | +0.03 [−0.01, +0.07] | 0.30 | 0.32 | 0.39 |
| Trillo | 21 | 0.27 | *0.29* | +0.02 [−0.01, +0.04] | 0.33 | 0.39 | 0.43 |
| Fast piano | 35 | 0.34 | 0.34 | [−0.04, +0.04] | 0.39 | 0.44 | 0.50 |
| Slow forte / slow piano | 55 / 53 | 0.31 / 0.32 | 0.31 / 0.31 | both span 0 | 0.38 / 0.36 | 0.42 / 0.37 | 0.49 / 0.48 |
| Straight | 61 | 0.24 | 0.25 | [−0.04, +0.05] | 0.30 | 0.36 | 0.43 |
| Belt | 27 | 0.27 | 0.24 | −0.03 [−0.12, +0.04] | 0.33 | 0.39 | 0.46 |
| Vocal fry | 31 | 0.22 | 0.18 | −0.04 [−0.09, +0.02] | 0.25 | 0.30 | 0.34 |
| Messa di voce | 17 | 0.03 | −0.03 | −0.06 [−0.14, +0.00]; variance-weighted R² down (interval excludes 0) | 0.05 | 0.16 | 0.26 |
| Lip trill | 34 | −0.13 | −0.03 | +0.10 [−0.01, +0.25] | −0.00 | 0.06 | 0.13 |

- **Clear gains:** sung words (excerpts), forte and trill.
- **Cosine-only gains** (direction better, R² within noise): vibrato, fast forte, breathy, pianissimo, trillo, arpeggios. Over all VocalSet, cosine rises 0.64 → 0.65 (interval excludes 0) while R² 0.30 → 0.31 doesn't.
- **By step 4000 every group is above both v1 and v2's step 2000**, including belt (0.24 → 0.33) and vocal fry (0.18 → 0.25), which had dipped. Messa di voce and lip trill are still near 0 at step 4000. By step 6000 messa reaches 0.16; lip trill swings between checkpoints (−0.14 at 5k, +0.06 at 6k), so its numbers are noise around a hard case.
- **At step 2000, no group was worse on all three metrics.** Belt and vocal fry fell in point estimate only. Messa di voce's variance-weighted R² fell, but it has 17 crops.
- *Caveats:* crops are resampled independently, but VocalSet val has few singers and several crops per recording, so these intervals are optimistic. With 19 groups tested, about one false positive is expected at 5%.

Sustained and non-speech-like phonation is hardest (messa di voce, lip trill, vocal fry, long tones). Sung words help. Singing is VocalSet only (6 h), the smallest real pool.

## Stage-1 result: does not transfer (2026-10-24)

Six heads each on student v2 (step 12,000) and control R, with H5×6 scored in the same run (`d034_heads.txt`). Six-seed means, ensemble (c) with teacher v3:

| | Women's F1 bias cut | Men's F1 bias rise | Hillenbrand F1 MAE cut | Passes? |
|---|---|---|---|---|
| Student v2 | −8.9 ± 4.9 | +5.4 ± 1.1 | −2.7 ± 3.4 | no |
| Control R | −0.9 ± 6.2 | −3.9 ± 2.7 | −16.7 ± 2.5 | no |
| H5 (Parakeet itself) | **+5.3 ± 1.1** | −2.8 ± 2.1 | **+4.1 ± 1.3** | yes |

- **The pipeline works.** On synthetic validation the student's heads match H5's (F1 MAE 19.8 vs 20.2 Hz, f0 < 150 Hz), and on the gold audio the student lines up with Parakeet: R² peaks at zero shift (`d034_gold_fidelity.txt`).
- **It fails on real voices.** The student's heads overread women's F1 on PTDB-TUG (+14.6 Hz against H5's −11.0), like our own teacher, and scatter more on Hillenbrand (F1 MAE 54 vs 43).
- **Fidelity on the gold audio**: PTDB-TUG 0.49, like held-out speech; Hillenbrand only 0.10. Its isolated words of under half a second, with silence around them, look nothing like the 4 s crops of running speech the student trained on.
- Stage 1b didn't run (it needed "transfers"); D-031 resumed.

## Next (before the result)

1. **v2 finished** at 03:22 on 2026-10-24.
2. **Head stage, as pre-registered:**
   - six H5-recipe heads on the frozen v2 student, and six on control R (the same architecture, untrained);
   - D-033's part-2 rule on six-seed means: women's closed-phase F1 bias cut ≥ 5 Hz, men's not raised by > 5 Hz, Hillenbrand F1 MAE within +4 Hz.
   - "transfers" if the student passes and control R fails.
3. Before that, a one-seed head on v1's step-2000 checkpoint, as a pipeline check (no decision).
