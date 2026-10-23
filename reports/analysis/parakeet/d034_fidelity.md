# D-034 stage 1: student fidelity to Parakeet layer 5

*Updated 2026-10-23 evening, after student v2's step 2000. Training runs to step 12,000, finishing in the early hours of 2026-10-24; later checkpoints are added here.*

**What this measures.** How closely the student (32 M parameters, our encoder architecture at 10 ms) reproduces Parakeet's layer-5 features on audio it was not trained on. R² is averaged over the 1,024 dimensions, on per-dimension standardised targets. Variance-weighted R² and mean cosine track it closely (see `fid_breakdown.py`).

This is not the bias test. The pre-registered decision comes from six heads trained on the frozen student (and on control R), scored on the real-voice gold sets.

## The two runs

| | Real audio in training | Synthetic |
|---|---|---|
| **v1** (stopped at step 2,800) | 30k crops: Korean (Zeroth, Common Voice) and VocalSet singing | 30k |
| **v2** (running) | 45k crops: English adults (LibriTTS-R), English children and adults (speechocean762), Icelandic children (Samrómur), Korean, VocalSet, Italian / Portuguese / Polish (MLS) | 45k |

v1 was stopped because, at step 2000, it copied Parakeet well only on the kinds of audio it had heard. That made the blended restart a decision on fidelity alone, taken before any bias result (D-034 amendment).

## Fidelity by population (step 2000 for both)

| Held-out population | v1 | **v2** |
|---|---|---|
| English adults (LibriTTS-R dev-clean) | 0.21 | **0.48** |
| English children ≤ 12 (speechocean762 test) | 0.23 | **0.38** |
| English adults ≥ 19 (speechocean762 test) | 0.24 | **0.37** |
| Icelandic children ≤ 12 (Samrómur test) | 0.24 | **0.35** |
| Icelandic teens 13–17 (Samrómur test) | 0.23 | **0.33** |
| Italian (MLS dev) | 0.28 | **0.37** |
| Portuguese (MLS dev) | 0.26 | **0.37** |
| Polish (MLS dev) | 0.26 | **0.35** |
| Korean (Zeroth val) | **0.42** | 0.39 |
| Singing (VocalSet val) | 0.30 | 0.31 |
| Synthetic (train_v2 validation) | 0.47 | **0.50** |
| Synthetic (train_v2s validation) | 0.42 | **0.45** |

- **The coverage gap has closed.** Every speech population is now within 0.33–0.48, against v1's 0.21–0.42. English adults went from worst to best.
- **Korean dropped a little** as its share of the real crops fell from 75% to 22%.
- **Synthetic improved slightly** at the same synthetic share. This is where the formant heads train.
- **Children and teens are the hardest speech** (0.33–0.38). They're still a small share of the mix.
- *Caveat:* v2's held-out speech sets are new speakers from corpora it trains on, so part of the gain is coverage of those corpora. The real-voice gold sets (Hillenbrand, PTDB-TUG) are separate corpora, tested only through the heads.

## Singing by VocalSet task and technique (small groups; v1 → v2)

R² per dimension at step 2000. **Bold:** the paired bootstrap's 95% interval for v2 − v1 excludes 0 on all three metrics (R² per dimension, variance-weighted R², mean cosine). *Italic:* only cosine's does. The bootstrap uses 2,000 resamples of crops, on the same 512 VocalSet val crops for both models (`fid_paired.py`).

| Task | n | v1 | v2 | v2 − v1, R² [95%] |
|---|---|---|---|---|
| Excerpts (sung words) | 26 | 0.34 | **0.40** | +0.06 [+0.04, +0.08] |
| Arpeggios | 136 | 0.25 | *0.28* | +0.03 [−0.01, +0.07] |
| Scales | 220 | 0.32 | 0.32 | [−0.03, +0.02] |
| Long tones | 130 | 0.25 | 0.25 | [−0.01, +0.02] |

| Technique | n | v1 | v2 | v2 − v1, R² [95%] |
|---|---|---|---|---|
| Forte | 24 | 0.26 | **0.31** | +0.05 [+0.02, +0.09] |
| Trill | 20 | 0.27 | **0.31** | +0.04 [+0.00, +0.08] |
| Vibrato | 35 | 0.35 | *0.38* | +0.03 [−0.00, +0.07] |
| Fast forte | 27 | 0.33 | *0.35* | +0.02 [−0.00, +0.06] |
| Breathy | 35 | 0.22 | *0.25* | +0.03 [−0.02, +0.07] |
| Pianissimo | 18 | 0.21 | *0.24* | +0.03 [−0.01, +0.07] |
| Trillo | 21 | 0.27 | *0.29* | +0.02 [−0.01, +0.04] |
| Fast piano | 35 | 0.34 | 0.34 | [−0.04, +0.04] |
| Slow forte / slow piano | 55 / 53 | 0.31 / 0.32 | 0.31 / 0.31 | both span 0 |
| Straight | 61 | 0.24 | 0.25 | [−0.04, +0.05] |
| Belt | 27 | 0.27 | 0.24 | −0.03 [−0.12, +0.04] |
| Vocal fry | 31 | 0.22 | 0.18 | −0.04 [−0.09, +0.02] |
| Messa di voce | 17 | 0.03 | −0.03 | −0.06 [−0.14, +0.00]; variance-weighted R² down (interval excludes 0) |
| Lip trill | 34 | −0.13 | −0.03 | +0.10 [−0.01, +0.25] |

- **Clear gains:** sung words (excerpts), forte and trill.
- **Cosine-only gains** (direction better, R² within noise): vibrato, fast forte, breathy, pianissimo, trillo, arpeggios. Over all VocalSet, cosine rises 0.64 → 0.65 (interval excludes 0) while R² 0.30 → 0.31 doesn't.
- **No group is worse on all three metrics.** Belt and vocal fry fall in point estimate only. Messa di voce's variance-weighted R² falls, but it has 17 crops.
- *Caveats:* crops are resampled independently, but VocalSet val has few singers and several crops per recording, so these intervals are optimistic. With 19 groups tested, about one false positive is expected at 5%.

Sustained and non-speech-like phonation is hardest (messa di voce, lip trill, vocal fry, long tones). Sung words help. Singing is VocalSet only (6 h), the smallest real pool.

## Next

1. **v2 finishes** around 02:00 on 2026-10-24, with fidelity at steps 4k, 6k, 8k, 10k and 12k (added here).
2. **Head stage, as pre-registered:**
   - six H5-recipe heads on the frozen v2 student, and six on control R (the same architecture, untrained);
   - D-033's part-2 rule on six-seed means: women's closed-phase F1 bias cut ≥ 5 Hz, men's not raised by > 5 Hz, Hillenbrand F1 MAE within +4 Hz.
   - "transfers" if the student passes and control R fails.
3. Before that, a one-seed head on v1's step-2000 checkpoint, as a pipeline check (no decision).
