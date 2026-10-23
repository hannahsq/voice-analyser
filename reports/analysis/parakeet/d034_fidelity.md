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

| Task | v1 | v2 |
|---|---|---|
| Excerpts (sung words) | 0.34 | **0.40** |
| Scales | 0.32 | 0.32 |
| Arpeggios | 0.25 | 0.28 |
| Long tones | 0.25 | 0.26 |

| Technique | v1 | v2 |
|---|---|---|
| Vibrato | 0.35 | **0.38** |
| Fast forte / fast piano | 0.33 / 0.34 | 0.35 / 0.34 |
| Slow forte / slow piano | 0.31 / 0.32 | 0.31 / 0.31 |
| Forte | 0.26 | 0.31 |
| Trill / trillo | 0.27 / 0.27 | 0.31 / 0.29 |
| Belt | 0.27 | 0.24 |
| Breathy | 0.22 | 0.25 |
| Straight | 0.24 | 0.25 |
| Pianissimo | 0.21 | 0.24 |
| Vocal fry | 0.22 | 0.18 |
| Messa di voce | 0.03 | −0.03 |
| Lip trill | −0.13 | −0.03 |

Sustained and non-speech-like phonation is hardest (messa di voce, lip trill, vocal fry, long tones). Sung words help. Singing is VocalSet only (6 h), the smallest real pool.

## Next

1. **v2 finishes** around 02:00 on 2026-10-24, with fidelity at steps 4k, 6k, 8k, 10k and 12k (added here).
2. **Head stage, as pre-registered:**
   - six H5-recipe heads on the frozen v2 student, and six on control R (the same architecture, untrained);
   - D-033's part-2 rule on six-seed means: women's closed-phase F1 bias cut ≥ 5 Hz, men's not raised by > 5 Hz, Hillenbrand F1 MAE within +4 Hz.
   - "transfers" if the student passes and control R fails.
3. Before that, a one-seed head on v1's step-2000 checkpoint, as a pipeline check (no decision).
