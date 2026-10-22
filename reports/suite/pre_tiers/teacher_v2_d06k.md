# Teacher evaluation: teacher_v2_d06k

Generated 2026-10-08 11:09 UTC by `va baselines` at commit `e43dfdd`. Don't edit by hand; rerun the command.

Intervals are 95 % speaker-bootstrap percentile intervals (1000 resamples, whole speakers resampled). Detection rate (DR) counts a frame as correct when its error is within **both** bounds (decisions D-004); missing estimates count as misses. Gross error means more than 30 % off, or missing.

## Formants: Hillenbrand et al. (1995)

1,668 tokens from 139 speakers. Each token has 4 reference frames (steady state; 20/50/80 % of the vowel); F4 exists only at steady state. These are hand-checked measurements, and Hillenbrand is never trained on.

- **praat_default**: Praat Burg, defaults (5 formants, 5500 Hz ceiling)
- **praat_sweep**: Praat Burg, 4000-8000 Hz ceiling sweep, FastTrack-style DCT selection (oracle vowel span)
- **praat_fasttrack**: Praat Burg, FastTrack's per-talker-group ceiling ranges (men 4500-6500, women 5000-7000, children 5500-7500 Hz) and selection error (speaker group + oracle vowel span)
- **tvlp**: TVLP, voice-dataset-labeler @ 0ab5568, 150 ms window
- **teacher**: teacher teacher_v2_d06k step 10000 (sha 6ce6aaa95630), trained on synthetic data only

| Baseline | Formant | n | MAE Hz | MAE % | RMSE Hz | DR (20 %, 200 Hz) % | DR (25 %, 250 Hz) % | DR (30 %, 300 Hz) % | Gross % | Missing % |
|---|---|---|---|---|---|---|---|---|---|---|
| praat_default | F1 | 6672 | 18 [16, 19] | 3.0 | 34 | 98.9 [98.5, 99.3] | 99.3 | 99.6 | 0.4 | 0.0 |
| praat_default | F2 | 6636 | 81 [68, 97] | 4.3 | 232 | 93.3 [91.7, 94.8] | 94.5 | 95.1 | 3.0 | 0.0 |
| praat_default | F3 | 6505 | 109 [94, 125] | 3.7 | 250 | 88.1 [86.2, 89.9] | 90.0 | 91.3 | 1.9 | 0.0 |
| praat_default | F4 | 1425 | 191 [168, 217] | 4.5 | 395 | 79.4 [76.7, 82.0] | 82.2 | 84.8 | 2.5 | 0.0 |
| praat_sweep | F1 | 6672 | 36 [31, 41] | 5.0 | 83 | 93.1 [91.7, 94.6] | 95.0 | 96.9 | 3.0 | 0.0 |
| praat_sweep | F2 | 6636 | 193 [160, 224] | 9.9 | 468 | 82.8 [79.7, 85.8] | 83.5 | 84.0 | 14.5 | 0.0 |
| praat_sweep | F3 | 6505 | 243 [202, 283] | 8.1 | 531 | 80.7 [77.4, 83.9] | 81.5 | 82.2 | 14.0 | 0.0 |
| praat_sweep | F4 | 1425 | 357 [303, 411] | 8.5 | 730 | 73.7 [70.6, 76.8] | 76.1 | 77.5 | 13.5 | 0.0 |
| praat_fasttrack | F1 | 6672 | 17 [16, 18] | 2.9 | 30 | 99.4 [99.1, 99.7] | 99.7 | 99.8 | 0.2 | 0.0 |
| praat_fasttrack | F2 | 6636 | 43 [37, 51] | 2.6 | 119 | 98.1 [97.4, 98.6] | 98.6 | 98.8 | 0.8 | 0.0 |
| praat_fasttrack | F3 | 6505 | 55 [48, 61] | 1.9 | 127 | 95.7 [94.8, 96.7] | 97.1 | 97.7 | 0.4 | 0.0 |
| praat_fasttrack | F4 | 1425 | 116 [104, 129] | 2.8 | 241 | 87.0 [84.9, 89.0] | 89.5 | 91.3 | 0.7 | 0.0 |
| tvlp | F1 | 6672 | 27 [26, 29] | 4.7 | 49 | 98.0 [97.4, 98.5] | 98.6 | 98.9 | 1.0 | 0.0 |
| tvlp | F2 | 6636 | 139 [118, 159] | 7.6 | 368 | 89.6 [88.1, 91.1] | 92.1 | 93.5 | 5.6 | 0.0 |
| tvlp | F3 | 6505 | 130 [116, 144] | 4.7 | 259 | 84.1 [82.0, 86.3] | 87.7 | 89.9 | 2.5 | 0.0 |
| tvlp | F4 | 1425 | 171 [150, 195] | 4.2 | 346 | 80.6 [77.8, 83.3] | 84.8 | 87.0 | 2.1 | 0.0 |
| teacher | F1 | 6672 | 37 [34, 39] | 6.3 | 52 | 96.9 [96.1, 97.6] | 98.5 | 99.1 | 0.9 | 0.0 |
| teacher | F2 | 6636 | 50 [47, 54] | 3.5 | 90 | 97.8 [97.3, 98.4] | 98.9 | 99.3 | 0.4 | 0.0 |
| teacher | F3 | 6505 | 78 [67, 88] | 2.8 | 215 | 95.5 [94.4, 96.6] | 96.3 | 96.7 | 1.5 | 0.0 |
| teacher | F4 | 1425 | 154 [134, 172] | 3.8 | 367 | 87.1 [85.1, 89.1] | 89.3 | 90.2 | 2.4 | 0.0 |

### DR (20 %, 200 Hz) % by speaker group

| Baseline | men F1 | men F2 | men F3 | women F1 | women F2 | women F3 | boys F1 | boys F2 | boys F3 | girls F1 | girls F2 | girls F3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| praat_default | 99.3 | 98.0 | 91.7 | 99.5 | 93.8 | 89.1 | 98.3 | 91.4 | 88.4 | 97.4 | 83.5 | 76.7 |
| praat_sweep | 99.4 | 95.8 | 93.7 | 93.2 | 81.4 | 78.0 | 89.1 | 72.6 | 72.5 | 83.7 | 69.5 | 67.7 |
| praat_fasttrack | 99.8 | 99.0 | 97.4 | 99.7 | 97.8 | 95.3 | 99.0 | 98.0 | 96.3 | 98.1 | 96.6 | 92.0 |
| tvlp | 96.9 | 91.0 | 79.0 | 99.4 | 90.5 | 89.5 | 97.8 | 89.3 | 87.2 | 96.9 | 84.6 | 78.7 |
| teacher | 98.6 | 98.5 | 97.1 | 97.7 | 97.7 | 96.9 | 94.2 | 97.5 | 94.4 | 94.8 | 97.1 | 89.5 |

n frames per column: men 2160, women 2304, boys 1296, girls 912.

### DR (20 %, 200 Hz) % by f0 band (the token's steady-state f0)

| Baseline | <150 F1 | <150 F2 | <150 F3 | 150-300 F1 | 150-300 F2 | 150-300 F3 | 300-500 F1 | 300-500 F2 | 300-500 F3 |
|---|---|---|---|---|---|---|---|---|---|
| praat_default | 99.1 | 97.8 | 91.3 | 98.8 | 91.8 | 87.0 | 100.0 | 85.0 | 80.0 |
| praat_sweep | 99.5 | 95.4 | 94.0 | 90.8 | 78.2 | 75.8 | 100.0 | 90.0 | 87.5 |
| praat_fasttrack | 99.8 | 99.0 | 97.7 | 99.2 | 97.8 | 95.1 | 100.0 | 90.0 | 90.0 |
| tvlp | 97.1 | 91.3 | 80.0 | 98.3 | 89.0 | 85.6 | 100.0 | 92.5 | 82.5 |
| teacher | 99.0 | 98.7 | 97.3 | 96.1 | 97.6 | 94.8 | 100.0 | 90.0 | 90.0 |

n frames per column: <150 1752, 150-300 4880, 300-500 40.

### DR (20 %, 200 Hz) % by frame position

| Baseline | 20/50/80% F1 | 20/50/80% F2 | 20/50/80% F3 | steady F1 | steady F2 | steady F3 |
|---|---|---|---|---|---|---|
| praat_default | 99.1 | 93.3 | 88.1 | 98.4 | 93.4 | 88.1 |
| praat_sweep | 93.2 | 82.6 | 80.7 | 92.8 | 83.2 | 80.5 |
| praat_fasttrack | 99.5 | 98.1 | 95.8 | 99.2 | 98.1 | 95.7 |
| tvlp | 98.0 | 89.0 | 83.9 | 97.9 | 91.6 | 84.9 |
| teacher | 97.1 | 98.0 | 95.5 | 96.4 | 97.4 | 95.5 |

n frames per column: 20/50/80% 5004, steady 1668.

## f0

- **pesto**: PESTO mir-1k_g7 (pesto-pitch 2.0.1), 10 ms hop; voiced if confidence >= 0.5. Trained on: mir1k.
- **crepe**: CREPE full (torchcrepe 0.0.24), Viterbi decoding, 50-1600 Hz, 10 ms hop; voiced if periodicity >= 0.21. Trained on: mir1k, mdb_stem_synth.
- **praat_ac**: Praat autocorrelation (parselmouth), defaults except range 50-1000 Hz, 10 ms grid; unvoiced frames have no estimate. Trained on: none of our gold sets.
- **teacher**: teacher teacher_v2_d06k step 10000 (sha 6ce6aaa95630), trained on synthetic data only; voiced if voicing p >= 0.5. Trained on: none of our gold sets.

RPA (raw pitch accuracy) and RCA (raw chroma accuracy) use a 50-cent tolerance and count every reference-voiced frame, whatever the estimated voicing decision. The voicing thresholds are fixed in advance (not tuned on gold data). **†** marks a baseline whose training data includes that corpus, so its score there is optimistic and not a fair test.

| Corpus | Baseline | Speakers | Voiced frames | RPA % | RCA % | Octave err % | GPE % | Voicing P % | Voicing R % | Voicing F1 % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ptdb_tug | pesto | 20 | 807,780 | 84.5 [81.2, 87.9] | 87.8 [84.8, 90.6] | 3.3 | 8.0 | 95.5 | 60.6 | 74.2 [68.9, 78.4] | 0.046 |
| ptdb_tug | crepe | 20 | 807,780 | 84.9 [81.8, 87.8] | 87.2 [84.1, 89.9] | 2.3 | 5.4 | 79.0 | 91.2 | 84.7 [82.5, 86.6] | 0.305 |
| ptdb_tug | praat_ac | 20 | 807,780 | 83.9 [80.7, 86.8] | 85.5 [82.0, 88.5] | 1.5 | 10.9 | 87.9 | 91.4 | 89.6 [87.8, 91.0] | 0.002 |
| ptdb_tug | teacher | 20 | 807,780 | 87.0 [84.3, 89.3] | 89.0 [86.7, 91.0] | 2.0 | 3.6 | 83.9 | 89.1 | 86.4 [84.6, 88.1] | 0.040 |
| mir1k | pesto † | 19 | 280,400 | 98.3 [97.9, 98.7] | 98.6 [98.2, 98.8] | 0.2 | 0.3 | 98.7 | 96.8 | 97.7 [97.3, 98.2] | 0.046 |
| mir1k | crepe † | 19 | 280,400 | 97.4 [96.8, 97.8] | 97.4 [97.0, 97.8] | 0.1 | 0.3 | 88.7 | 98.8 | 93.5 [91.8, 95.2] | 0.367 |
| mir1k | praat_ac | 19 | 280,400 | 96.9 [96.1, 97.6] | 97.7 [97.3, 98.1] | 0.8 | 2.6 | 97.0 | 98.6 | 97.8 [97.2, 98.2] | 0.001 |
| mir1k | teacher | 19 | 280,400 | 95.0 [94.3, 95.6] | 95.3 [94.7, 95.7] | 0.3 | 0.6 | 97.8 | 93.6 | 95.7 [95.1, 96.1] | 0.040 |
| mdb_stem_synth | pesto | 30 | 1,767,764 | 97.4 [96.7, 98.0] | 97.7 [97.1, 98.2] | 0.3 | 1.0 | 99.8 | 49.7 | 66.3 [57.8, 73.9] | 0.043 |
| mdb_stem_synth | crepe † | 30 | 1,767,764 | 96.7 [95.5, 97.6] | 96.8 [95.7, 97.7] | 0.2 | 1.6 | 97.2 | 98.0 | 97.6 [97.1, 98.0] | 0.274 |
| mdb_stem_synth | praat_ac | 30 | 1,767,764 | 95.5 [94.5, 96.5] | 95.6 [94.6, 96.6] | 0.0 | 3.3 | 98.9 | 96.9 | 97.9 [97.5, 98.3] | 0.001 |
| mdb_stem_synth | teacher | 30 | 1,767,764 | 98.6 [98.2, 99.0] | 98.6 [98.3, 99.0] | 0.0 | 0.2 | 82.0 | 98.5 | 89.5 [81.7, 95.8] | 0.040 |

### RPA % by reference f0 band

| Corpus | Baseline | <150 | 150-300 | 300-500 | >500 |
|---|---|---|---|---|---|
| ptdb_tug | pesto | 75.7 (n=407,685) | 93.5 (n=397,781) | 92.9 (n=2,311) | 0.0 (n=3) |
| ptdb_tug | crepe | 77.2 (n=407,685) | 92.8 (n=397,781) | 90.4 (n=2,311) | 0.0 (n=3) |
| ptdb_tug | praat_ac | 77.8 (n=407,685) | 90.3 (n=397,781) | 91.8 (n=2,311) | 0.0 (n=3) |
| ptdb_tug | teacher | 82.5 (n=407,685) | 91.7 (n=397,781) | 88.3 (n=2,311) | 0.0 (n=3) |
| mir1k | pesto | 97.6 (n=46,175) | 98.2 (n=169,353) | 99.1 (n=63,511) | 98.8 (n=1,361) |
| mir1k | crepe | 96.5 (n=46,175) | 97.1 (n=169,353) | 98.6 (n=63,511) | 97.8 (n=1,361) |
| mir1k | praat_ac | 95.8 (n=46,175) | 96.6 (n=169,353) | 98.4 (n=63,511) | 98.2 (n=1,361) |
| mir1k | teacher | 95.5 (n=46,175) | 94.4 (n=169,353) | 96.3 (n=63,511) | 96.3 (n=1,361) |
| mdb_stem_synth | pesto | 87.8 (n=85,374) | 97.5 (n=1,156,780) | 98.7 (n=469,768) | 98.9 (n=55,842) |
| mdb_stem_synth | crepe | 85.5 (n=85,374) | 97.1 (n=1,156,780) | 97.6 (n=469,768) | 97.4 (n=55,842) |
| mdb_stem_synth | praat_ac | 86.5 (n=85,374) | 95.6 (n=1,156,780) | 96.9 (n=469,768) | 97.2 (n=55,842) |
| mdb_stem_synth | teacher | 93.1 (n=85,374) | 98.8 (n=1,156,780) | 99.1 (n=469,768) | 99.1 (n=55,842) |

### RPA % on ptdb_tug subsets

*Modal* frames are those where the laryngograph shows regular vibration at the reference f0 (decisions D-009): the f0 is well defined there, so this is the fairest accuracy figure. Headline numbers use all frames.

| Baseline | all frames | modal | modal, ref < 85 Hz | female | female, modal | male | male, modal |
|---|---|---|---|---|---|---|---|
| pesto | 84.5 [81.2, 87.9] | 90.5 [87.2, 93.7] | 62.8 [51.7, 64.6] | 87.5 [83.2, 91.4] | 96.0 [94.5, 97.2] | 81.6 [77.5, 85.9] | 85.7 [81.5, 90.1] |
| crepe | 84.9 [81.8, 87.8] | 91.5 [88.9, 93.9] | 72.1 [59.0, 76.0] | 86.4 [82.2, 90.4] | 95.6 [94.0, 96.9] | 83.4 [79.6, 86.9] | 87.9 [84.7, 91.1] |
| praat_ac | 83.9 [80.7, 86.8] | 89.9 [87.3, 92.4] | 73.4 [62.4, 78.7] | 87.3 [83.8, 90.7] | 94.0 [92.1, 95.6] | 80.6 [76.7, 83.8] | 86.4 [83.3, 89.3] |
| teacher | 87.0 [84.3, 89.3] | 93.7 [91.8, 95.2] | 85.5 [75.5, 90.2] | 87.9 [84.4, 91.0] | 95.8 [94.3, 97.1] | 86.2 [82.3, 88.8] | 91.7 [88.9, 93.6] |

Voiced reference frames: all frames 807,780, modal 670,251, modal, ref < 85 Hz 44,796, female 403,619, female, modal 314,202, male 404,161, male, modal 356,049.

### ptdb_tug: where estimates land, by laryngograph phonation category

Each voiced reference frame is categorised from the laryngograph signal alone (`eval/phonation.py`). Cells give % of frames within 50 cents of the reference / of 2 × / of ½ the reference, and % the baseline judged voiced. Outside *modal* the f0 is ambiguous or the reference unreliable, so these are descriptions, not accuracies.

| Category | Share | pesto | crepe | praat_ac | teacher |
|---|---|---|---|---|---|
| **modal**: regular vibration at the reference f0 | 83.0 % | 91 / 1 / 0; voiced 67 | 92 / 0 / 0; voiced 95 | 90 / 0 / 1; voiced 95 | 94 / 0 / 0; voiced 94 |
| **ref_error_2x**: folds cycle at 2 × the reference: the reference halved | 0.5 % | 5 / 89 / 0; voiced 53 | 6 / 88 / 0; voiced 94 | 59 / 31 / 0; voiced 95 | 21 / 71 / 0; voiced 90 |
| **alternating**: alternate cycles differ (period doubling): the reference and 2 × are both defensible | 3.7 % | 59 / 31 / 0; voiced 56 | 54 / 33 / 0; voiced 75 | 87 / 3 / 1; voiced 95 | 79 / 13 / 0; voiced 92 |
| **aperiodic**: irregular phonation (creak): no single f0 | 9.2 % | 61 / 5 / 0; voiced 22 | 60 / 4 / 0; voiced 78 | 51 / 1 / 2; voiced 70 | 54 / 3 / 1; voiced 65 |
| **weak_egg**: too little laryngograph signal; the reference is unreliable | 3.6 % | 45 / 4 / 1; voiced 23 | 40 / 4 / 0; voiced 61 | 32 / 2 / 0; voiced 47 | 37 / 3 / 7; voiced 37 |

## Synthetic hard-case suite (fixed seed, `data/synth/suite_v1`)

DR is within (20 %, 200 Hz) on sonorant frames (voiced, plus whispered in `whisper`); f0 RPA within 50 cents on voiced frames; voicing accuracy over all frames. The channel is clean except in `low_snr`, `phone` and `codec`.

| Case | Frames | F1 DR | F2 DR | F3 DR | F4 DR | F1 MAE Hz | F2 MAE Hz | f0 RPA | Voicing acc |
|---|---|---|---|---|---|---|---|---|---|
| adult_high_f0 | 15,531 | 99.9 | 99.6 | 99.4 | 98.2 | 10 | 15 | 99.8 | 99.7 |
| breathy | 15,334 | 98.3 | 99.4 | 95.6 | 84.2 | 17 | 24 | 99.9 | 99.5 |
| codec | 14,866 | 97.4 | 97.4 | 87.9 | 74.4 | 22 | 36 | 99.6 | 99.1 |
| f1_near_h1 | 15,558 | 39.6 | 93.7 | 90.4 | 78.1 | 320 | 74 | 99.7 | 99.5 |
| f1_near_h2 | 14,345 | 54.5 | 79.9 | 78.8 | 73.9 | 518 | 178 | 99.9 | 99.6 |
| f2_f3_close | 14,211 | 98.2 | 97.3 | 63.3 | 58.1 | 19 | 49 | 99.9 | 99.5 |
| high_f0 | 15,183 | 96.9 | 97.3 | 91.0 | 71.2 | 30 | 41 | 100.0 | 99.5 |
| low_snr | 14,488 | 88.4 | 76.9 | 58.1 | 48.5 | 50 | 191 | 95.9 | 95.1 |
| nasal | 14,837 | 94.5 | 96.0 | 91.0 | 77.6 | 33 | 49 | 99.9 | 99.4 |
| phone | 14,570 | 97.1 | 93.1 | 67.1 | 53.4 | 24 | 74 | 99.3 | 98.5 |
| reference | 14,449 | 98.0 | 98.9 | 95.2 | 86.5 | 17 | 24 | 99.9 | 99.5 |
| tube | 15,104 | 98.9 | 99.8 | 99.9 | 97.3 | 19 | 20 | 99.9 | 99.7 |
| whisper | 15,612 | 99.6 | 99.2 | 96.5 | 89.8 | 13 | 20 | – | 100.0 |
| wide_vibrato | 15,401 | 98.5 | 98.9 | 94.4 | 79.6 | 18 | 27 | 99.6 | 99.5 |

## Sim-to-real gap by f0 band

Formant DR (20 %, 200 Hz) % on the synthetic `reference` case versus Hillenbrand, by f0 band (Hillenbrand uses each token's steady-state f0).

| f0 band | Synthetic F1 / F2 / F3 | Hillenbrand F1 / F2 / F3 | Gap F1 / F2 / F3 (points) |
|---|---|---|---|
| <150 | 100.0 / 99.6 / 99.3 | 99.0 / 98.7 / 97.3 | +1.0 / +0.9 / +2.0 |
| 150-300 | 99.4 / 100.0 / 98.8 | 96.1 / 97.6 / 94.8 | +3.3 / +2.4 / +3.9 |
| 300-500 | 99.6 / 99.6 / 98.6 | 100.0 / 90.0 / 90.0 | -0.4 / +9.6 / +8.6 |
| >500 | 94.1 / 96.8 / 86.2 | – | – |

## Provenance

| | |
|---|---|
| Commit | `e43dfdd` |
| Submodule | `0ab55688d511d18ca9afd52787ddffc871b1d447 external/voice-dataset-labeler (heads/main)` |
| Python | 3.13.15 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.13.0+rocm10.0.0 |
| praat-parselmouth | 0.4.7 |
| pesto-pitch | 2.0.1 |
| torchcrepe | 0.0.24 |
| Manifest `hillenbrand` | `93c302ef20838268` |
| Manifest `mdb_stem_synth` | `f8fcec203d1c79be` |
| Manifest `mir1k` | `3c3856f4161505ce` |
| Manifest `ptdb_tug` | `2403475d28025082` |
| Manifest `vocalset` | `4cea60672f9934ed` |
