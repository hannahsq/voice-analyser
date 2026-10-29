# Teacher evaluation: d040_5b_s2_b20k

Generated 2026-10-29 09:32 UTC by `va baselines` at commit `b1db211`. Don't edit by hand; rerun the command.

Intervals are 95 % speaker-bootstrap percentile intervals (1000 resamples, whole speakers resampled). Detection rate (DR) counts a frame as correct when its error is within **both** bounds (decisions D-004); missing estimates count as misses. Gross error means more than 30 % off, or missing.

Speakers: tiers dev (D-024).

## Formants: Hillenbrand et al. (1995)

996 tokens from 83 speakers. Each token has 4 reference frames (steady state; 20/50/80 % of the vowel); F4 exists only at steady state. These are hand-checked measurements, and Hillenbrand is never trained on.

- **praat_default**: Praat Burg, defaults (5 formants, 5500 Hz ceiling)
- **praat_sweep**: Praat Burg, 4000-8000 Hz ceiling sweep, FastTrack-style DCT selection (oracle vowel span)
- **praat_fasttrack**: Praat Burg, FastTrack's per-talker-group ceiling ranges (men 4500-6500, women 5000-7000, children 5500-7500 Hz) and selection error (speaker group + oracle vowel span)
- **tvlp**: TVLP, voice-dataset-labeler @ 0ab5568, 150 ms window
- **qcp_fb_published**: QCP-FB as published (Gowda et al. 2017): duration quotient 0.7, 8 kHz, SEDREAMS GCIs (D-046 arm A; not adopted)
- **qcp_fb**: QCP-FB, duration quotient 0.3, 10 kHz, SEDREAMS GCIs (D-046 arm B; narrowly failed its gate, not adopted; report-only)
- **fasttrack_qcp_snap**: FastTrack (as praat_fasttrack), each formant snapped to the nearest QCP-FB (arm B) peak within 15 %
- **fasttrack_smooth_qcp_snap**: FastTrack smoothed (median 5 frames, then a zero-phase leaky integrator, alpha 0.15, log Hz), snapped as fasttrack_qcp_snap
- **fasttrack_dct_qcp_snap**: FastTrack's own DCT fit (intercept + 5 cosines) as the seed, snapped as fasttrack_qcp_snap
- **teacher**: teacher d040_5b_s2_b20k step 20000 (sha fa176ad17f13-fp32-w400), trained on synthetic data only

| Baseline | Formant | n | MAE Hz | MAE % | RMSE Hz | DR (20 %, 200 Hz) % | DR (25 %, 250 Hz) % | DR (30 %, 300 Hz) % | Gross % | Missing % |
|---|---|---|---|---|---|---|---|---|---|---|
| praat_default | F1 | 3984 | 18 [16, 20] | 3.0 | 35 | 98.9 [98.4, 99.4] | 99.3 | 99.6 | 0.4 | 0.0 |
| praat_default | F2 | 3962 | 84 [65, 103] | 4.4 | 241 | 93.1 [91.0, 95.1] | 94.3 | 94.9 | 3.1 | 0.0 |
| praat_default | F3 | 3863 | 106 [86, 125] | 3.5 | 243 | 88.5 [86.0, 91.1] | 90.3 | 91.6 | 1.7 | 0.0 |
| praat_default | F4 | 841 | 202 [169, 238] | 4.8 | 421 | 78.7 [75.6, 82.0] | 81.8 | 84.5 | 2.6 | 0.0 |
| praat_sweep | F1 | 3984 | 37 [31, 44] | 5.2 | 86 | 92.9 [90.8, 94.8] | 94.6 | 96.5 | 3.4 | 0.0 |
| praat_sweep | F2 | 3962 | 199 [158, 242] | 10.3 | 472 | 81.9 [77.7, 85.7] | 82.6 | 83.2 | 15.2 | 0.0 |
| praat_sweep | F3 | 3863 | 257 [204, 312] | 8.5 | 554 | 80.2 [75.8, 84.2] | 80.8 | 81.4 | 15.2 | 0.0 |
| praat_sweep | F4 | 841 | 368 [303, 432] | 8.8 | 751 | 73.0 [69.0, 76.6] | 75.4 | 76.2 | 13.3 | 0.0 |
| praat_fasttrack | F1 | 3984 | 17 [16, 19] | 2.9 | 31 | 99.4 [99.1, 99.7] | 99.8 | 99.8 | 0.2 | 0.0 |
| praat_fasttrack | F2 | 3962 | 45 [37, 55] | 2.6 | 123 | 97.6 [96.7, 98.4] | 98.2 | 98.5 | 0.9 | 0.0 |
| praat_fasttrack | F3 | 3863 | 53 [46, 62] | 1.8 | 129 | 96.1 [94.9, 97.3] | 97.4 | 97.9 | 0.4 | 0.0 |
| praat_fasttrack | F4 | 841 | 125 [109, 144] | 3.0 | 261 | 86.3 [83.4, 89.0] | 88.7 | 90.4 | 0.8 | 0.0 |
| tvlp | F1 | 3984 | 28 [26, 31] | 4.8 | 52 | 97.5 [96.7, 98.3] | 98.2 | 98.6 | 1.4 | 0.0 |
| tvlp | F2 | 3962 | 144 [118, 174] | 7.9 | 379 | 89.0 [86.8, 90.9] | 91.7 | 93.0 | 5.8 | 0.0 |
| tvlp | F3 | 3863 | 129 [113, 144] | 4.7 | 252 | 84.1 [81.3, 86.7] | 87.3 | 89.9 | 2.3 | 0.0 |
| tvlp | F4 | 841 | 182 [157, 211] | 4.5 | 367 | 80.0 [76.0, 83.4] | 83.5 | 86.1 | 2.1 | 0.0 |
| qcp_fb_published | F1 | 3984 | 79 [69, 90] | 10.8 | 137 | 78.9 [75.5, 82.5] | 84.6 | 89.8 | 10.2 | 0.0 |
| qcp_fb_published | F2 | 3962 | 708 [632, 779] | 35.6 | 1003 | 37.9 [32.5, 43.2] | 39.7 | 41.2 | 56.9 | 0.0 |
| qcp_fb_published | F3 | 3863 | 769 [702, 833] | 26.2 | 993 | 32.7 [27.8, 37.7] | 33.1 | 33.7 | 45.1 | 0.0 |
| qcp_fb_published | F4 | 841 | 1004 [899, 1105] | 24.0 | 1217 | 20.3 [14.8, 25.9] | 20.9 | 21.3 | 42.9 | 12.6 |
| qcp_fb | F1 | 3984 | 72 [63, 83] | 9.9 | 137 | 82.7 [79.5, 85.7] | 85.2 | 88.2 | 11.8 | 0.0 |
| qcp_fb | F2 | 3962 | 444 [372, 517] | 21.8 | 758 | 58.7 [52.8, 64.7] | 59.6 | 60.5 | 33.9 | 0.0 |
| qcp_fb | F3 | 3863 | 557 [479, 634] | 18.5 | 822 | 49.4 [43.0, 56.0] | 50.5 | 51.6 | 32.2 | 0.0 |
| qcp_fb | F4 | 841 | 812 [698, 929] | 19.0 | 1139 | 38.5 [33.1, 44.1] | 41.3 | 43.0 | 28.9 | 1.0 |
| fasttrack_qcp_snap | F1 | 3984 | 30 [27, 33] | 4.9 | 45 | 98.8 [98.3, 99.2] | 99.4 | 99.7 | 0.3 | 0.0 |
| fasttrack_qcp_snap | F2 | 3962 | 58 [50, 68] | 3.4 | 130 | 97.1 [95.9, 98.2] | 98.1 | 98.4 | 0.9 | 0.0 |
| fasttrack_qcp_snap | F3 | 3863 | 70 [63, 79] | 2.4 | 143 | 94.8 [93.5, 95.9] | 96.4 | 97.3 | 0.5 | 0.0 |
| fasttrack_qcp_snap | F4 | 841 | 139 [120, 160] | 3.3 | 279 | 83.6 [80.0, 86.9] | 88.3 | 91.2 | 1.0 | 0.0 |
| fasttrack_smooth_qcp_snap | F1 | 3984 | 30 [27, 32] | 4.8 | 45 | 98.8 [98.3, 99.3] | 99.4 | 99.7 | 0.3 | 0.0 |
| fasttrack_smooth_qcp_snap | F2 | 3962 | 58 [49, 67] | 3.4 | 129 | 97.1 [95.8, 98.2] | 98.1 | 98.4 | 0.8 | 0.0 |
| fasttrack_smooth_qcp_snap | F3 | 3863 | 70 [63, 78] | 2.4 | 142 | 94.8 [93.6, 96.0] | 96.5 | 97.3 | 0.5 | 0.0 |
| fasttrack_smooth_qcp_snap | F4 | 841 | 136 [117, 157] | 3.2 | 274 | 83.9 [80.4, 87.4] | 88.6 | 91.6 | 0.8 | 0.0 |
| fasttrack_dct_qcp_snap | F1 | 3984 | 30 [27, 32] | 4.8 | 45 | 98.9 [98.4, 99.3] | 99.4 | 99.7 | 0.3 | 0.0 |
| fasttrack_dct_qcp_snap | F2 | 3962 | 58 [49, 67] | 3.4 | 130 | 97.1 [95.9, 98.2] | 98.1 | 98.4 | 0.8 | 0.0 |
| fasttrack_dct_qcp_snap | F3 | 3863 | 71 [63, 79] | 2.4 | 144 | 94.7 [93.5, 95.9] | 96.3 | 97.2 | 0.5 | 0.0 |
| fasttrack_dct_qcp_snap | F4 | 841 | 138 [119, 161] | 3.3 | 277 | 83.1 [79.6, 86.5] | 88.1 | 91.1 | 0.8 | 0.0 |
| teacher | F1 | 3984 | 39 [35, 43] | 6.6 | 55 | 96.7 [95.5, 97.7] | 98.6 | 99.2 | 0.8 | 0.0 |
| teacher | F2 | 3962 | 44 [40, 47] | 2.9 | 64 | 98.3 [97.4, 99.0] | 99.1 | 99.5 | 0.2 | 0.0 |
| teacher | F3 | 3863 | 45 [41, 49] | 1.6 | 82 | 98.3 [97.6, 98.9] | 99.0 | 99.3 | 0.1 | 0.0 |
| teacher | F4 | 841 | 97 [82, 116] | 2.3 | 220 | 91.7 [89.3, 93.7] | 93.9 | 94.9 | 0.2 | 0.0 |

### DR (20 %, 200 Hz) % by speaker group

| Baseline | men F1 | men F2 | men F3 | women F1 | women F2 | women F3 | boys F1 | boys F2 | boys F3 | girls F1 | girls F2 | girls F3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| praat_default | 99.2 | 97.9 | 92.4 | 99.5 | 94.3 | 90.1 | 98.8 | 91.8 | 89.7 | 96.6 | 79.6 | 72.8 |
| praat_sweep | 99.8 | 97.0 | 94.8 | 93.1 | 80.5 | 78.1 | 88.9 | 71.6 | 72.0 | 81.1 | 62.9 | 61.0 |
| praat_fasttrack | 100.0 | 99.2 | 97.4 | 99.6 | 97.2 | 96.0 | 99.5 | 97.5 | 96.9 | 97.5 | 94.9 | 91.9 |
| tvlp | 96.3 | 91.1 | 79.5 | 99.3 | 89.9 | 89.9 | 97.9 | 90.9 | 88.4 | 95.5 | 78.7 | 73.6 |
| qcp_fb_published | 98.6 | 65.3 | 58.1 | 76.0 | 33.0 | 27.5 | 66.3 | 15.3 | 12.5 | 56.8 | 15.6 | 12.6 |
| qcp_fb | 98.4 | 89.0 | 81.4 | 80.1 | 51.6 | 40.4 | 74.7 | 40.5 | 30.8 | 62.3 | 28.8 | 20.5 |
| fasttrack_qcp_snap | 99.8 | 99.3 | 96.9 | 98.6 | 96.1 | 94.3 | 99.1 | 98.0 | 94.8 | 96.6 | 93.1 | 90.7 |
| fasttrack_smooth_qcp_snap | 99.8 | 99.3 | 96.9 | 98.4 | 96.0 | 94.6 | 99.5 | 98.2 | 94.6 | 96.6 | 93.0 | 90.9 |
| fasttrack_dct_qcp_snap | 99.8 | 99.2 | 96.9 | 98.7 | 96.2 | 94.4 | 99.2 | 98.2 | 94.6 | 96.8 | 92.8 | 90.6 |
| teacher | 99.8 | 99.0 | 98.9 | 96.5 | 98.4 | 98.2 | 96.0 | 98.4 | 99.2 | 90.7 | 96.0 | 95.7 |

n frames per column: men 1296, women 1392, boys 768, girls 528.

### DR (20 %, 200 Hz) % by f0 band (the token's steady-state f0)

| Baseline | <150 F1 | <150 F2 | <150 F3 | 150-300 F1 | 150-300 F2 | 150-300 F3 | 300-500 F1 | 300-500 F2 | 300-500 F3 |
|---|---|---|---|---|---|---|---|---|---|
| praat_default | 98.9 | 97.7 | 91.5 | 98.8 | 91.6 | 87.6 | 100.0 | 85.0 | 80.0 |
| praat_sweep | 100.0 | 96.7 | 95.0 | 90.4 | 76.8 | 75.1 | 100.0 | 90.0 | 87.5 |
| praat_fasttrack | 100.0 | 99.5 | 97.8 | 99.2 | 97.1 | 95.6 | 100.0 | 90.0 | 90.0 |
| tvlp | 96.4 | 91.8 | 80.7 | 97.9 | 88.0 | 85.2 | 100.0 | 92.5 | 82.5 |
| qcp_fb_published | 99.5 | 64.2 | 57.7 | 72.3 | 29.3 | 24.5 | 65.0 | 17.5 | 12.5 |
| qcp_fb | 98.8 | 89.9 | 83.6 | 77.4 | 48.2 | 38.3 | 72.5 | 55.0 | 15.0 |
| fasttrack_qcp_snap | 99.9 | 99.9 | 97.2 | 98.4 | 96.3 | 94.1 | 97.5 | 90.0 | 82.5 |
| fasttrack_smooth_qcp_snap | 99.9 | 99.9 | 97.2 | 98.5 | 96.3 | 94.2 | 97.5 | 90.0 | 82.5 |
| fasttrack_dct_qcp_snap | 99.9 | 99.8 | 97.2 | 98.6 | 96.3 | 94.1 | 97.5 | 90.0 | 82.5 |
| teacher | 99.9 | 99.4 | 99.0 | 95.6 | 97.9 | 98.1 | 100.0 | 100.0 | 95.0 |

n frames per column: <150 988, 150-300 2956, 300-500 40.

### DR (20 %, 200 Hz) % by frame position

| Baseline | 20/50/80% F1 | 20/50/80% F2 | 20/50/80% F3 | steady F1 | steady F2 | steady F3 |
|---|---|---|---|---|---|---|
| praat_default | 99.1 | 93.1 | 88.5 | 98.3 | 92.9 | 88.3 |
| praat_sweep | 93.0 | 81.7 | 80.3 | 92.6 | 82.2 | 79.9 |
| praat_fasttrack | 99.5 | 97.6 | 96.0 | 99.2 | 97.6 | 96.4 |
| tvlp | 97.6 | 88.3 | 83.8 | 97.4 | 91.2 | 84.8 |
| qcp_fb_published | 78.7 | 37.6 | 32.6 | 79.6 | 38.6 | 33.1 |
| qcp_fb | 82.8 | 58.6 | 49.7 | 82.3 | 58.8 | 48.6 |
| fasttrack_qcp_snap | 98.8 | 97.1 | 94.6 | 98.7 | 97.2 | 95.3 |
| fasttrack_smooth_qcp_snap | 98.8 | 97.1 | 94.7 | 98.8 | 97.2 | 95.3 |
| fasttrack_dct_qcp_snap | 98.9 | 97.1 | 94.6 | 98.9 | 97.3 | 95.2 |
| teacher | 97.1 | 98.4 | 98.3 | 95.4 | 98.0 | 98.3 |

n frames per column: 20/50/80% 2988, steady 996.

## f0

- **pesto**: PESTO mir-1k_g7 (pesto-pitch 2.0.1), 10 ms hop; voiced if confidence >= 0.5. Trained on: mir1k.
- **crepe**: CREPE full (torchcrepe 0.0.24), Viterbi decoding, 50-1600 Hz, 10 ms hop; voiced if periodicity >= 0.21. Trained on: mir1k, mdb_stem_synth.
- **praat_ac**: Praat autocorrelation (parselmouth), defaults except range 50-1000 Hz, 10 ms grid; unvoiced frames have no estimate. Trained on: none of our gold sets.
- **teacher**: teacher d040_5b_s2_b20k step 20000 (sha fa176ad17f13-fp32-w400), trained on synthetic data only; voiced if voicing p >= 0.5. Trained on: none of our gold sets.

RPA (raw pitch accuracy) and RCA (raw chroma accuracy) use a 50-cent tolerance and count every reference-voiced frame, whatever the estimated voicing decision. The voicing thresholds are fixed in advance (not tuned on gold data). **†** marks a baseline whose training data includes that corpus, so its score there is optimistic and not a fair test.

| Corpus | Baseline | Speakers | Voiced frames | RPA % | RCA % | Octave err % | GPE % | Voicing P % | Voicing R % | Voicing F1 % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ptdb_tug | pesto | 12 | 465,079 | 83.7 [79.6, 88.2] | 86.6 [82.6, 90.7] | 2.9 | 8.3 | 93.5 | 56.7 | 70.6 [65.2, 75.3] | 0.060 |
| ptdb_tug | crepe | 12 | 465,079 | 83.9 [79.7, 87.9] | 86.0 [82.0, 89.9] | 2.0 | 6.1 | 77.6 | 89.9 | 83.3 [80.0, 86.2] | 0.456 |
| ptdb_tug | praat_ac | 12 | 465,079 | 82.9 [78.9, 86.7] | 84.3 [80.4, 88.3] | 1.5 | 11.9 | 85.8 | 90.4 | 88.0 [85.6, 90.2] | 0.002 |
| ptdb_tug | teacher | 12 | 465,079 | 87.2 [83.0, 90.6] | 89.6 [86.2, 92.4] | 2.4 | 4.7 | 86.6 | 86.0 | 86.3 [83.9, 88.5] | 0.031 |
| mir1k | pesto † | 11 | 178,559 | 98.6 [98.2, 99.0] | 98.7 [98.3, 99.0] | 0.1 | 0.2 | 98.9 | 96.3 | 97.6 [97.0, 98.2] | 0.060 |
| mir1k | crepe † | 11 | 178,559 | 97.6 [97.0, 98.1] | 97.6 [97.0, 98.1] | 0.0 | 0.3 | 87.8 | 98.8 | 93.0 [90.8, 95.9] | 0.399 |
| mir1k | praat_ac | 11 | 178,559 | 97.3 [96.7, 97.9] | 97.7 [97.3, 98.1] | 0.4 | 2.2 | 96.6 | 98.6 | 97.6 [96.9, 98.2] | 0.001 |
| mir1k | teacher | 11 | 178,559 | 95.6 [94.9, 96.3] | 95.6 [94.9, 96.3] | 0.0 | 0.3 | 98.1 | 93.0 | 95.5 [94.8, 96.1] | 0.031 |
| mdb_stem_synth | pesto | 18 | 1,045,053 | 97.1 [96.2, 97.9] | 97.4 [96.6, 98.1] | 0.3 | 1.1 | 99.8 | 50.2 | 66.8 [52.2, 77.4] | 0.061 |
| mdb_stem_synth | crepe † | 18 | 1,045,053 | 96.2 [94.6, 97.6] | 96.4 [95.0, 97.7] | 0.2 | 1.8 | 97.1 | 97.8 | 97.4 [96.7, 98.1] | 0.284 |
| mdb_stem_synth | praat_ac | 18 | 1,045,053 | 95.6 [94.1, 97.0] | 95.6 [94.2, 97.0] | 0.0 | 3.0 | 98.8 | 97.2 | 98.0 [97.4, 98.6] | 0.001 |
| mdb_stem_synth | teacher | 18 | 1,045,053 | 98.3 [97.6, 99.0] | 98.5 [97.9, 99.0] | 0.2 | 0.5 | 99.9 | 95.8 | 97.8 [97.3, 98.5] | 0.035 |

### RPA % by reference f0 band

| Corpus | Baseline | <150 | 150-300 | 300-500 | >500 |
|---|---|---|---|---|---|
| ptdb_tug | pesto | 75.2 (n=247,819) | 93.4 (n=215,092) | 93.0 (n=2,167) | 0.0 (n=1) |
| ptdb_tug | crepe | 76.3 (n=247,819) | 92.7 (n=215,092) | 90.8 (n=2,167) | 0.0 (n=1) |
| ptdb_tug | praat_ac | 76.6 (n=247,819) | 90.0 (n=215,092) | 92.0 (n=2,167) | 0.0 (n=1) |
| ptdb_tug | teacher | 82.3 (n=247,819) | 92.8 (n=215,092) | 90.1 (n=2,167) | 0.0 (n=1) |
| mir1k | pesto | 97.5 (n=24,622) | 98.7 (n=109,535) | 99.2 (n=43,201) | 98.7 (n=1,201) |
| mir1k | crepe | 96.1 (n=24,622) | 97.5 (n=109,535) | 98.7 (n=43,201) | 97.8 (n=1,201) |
| mir1k | praat_ac | 95.0 (n=24,622) | 97.2 (n=109,535) | 98.7 (n=43,201) | 98.1 (n=1,201) |
| mir1k | teacher | 95.5 (n=24,622) | 95.3 (n=109,535) | 96.4 (n=43,201) | 95.6 (n=1,201) |
| mdb_stem_synth | pesto | 90.3 (n=56,901) | 97.0 (n=648,831) | 98.3 (n=284,856) | 98.9 (n=54,465) |
| mdb_stem_synth | crepe | 88.5 (n=56,901) | 96.5 (n=648,831) | 96.9 (n=284,856) | 97.5 (n=54,465) |
| mdb_stem_synth | praat_ac | 88.2 (n=56,901) | 95.7 (n=648,831) | 96.7 (n=284,856) | 97.1 (n=54,465) |
| mdb_stem_synth | teacher | 93.9 (n=56,901) | 98.4 (n=648,831) | 98.8 (n=284,856) | 99.2 (n=54,465) |

### RPA % on ptdb_tug subsets

*Modal* frames are those where the laryngograph shows regular vibration at the reference f0 (decisions D-009): the f0 is well defined there, so this is the fairest accuracy figure. Headline numbers use all frames.

| Baseline | all frames | modal | modal, ref < 85 Hz | female | female, modal | male | male, modal |
|---|---|---|---|---|---|---|---|
| pesto | 83.7 [79.6, 88.2] | 90.0 [85.6, 94.4] | 64.2 [28.5, 65.0] | 86.7 [80.7, 91.9] | 95.3 [92.6, 97.1] | 81.0 [75.5, 86.7] | 85.7 [80.1, 91.5] |
| crepe | 83.9 [79.7, 87.9] | 90.9 [87.4, 94.3] | 73.1 [23.4, 77.0] | 85.3 [79.3, 90.6] | 94.8 [92.2, 96.7] | 82.6 [76.5, 87.2] | 87.7 [82.5, 92.0] |
| praat_ac | 82.9 [78.9, 86.7] | 89.5 [86.3, 92.4] | 75.1 [44.9, 80.1] | 85.6 [79.6, 90.9] | 93.0 [90.1, 95.1] | 80.4 [74.4, 84.1] | 86.7 [81.7, 90.3] |
| teacher | 87.2 [83.0, 90.6] | 94.1 [91.4, 96.1] | 86.3 [66.8, 91.6] | 87.8 [81.5, 92.9] | 96.0 [93.2, 97.7] | 86.6 [80.9, 90.7] | 92.6 [88.0, 95.0] |

Voiced reference frames: all frames 465,079, modal 382,673, modal, ref < 85 Hz 34,152, female 223,269, female, modal 172,432, male 241,810, male, modal 210,241.

### ptdb_tug: where estimates land, by laryngograph phonation category

Each voiced reference frame is categorised from the laryngograph signal alone (`eval/phonation.py`). Cells give % of frames within 50 cents of the reference / of 2 × / of ½ the reference, and % the baseline judged voiced. Outside *modal* the f0 is ambiguous or the reference unreliable, so these are descriptions, not accuracies.

| Category | Share | pesto | crepe | praat_ac | teacher |
|---|---|---|---|---|---|
| **modal**: regular vibration at the reference f0 | 82.3 % | 90 / 1 / 0; voiced 63 | 91 / 0 / 0; voiced 94 | 90 / 0 / 1; voiced 95 | 94 / 0 / 0; voiced 92 |
| **ref_error_2x**: folds cycle at 2 × the reference: the reference halved | 0.4 % | 6 / 88 / 0; voiced 41 | 6 / 87 / 0; voiced 91 | 64 / 25 / 0; voiced 95 | 23 / 71 / 0; voiced 91 |
| **alternating**: alternate cycles differ (period doubling): the reference and 2 × are both defensible | 3.4 % | 62 / 25 / 0; voiced 46 | 56 / 28 / 0; voiced 73 | 85 / 2 / 2; voiced 94 | 82 / 10 / 0; voiced 89 |
| **aperiodic**: irregular phonation (creak): no single f0 | 9.7 % | 60 / 4 / 0; voiced 21 | 59 / 4 / 0; voiced 77 | 50 / 1 / 2; voiced 69 | 57 / 2 / 1; voiced 54 |
| **weak_egg**: too little laryngograph signal; the reference is unreliable | 4.2 % | 39 / 5 / 1; voiced 23 | 34 / 5 / 0; voiced 54 | 29 / 3 / 0; voiced 44 | 32 / 3 / 15; voiced 34 |

## Synthetic hard-case suite (fixed seed, `data/synth/suite_v1`)

DR is within (20 %, 200 Hz) on sonorant frames (voiced, plus whispered in `whisper`); f0 RPA within 50 cents on voiced frames; voicing accuracy over all frames. The channel is clean except in `low_snr`, `phone` and `codec`.

| Case | Frames | F1 DR | F2 DR | F3 DR | F4 DR | F1 MAE Hz | F2 MAE Hz | f0 RPA | Voicing acc |
|---|---|---|---|---|---|---|---|---|---|
| adult_high_f0 | 15,531 | 99.9 | 100.0 | 99.6 | 99.0 | 9 | 12 | 99.9 | 99.7 |
| breathy | 15,334 | 98.8 | 99.6 | 96.6 | 88.8 | 16 | 20 | 99.9 | 99.6 |
| codec | 14,866 | 97.8 | 98.1 | 89.7 | 77.6 | 19 | 30 | 99.4 | 99.3 |
| f1_near_h1 | 15,558 | 49.0 | 95.2 | 92.2 | 81.9 | 232 | 59 | 99.6 | 99.7 |
| f1_near_h2 | 14,345 | 59.3 | 82.6 | 80.8 | 77.9 | 412 | 154 | 99.9 | 99.6 |
| f2_f3_close | 14,211 | 98.3 | 98.5 | 79.4 | 71.3 | 18 | 40 | 99.9 | 99.6 |
| high_f0 | 15,183 | 98.1 | 98.0 | 93.2 | 76.1 | 25 | 38 | 100.0 | 99.6 |
| low_snr | 14,488 | 90.7 | 77.9 | 59.6 | 50.6 | 45 | 178 | 96.2 | 95.5 |
| nasal | 14,837 | 96.5 | 96.9 | 94.1 | 83.4 | 26 | 37 | 99.8 | 99.5 |
| phone | 14,570 | 97.9 | 93.1 | 69.7 | 55.7 | 21 | 76 | 99.4 | 99.1 |
| reference | 14,449 | 98.8 | 98.9 | 96.3 | 89.4 | 14 | 22 | 99.9 | 99.6 |
| tube | 15,104 | 99.2 | 99.6 | 100.0 | 98.1 | 16 | 18 | 99.9 | 99.7 |
| whisper | 15,612 | 99.7 | 99.4 | 97.8 | 94.2 | 10 | 17 | – | 100.0 |
| wide_vibrato | 15,401 | 98.5 | 99.3 | 96.1 | 86.7 | 17 | 24 | 99.6 | 99.6 |

## Sim-to-real gap by f0 band

Formant DR (20 %, 200 Hz) % on the synthetic `reference` case versus Hillenbrand, by f0 band (Hillenbrand uses each token's steady-state f0).

| f0 band | Synthetic F1 / F2 / F3 | Hillenbrand F1 / F2 / F3 | Gap F1 / F2 / F3 (points) |
|---|---|---|---|
| <150 | 100.0 / 99.8 / 100.0 | 99.9 / 99.4 / 99.0 | +0.1 / +0.5 / +1.0 |
| 150-300 | 99.9 / 100.0 / 99.2 | 95.6 / 97.9 / 98.1 | +4.4 / +2.1 / +1.1 |
| 300-500 | 99.6 / 99.8 / 99.0 | 100.0 / 100.0 / 95.0 | -0.4 / -0.2 / +4.0 |
| >500 | 96.4 / 96.7 / 88.7 | – | – |

## Provenance

| | |
|---|---|
| Commit | `b1db211` |
| Submodule | `7263f6c9e79442527229c4ae2be79a7c83f1beb2 external/random-cafe (heads/main)` |
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
| Manifest `zeroth_korean` | `bebcb05382d5074e` |
