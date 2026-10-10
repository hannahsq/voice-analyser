# Analysis scripts

The scripts behind `reports/analysis/`, which `docs/decisions.md` (D-019 onwards) cites. They are research scripts, not part of the package: each takes its inputs on the command line and prints or writes a plain-text report.

**Where to run them.** Most need this branch's code (`vocal_analysis.eval.invariance`, `eval.closed_phase`, `synth.coupling`). Run them from this worktree with `PYTHONPATH=.`, using the main checkout's virtual environment. Checkpoints, data and caches live in the main checkout (`../voice-analyser/runs`, `../voice-analyser/data`). The docstrings say where each script expects to run.

**Scratch directory.** Several scripts take a scratch directory as their first argument. It holds intermediate files that are large or regenerable:
- the closed-phase frames (`arctic_egg.npz`, `ptdb_egg.npz`, and their `_sub` versions);
- the paired comparisons' prediction caches (`m2cache/`, `m2cache_sub/`).

| Script | Produces | Decision |
|---|---|---|
| `arctic_egg.py`, `ptdb_egg.py` | closed-phase reference frames (scratch `.npz`) | D-019 |
| `add_model.py` | one teacher's F1–F3 added to those frames (optional `_sub` subset) | D-020 onwards |
| `cp_cmp.py` | `closed_phase/`: closed-phase F1 bias and error by voice and sex, with a speaker bootstrap on model pairs | D-020, D-022, seed replicate |
| `m2_paired.py`, `m2_paired_sub.py` | `paired/`: Hillenbrand F1–F4 and PTDB-TUG f0 RPA, paired speaker bootstrap (`_sub`: every 3rd PTDB-TUG file per speaker) | D-014, D-020, D-022 |
| `invariance_run.py` | `invariance/`: F1 slope against f0 on the tube probe | D-020 onwards |
| `cross_val.py` | `cross_val/`: swapped synthetic validation | D-020, D-022 |
| `sigma_check.py` | `sigma/`: σ against error, synthetic and closed phase | D-021 |
| `sensitivity.py`, `sensitivity_f123.py`, `sensitivity_praat.py`, `sensitivity_bw.py`, `sensitivity_bw_cmp.py`, `f1_ratio_sweep.py` | `sensitivity/`: controlled renders, one factor at a time | D-022, D-023 |
| `open_glottis.py`, `coupling_check.py` | `sensitivity/`: open-glottis tube poles; coupled-render check | D-023 |
| `birkholz_eval.py` | `external/`: Birkholz et al. (2019) audio, evaluation only | D-023 |
| `listen.py` | listening WAVs (`runs/listening/`) | D-023 |
| `bench_loader.py`, `bench_infer.py` | data-loader and inference speed (`bench_small.txt`) | small-model proxy |

`chains/` holds the shell queues that ran training and measures one after another on a single machine. They are kept as a record of exactly what was run, in what order. They hard-code that machine's paths and take the scratch directory as their argument.
