# AGENTS.md — voice-analyser

A voice analyser: f0, voicing, formants F1–F4 with bandwidths, and related measures, from a teacher model trained on synthetic and resynthesised speech and singing. Python package `vocal_analysis`, CLI `va`. Personal research project; the long-term aim is measuring voice-surgery outcomes on consented data.

## Layout

- `vocal_analysis/`: `synth/` (synthesiser, channel model, codecs), `resynth/` (resynthesis of real recordings), `train/` (teacher training, data loading), `models/` (teacher, inference), `eval/` (metrics, gold-set loaders, tiers), `baselines/`, `backend/`, `features/` (VQT), `cli.py`.
- `docs/decisions.md`: the decision log (D-xxx). **Read the relevant entries before changing anything they cover.** `docs/roadmap.md` lists what's next; `docs/models.md` is the model register.
- `scripts/analysis/`: one-off analyses and decision scripts; `scripts/analysis/chains/` holds the GPU job chains.
- `reports/`: committed results (`reports/suite/` for whole-suite reports, `reports/analysis/` per study). `runs/`, `data/` and caches are not committed.
- `external/`: submodules. `voice-dataset-labeler` (TVLP tracker, pinned); `random-cafe` (background-scene augmentation, optional: `update = none`).

## Running

- Tests: `.venv/Scripts/python.exe -m pytest -q` (Windows venv; GPU via ROCm, `VA_DEVICE=cuda`).
- Training and evaluation: `va train teacher ...`, `va eval teacher ...`, `va baselines ...` (`--tiers`, dev by default).

## Rules that are not optional

- **Pre-register, then run.** Every experiment's arms, data, measures and decision rule go into `docs/decisions.md` before anything runs. Change one thing at a time. Amendments are recorded, with whether results had been seen.
- **Evaluation tiers.** Decide on dev speakers; use semi-private only to confirm, logging each use; lockbox only at milestone acceptance.
- **Evaluation sets are never trained on** and never fit synthesis priors: Hillenbrand, PTDB-TUG, ARCTIC, Birkholz 2019, OPENGLOT, MOCHA-TIMIT, Saarbrücken (also MIR-1K and MDB-stem-synth for f0).
- **Data handling.**
  - Never try to identify speakers.
  - Zeroth-Korean: numeric ids and recorded sex only, never names.
  - No unlicensed TIMIT.
  - Avoid Hugging Face downloads; fetch originals (OpenSLR, Zenodo, GitHub releases).
  - One scraped dataset must never be used here in any form.
- **Licences.** Code MIT. Weights take the most restrictive licence of their training data: CC BY 4.0 for the current pools, so no NC/SA data in training. Record every corpus in `ATTRIBUTION.md` before use.
- **Downloads are untrusted.** Each goes into its own new directory under `C:\github\datasets\`. Check archive sizes and paths before unpacking.

## Working conventions

- **GPU jobs run as chains** (`scripts/analysis/chains/`), copied before running so edits don't affect a running queue. They log to `runs/orchestrator.log` ("FAILED: ..." on failure). Every step skips if its output exists.
- **Stop whole process trees.** Orphaned loader workers can hold tens of GB.
- **The iGPU drives the display.** Keep single GPU operations short: chunk big matmuls.
- **Prefer seed means over single seeds** (seed-to-seed variation on real voices is large), and paired comparisons with bootstrap intervals.
- **Commit messages** say what changed and why, referencing D-xxx.
