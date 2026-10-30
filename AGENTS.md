# AGENTS.md — voice-analyser

A voice analyser: f0, voicing, formants F1–F4 with bandwidths, and related measures, from a teacher model trained on synthetic and resynthesised speech and singing. Python package `vocal_analysis`, CLI `va`. Personal research project; the long-term aim is measuring voice-surgery outcomes on consented data.

## Layout

- `vocal_analysis/`: `synth/` (synthesiser, channel model, codecs), `resynth/` (resynthesis of real recordings), `train/` (teacher training, data loading), `models/` (teacher, inference), `eval/` (metrics, gold-set loaders, tiers), `baselines/`, `backend/`, `features/` (VQT), `cli.py`.
- `docs/decisions.md`: the decision log (D-xxx). **Read the relevant entries before changing anything they cover.** `docs/roadmap.md` lists what's next; `docs/considering.md` holds unscheduled ideas; `docs/models.md` is the model register. `docs/f1_bias_register.md` indexes everything tried for the women's F1 overread; add a row when an entry tests a cause or fix.
- `scripts/analysis/`: one-off analyses and decision scripts; `scripts/analysis/chains/` holds the GPU job chains.
- `scripts/parakeet/`: the Parakeet probe, student distillation (D-033, D-034) and VTL probe scripts (D-035; the MRI midline scripts moved to the private repo `vtl-mri` on 2026-10-28); merged from the `parakeet-probe` worktree. They read `cache/` (feature and label caches) and `external_models/` (Parakeet weights), both uncommitted; `pip install -e .[parakeet]`.
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
- **Before moving or deleting a directory (worktrees especially), run `python scripts/links_into.py <dir>`.** We use junctions to share data between parallel worktrees: a junction pointing *into* the directory means another tree's files live there (main's `reports/analysis` once pointed into a worktree, and removing the worktree deleted them); one *inside* it can be followed by a recursive delete. Remove a junction with `rmdir`, which leaves its target.
- **Every script gets a main guard (`if __name__ == "__main__":`), throwaway ones included.** DataLoader workers and other multiprocessing re-import the script on Windows (spawn), so top-level code runs again in every worker: it has crashed loaders and hung a GPU check.
- **The iGPU drives the display.** Keep single GPU operations short: chunk big matmuls.
- **Prefer seed means over single seeds** (seed-to-seed variation on real voices is large), and paired comparisons with bootstrap intervals.
- **Check a hypothesis can hold before pre-registering it**: estimate the measure's noise floor (e.g. a shuffle null; a small effect seen once may be sampling noise), and trace the code path to confirm the proposed cause can reach what is measured.
- **Committed reports are immutable.** A correction or a new layout goes into a new file, with a dated note in the decision log or plan pointing to it; the original stays as the record of what was reported then (the git history is to be archived with a DOI).
- **Commit messages** say what changed and why, referencing D-xxx.
