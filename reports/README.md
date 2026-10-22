# Reports

- `suite/`: whole-suite reports written by the CLI (`va baselines`, `va eval teacher`): every baseline and model on
  the gold sets (Hillenbrand, PTDB-TUG, MIR-1K, MDB-stem-synth) and the synthetic suite, with a provenance header.
  Scored on dev speakers by default; the tiers are in the file name and the header (D-024, D-032), and any report
  beyond dev is a use logged in `docs/decisions.md`. `suite/pre_tiers/` holds the ones written before the tiers
  existed, on every gold speaker.
- `analysis/`: outputs of `scripts/analysis/`, one experiment or decision each, cited by `docs/decisions.md`.

Model names and checkpoints are in `docs/models.md`.
