# Measurement data

The selected dataset contains 168 completed runs and 672 JSON files: `config.json`, `pre_measurements.json`, `post_measurements.json`, and `summary.json` per run. Agent names denote synthetic roles, not human participants.

`data/run_selection.json` assigns each run a scenario, condition, variant alias, memory mode, and original source path. The `source` field preserves provenance in the original workspace; offline analysis loads the local `data/<run_id>/` directory and does not require that old path. `data/measurement_checksums.json` records the SHA-256 of each measurement file. The copies are byte-preserving.

Coverage:

| Family | Runs |
|---|---:|
| Primary C1–C5, two scenarios | 56 |
| Five additional variants, C2/C4/C5, two scenarios | 90 |
| Primary verbatim-history C2/C5, two scenarios | 12 |
| Primary solo-generation C6, two scenarios | 10 |
| Total | 168 |

The 128-run original ledger plus selected cross-variant C4 and formal C6 manifests determine inclusion. All ten cross-variant C2/C4 scenario cells are present, including the slightly negative DeepSeek Riverton effect. Frozen-state temperature-zero measurements are in `results/reference/` and do not increase the run count.

Raw BFI rows contain ten Likert responses, five derived factor scores, and agent identity. Pre/post arrays also contain allocation and conformity outputs; post arrays can contain wording-consistency and prediction outputs. Missing predictions in no-interaction conditions reflect an instrument that requires prior interactions, not a missing BFI observation.

The compact package excludes conversation transcripts, persistence trees, model checkpoints, exposure streams, incomplete attempts, and large pilot datasets. Consequently, the raw outcome estimates can be verified, but the historical serving/trace audits cannot be reconstructed in full from this release alone. Historical metadata gaps are recorded explicitly in the reference ledger.
