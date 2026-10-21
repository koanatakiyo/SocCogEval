# Reproducibility

## Offline outcome reanalysis

Run `python scripts/reproduce_paper.py` from the repository root. This entry point reads only the explicit 168-run selection and produces fresh files under `results/reproduced/`.

The command verifies file hashes; checks all eight agents per run; reverses BFI items 1, 3, 4, 5, and 7; checks the factor means against stored scores; and recomputes run means. The five-factor L1 range is 0–20. It recomputes exact permutation tests at the run level, including the 3-versus-3 minimum two-sided p-value of 0.10. Primary/C6 bootstrap draws use the final audit's 50,000 draws and seed sequence beginning at 20260728. Cross-model sign summaries average scenarios within a model first.

The T=0 numerical check recomputes factor scores from all raw frozen-probe ratings in `results/reference/bfi_t0_*.json`, verifies eight agents per run, and compares each run mean with the saved sensitivity summary. Those probes are sensitivity measurements of existing histories, not new trajectories. The offline script does not reproduce serving-audit findings from the compact data.

## Primary source tables

| Result | Audited source |
|---|---|
| Primary C2–C4 and C6 contrasts | `results/reference/c6_decomposition.csv` |
| Primary agent/run diagnostics | `replica_level_bfi_diagnostics.csv`, `variance_icc_design_effect.csv` |
| Signed factor movements | `signed_factor_coherence.csv` |
| Five-model C2–C4 | `cross_variant_c4_results.csv`, `cross_variant_c4_variant_summary.csv` |
| Model run selection | `cross_variant_c4_selected_runs.csv` |
| No-SRM | `nosrm_exact_permutation.csv` |
| C5 floor | `c5_floor_audit.csv` |
| T=0 sensitivity | `bfi_t0_summary.json` |
| C6 protocol/exposure audit | `c6_validation.csv`, `c6_run_level.csv`, `c6_exposure_summary.csv` |

Names without a full path are under `results/reference/`. The early `primary_c2_c4_run_level.csv` uses a 10,000-draw bootstrap; its endpoints differ slightly from the final 50,000-draw C6 table. The paper consistently uses the latter. Point estimates and exact p-values agree.

Signed directional agreement compares each C2 run's signed movement with the mean signed C4 movement. It does not count only raw positive/negative post-minus-pre scores. The C6 fractions 75%/78% are ratios of observed contrasts, not causal mediation estimates.

## Original utilities and required artifacts

The original scripts are retained for method inspection and expanded reruns. Their historical output directory is generally `rebuttal/2026-07-25/`; this path refers to generated statistical artifacts, not reviewer correspondence.

- `build_rebuttal_evidence.py`: can recompute measurement-based statistics, but its historical memory and trace audit needs full original persistence directories. Regenerated metadata from a compact checkout cannot certify the historical audit.
- `analyze_c6_decomposition.py`: requires full C6 `exposure_calls.jsonl`, round artifacts, and serving metadata. Create its output directory before use. The compact package supplies the original audited summaries instead.
- `run_bfi_temperature_sensitivity.py`: makes live model calls and needs persisted memory checkpoints. Do not run it as an offline check.
- `analyze_bfi_temperature_sensitivity.py`: expects temperature snapshots at its historical evidence path; the release stores them under `results/reference/`.
- `analysis_*` scripts: original supporting analyses, some requiring earlier generated analysis files in `data/` or pilot inputs outside the compact selection. Historical agent-level p-values in these utilities do not override the revised paper's run-level inference.
- Objective-specificity utilities and `cogclaw/objective_probe.py`: experimental follow-up tooling. Objective-capability results are not part of this paper; external benchmark inputs for those utilities are not bundled.

## Live regeneration

The local model service must expose the model name passed to the runner. Configure endpoint/model/environment values explicitly. The model-cache root is controlled by `LOCAL_MODEL_CACHE`; no developer-machine paths are required. Gemini uses Google ADC and the environment variables described in `.env.example`. DeepSeek defaults inherited from the original source can be overridden by the explicit route in that file. Final selected historical DeepSeek C2/C4 identifiers differ, so no exact revision match is claimed.

Install and pin the semantic embedding model for SRM. The source can fall back if that dependency is unavailable; such a fallback is a different experimental environment and must be recorded. Historical request sampling seeds and full serving revisions were not captured. Recording a Python seed alone does not make hosted model calls deterministically reproducible.

`run_experiment.py` writes timestamped experiment directories under `data/`. Use a distinct `--run-id` for new work. The fixed selection manifest keeps new runs out of the published offline estimates. C6 requires an inspectable local vLLM process matching its frozen TP=1, max-context 8192, max-sequences 8 serving profile and a checkpoint/tokenizer path. The full protocol and amendment are in `docs/protocols/`.

## Paper build

```bash
cd paper
python make_tables.py && python make_tables_cr.py   # table fragments from evidence/, evidence_sub/, evidence_cr/
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`paper/figscripts/` regenerates the result plots (matplotlib; Georgia must be installed, or matplotlib falls back to a default serif). `paper/figures_src/` holds the editable draw.io sources of Figures 1–4 and the scripts that set their 8pt labels. `paper/analysis/` contains the camera-ready run-level analyses. They read `data/` and `results/reference/` only, except `belief_time_course.py`, which needs the per-round probe files that this compact release leaves out.
