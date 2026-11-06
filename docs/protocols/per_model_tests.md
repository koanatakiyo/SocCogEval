# Per-model (C2)−(C4) tests

Paper: Appendix E.1, Table 12. Reproduce with `python scripts/reproduce_per_model.py`.

## Purpose

The cross-variant panel has 3 C2 and 3 C4 runs per model and scenario, so no single model can reach
p < 0.05 (the minimum two-sided exact p is 0.10). These runs bring each tested model to 8 C2 and 5 C4 runs per
scenario, matching the primary model, so that each model can be tested on its own.

## Models

| Alias | Model | Model ID in `config.json` | Route | Runs |
|---|---|---|---|---|
| `gpt` | GPT-4.1-mini | `gpt-4.1-mini` | OpenAI API | extends the 3+3 panel runs with C2 rep03–07, C4 rep03–04 |
| `l4api` | Llama-4-Scout | `llama4-scout` | OpenRouter `meta-llama/llama-4-scout:floor` | extends the 3+3 panel runs with C2 rep03–07, C4 rep03–04 |
| `dsf` | DeepSeek-V4.1-Flash | `deepseek-flash` | DeepSeek API, thinking disabled | C2 rep00–07, C4 rep00–04 |
| `sdm` | Seed-2.0-mini | `seed-2.0-mini` | OpenRouter, pinned to the Seed provider, reasoning off | C2 rep00–07, C4 rep00–04 |
| `glm47` | GLM-4.7-Flash | `glm-4.7-flash` | OpenRouter, pinned to the Cloudflare provider, reasoning off | C2 rep00–07, C4 rep00–04 |

Panel variants were included when their C2 and C4 runs used the same configured model identifier. DeepSeek-V3's
panel runs used different identifiers for C2 and C4, so DeepSeek-V4.1-Flash is a separate variant and is not pooled
with DeepSeek-V3. `deepseek-flash` is a rolling provider name without a dated snapshot. Gemini-2.5-Flash met the
inclusion rule but was not run because its cloud billing account was unavailable. Seed-2.0-mini and GLM-4.7-Flash
add model families. Pinned providers have fallback disabled (`cogclaw/config.py`, `OPENROUTER_PINNED_MODELS`), so an
outage fails calls instead of switching serving stacks.

## Design

- Configurations: `configs/condition_2_segreg_asym_seed_<scenario>.json` and
  `configs/condition_4_seed_only_no_interact_<scenario>.json`, SRM memory, 30 rounds, frozen temperatures.
- Seeds: C2 = base + rep, C4 = base + 200 + rep, with base 2000 (gpt), 4000 (l4api), 7000 (dsf), 9000 (sdm),
  8000 (glm47).
- C4 measures BFI-10 before any prediction probe; the prediction probe needs prior interaction and is skipped in C4.

## Analysis

- Outcome: run-mean BFI-10 L1 drift, recomputed from the raw ratings.
- Per model and scenario: Δ = mean(C2) − mean(C4), two-sided exact permutation test over all label allocations,
  run-bootstrap 95% interval (50,000 draws, `random.Random(20261020)`).
- Every cell is reported whatever its result. The 3+3 panel numbers in the paper are unchanged.
- The table also shows each model's C4 drift, its no-interaction retest level, next to the difference.

## Notes

- GLM-4.7-Flash answers BFI-10 almost at random with reasoning off. In a post hoc check, its retest L1 for an
  identical prompt at T = 0.7 was 4.85, against 5.67 for uniform random answers and 0.55 for GPT-4.1-mini, so its
  cells are uninformative about interaction. No run is excluded.
- The planned inclusion time was 2026-11-06 12:00 SGT. One Llama-4-Scout Riverton C2 run (rep05) finished later,
  after a provider outage and a restart, and is included. Without it, the cell is Δ = +1.089 (7 vs. 5 runs,
  p = 0.0025).
