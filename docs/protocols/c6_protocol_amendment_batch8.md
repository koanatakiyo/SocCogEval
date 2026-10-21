# C6 throughput-only protocol amendment

Frozen amendment date: 2026-07-28, before inspection of any formal C6 outcome.

## Reason

The original serial smoke/profile used one logical request at a time
(`max_num_seqs=1`). During the first two incomplete formal runs, both dedicated
H200 endpoints remained at roughly 31–32% GPU utilization. The amendment is
based only on serving throughput and elapsed time, not on experimental
outcomes. Neither partial run produced `summary.json` or post-battery results.

## Change

- Keep one C6 process and one dedicated TP=1 endpoint per GPU.
- Set vLLM `max_num_seqs=8`.
- For each round, use four explicit barriers:
  1. eight independent agent turn-1 calls;
  2. eight independent agent turn-2 calls after all turn-1 calls finish;
  3. eight independent agent turn-3 calls after all turn-2 calls finish;
  4. eight independent SRM extraction calls after all turn-3 calls finish.
- Preserve deterministic agent ordering in saved round artifacts.
- Serialize token counting, counters, and JSONL exposure writes with a lock.

## Unchanged

- Model checkpoint and tokenizer hashes.
- TP=1, FP8, dtype, max model length, eager mode, GDN backend, endpoint
  isolation, and GPU-memory-utilization setting.
- Scenario, A/B brief assignment, prompts, prompt dependencies, agent state,
  memory policy, temperatures, maximum output tokens, call counts, number of
  rounds, probe cadence, and absence of partner information.
- No request-level sampling seed, matching the submitted C2 path.

The amendment changes request scheduling and possible vLLM batch placement
only. Batch placement is therefore recorded prospectively in
`serving_metadata.json` and every exposure row.

## Exclusions and restart rule

The two serial partial runs below are operationally aborted and excluded:

- `c6rep00_solo_generation_greenfield_20260727_235920`
- `c6rep02_solo_generation_greenfield_20260727_235926`

They are preserved rather than deleted. No serial partial run may be combined
with batch-8 formal results. All ten formal replicas must restart from round 1
under this amended configuration.

## Go/no-go gate

Before formal launch, run one 1-round smoke on each endpoint. Proceed only if:

- serving validation records TP=1 and `max_num_seqs=8`;
- each smoke has 24 content and 8 extraction calls;
- all exposure rows record client batch size 8;
- no endpoint is shared by another C6 process;
- both smokes complete without OOM or malformed artifacts; and
- elapsed time is materially lower than the serial local smoke baseline.

All formal replicas, including nulls and reversals, will be reported.
