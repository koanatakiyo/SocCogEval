# C6 matched solo-generation protocol

Frozen: 2026-07-25, before inspecting any C6 outcomes.

Factual clarification: 2026-07-27, before any C6 model call. Production C2
matches the total per-round exposure budget, but its random pairing does not
fix that budget per agent.

Operational amendment: 2026-07-27, before any C6 model call. Available GPUs
are used as three independent one-GPU workers (one remote, two local), all
with the same frozen serving profile: tensor parallel 1, max model length
8,192, max concurrent sequences 1, GPU memory utilization 0.92, eager mode,
and Triton GDN prefill. Endpoint port differs only to host the second local

## Estimands

- C6-C4: extended solo generation/history under the same asymmetric brief.
- C2-C6: reciprocal/adaptive interaction beyond matched generation exposure.
- C2-C4: total interaction-plus-generation contrast under the same brief.

C6-C5 is prohibited as a generation-effect estimate because C6 has briefs and
C5 does not.

## Configuration

- Model: Qwen3-122B-FP8, using the same exact served revision as the matched
  production C2/C4 runs if recoverable; otherwise record the new revision and
  disclose the mismatch.
- Scenarios: Greenfield and Riverton.
- Replicas: 5 per scenario.
- Agents: 8 controlled clones.
- Briefs: agents 0-3 receive Brief A; agents 4-7 receive Brief B.
- Horizon: 30 rounds.
- Generation calls: 24 solo content-generation calls per round, matching C2's
  4 conversations × 6 turns = 24 content calls; 720 calls per run.
- Distribution: exactly 3 solo calls per agent per round. Report this
  difference from C2's randomly realized per-agent speaking counts.
- Memory: the same SRM mode and total extraction budget as production C2:
  8 extraction calls per round. C6 fixes this at one extraction per agent;
  C2 runs two extractions per conversation, so random pairing also makes its
  per-agent extraction count variable. Report this distributional difference.
- Probe cadence: the same pre battery, periodic probes after rounds
  5/10/15/20/25/30, and immediate post battery.
- Decoding: content generation T=.7, max 1,024 tokens; memory extraction T=0;
  probes use their submitted instrument-specific settings.
- No partner names, partner messages, dialogue simulation, majority cues, or
  adaptive feedback appear in the solo-generation prompts.
- Each agent's own three outputs from the current round are visible within
  that round. Prior-round access occurs only through the same frozen memory
  interface used in C2.

## Frozen solo prompts

Turn 1:

> This is round {round_num} of a private reflection exercise about the
> committee topic. Develop your current reasoning about the allocation
> problem. Consider the values, evidence, and tradeoffs that matter to you.
> Do not address, quote, predict, or simulate another person. Write only your
> own reflection.

Turn 2:

> Continue your private reflection. Reconsider your reasoning from another
> angle, identify a genuine tension or uncertainty, and explain how you would
> resolve it. Do not address, quote, predict, or simulate another person.
>
> Your reflection so far:
> {own_turn_1}

Turn 3:

> Complete this round's private reflection by stating your current best
> allocation reasoning and what, if anything, changed in your own view. Do
> not address, quote, predict, or simulate another person.
>
> Your reflection so far:
> {own_turn_1}
> {own_turn_2}

The prompts are identical across agents except for round number and the
agent's own prior output.

## Exposure logging

For every content and extraction call, save:

- run, scenario, replica, agent, round, and turn;
- request-system hash and user-prompt hash;
- model ID and served revision;
- temperature and maximum tokens;
- output text hash;
- provider-reported input/output token counts where available;
- otherwise tokenizer ID/revision and locally counted token IDs;
- serving endpoint, precision, and batch configuration.

Report intended call counts and realized input/output token totals by
condition. Do not truncate outcomes after inspection to force exact equality.

## Frozen interpretation

|Pattern|Interpretation|
|---|---|
|C2 > C6 approximately C4|Reciprocal interaction is the main driver.|
|C2 approximately C6 > C4|Extended generation/history is sufficient; an interaction-specific mechanism is unsupported.|
|C2 > C6 > C4|Generation and reciprocal interaction both contribute.|
|C6 > C2|Solo generation shifts more; narrow or abandon the proposed interaction mechanism.|
|All near zero|The added control does not reproduce the submitted effect; report uncertainty without mechanism expansion.|

All contrasts use run-level raw effects and descriptive run-bootstrap
intervals. No per-cell p<.05 success gate is imposed.
