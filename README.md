# CogClaw

**Computational Social Cognition Experiment Pipeline for LLM Agent Societies**

CogClaw is a research pipeline for studying how social information structures produce behavioral shifts in identically initialized LLM agents. It supports controlled factorial experiments where agents interact under different network topologies (open vs. segregated) with asymmetric seed information, measuring divergence across multiple tasks including allocation decisions, conformity behavior, personality profiles, and predictions of others.

## Paper

**Beyond Opinion Divergence: Cross-Task Evidence for Socially Induced Behavioral Shifts in Identically Initialized LLM Agents**

- Deadline: Abstract May 4, Full Paper May 6, 2026 (AoE)
- Target: NeurIPS 2025 format, ~10 pages + appendix
- Canonical plan: `working_plan_v3.md`

## Key Idea

We run a controlled experiment where identically initialized LLM agents interact under different social information structures. After 30 rounds of conversation, we measure divergence not only in discussion-related allocation decisions, but also in **unrelated** conformity tests, personality profiles, and predictions of others. The key finding is whether social interaction produces behavioral shifts that **transfer across tasks** -- indicating changes beyond surface opinion, toward persistent internal state shifts.

## Experimental Design

### 5 Conditions (x 2 scenarios x repeats)

| # | Condition | Network | Seed | Interaction | Repeats |
|---|-----------|---------|------|-------------|---------|
| 1 | Open+Seed | All-to-all | Same | 30 rounds | 7 |
| 2 | Segreg+AsymSeed | Two groups | Different | 30 rounds | 8 |
| 3 | Segreg+NoSeed | Two groups | None | 30 rounds | 5 |
| 4 | SeedOnly+NoInteract | N/A | Different | None | 5 |
| 5 | NoSeed+NoInteract | N/A | None | None | 3 |

### 2 Scenarios

- **Greenfield**: $10M community grant allocation across 5 investment areas
- **Riverton**: $5M AI transition budget allocation across 5 areas

### 4 Primary Metrics

| # | Metric | What it shows |
|---|--------|---------------|
| P1 | Allocation vector divergence | Decision divergence (main result) |
| P2 | Prediction accuracy (in-group vs out-group) | Internalized group models |
| P3 | Conformity rate change (pre vs post) | Cross-task transfer |
| P4 | Causal decomposition | Mechanism (seed vs interaction vs structure) |

## Project Structure

```
cogclaw/
├── cogclaw/                         # Core pipeline package
│   ├── config.py                    # Configuration (API params, ExperimentConfig)
│   ├── llm.py                       # LLM client (Anthropic Claude + DeepSeek)
│   ├── agent.py                     # Agent class + SocialRelationalMemory
│   ├── memory.py                    # Memory extraction (LLM -> JSON) + consistency validation
│   ├── network.py                   # Network topology (all-to-all, segregated, none)
│   ├── orchestrator.py              # Conversation orchestrator (rounds, pairs, data saving)
│   └── conditions.py                # Condition factory (5 conditions from the plan)
├── scenarios/                       # Scenario prompts and measurement instruments
│   ├── greenfield.py                # Community grant scenario (context, seeds, probes)
│   └── riverton.py                  # AI transition scenario (context, seeds, probes)
├── configs/                         # Generated experiment configs (5 conditions x 2 scenarios)
├── data/                            # Experiment output data
├── benchmark_datasets/              # Downloaded baseline datasets
│   ├── PsychoBench/                 # BFI-10 personality (pre/post measurement)
│   ├── CogMir/                      # Conformity/herd effect test items
│   ├── ToMBench/                    # Theory of Mind reference (2,860 items)
│   └── CogBench/                    # Cognitive psychology experiments (citation)
├── run_experiment.py                # Main experiment runner
├── toy_test.py                      # Quick validation (4 agents x 5 rounds)
├── working_plan_v3.md               # Canonical research plan (FROZEN)
├── benchmark_analysis.md            # Dataset analysis and evaluation strategy
└── requirements.txt                 # Python dependencies
```

### Reference Codebases

| Directory | Project | Relevance |
|-----------|---------|-----------|
| `FutureMap-main/` | Prediction engine with LLM reasoning loops | LLM client patterns, JSON repair |
| `KnowledgeClaw/` | Knowledge graph engine (entity-hyperedge) | Storage, retrieval, graph patterns |
| `MiroFish-main/` | Multi-agent simulation orchestrator | Profile generation, OASIS integration |
| `oasis/` | OASIS social media simulation platform | Agent/environment/network patterns |
| `openclaw-main/` | AI assistant framework | Provider/model selection patterns |
| `HyperGraphRAG/` | Hypergraph-based RAG (NeurIPS 2025) | Future: hypergraph memory (Paper 2) |

## Quick Start

### Prerequisites

```bash
conda activate cogclaw
pip install -r requirements.txt
```

API keys (place in project root):
- `api_key_anthropic.txt` -- for Claude Sonnet 4
- `api_key_deepseek.txt` -- for DeepSeek-V3 (pilot/cross-check)

### Run Toy Test

```bash
# 4 agents, 5 rounds, segregated + asymmetric seed (DeepSeek for cost savings)
python toy_test.py --model deepseek-chat

# Memory extraction consistency validation (target: >90%)
python toy_test.py --consistency-test --model deepseek-chat
```

### Run Full Experiment

```bash
# Single condition
python run_experiment.py --condition 2 --scenario greenfield --model deepseek-chat

# Override round count (e.g., for pilot)
python run_experiment.py --condition 2 --scenario greenfield --n-rounds 10 --model deepseek-chat

# All 5 conditions for one scenario
python run_experiment.py --all --scenario greenfield --model deepseek-chat

# With S2 deliberate reasoning
python run_experiment.py --condition 2 --scenario greenfield --s2

# Generate all config files
python run_experiment.py --generate-configs

# Run from config file
python run_experiment.py --config-file configs/condition_2_segreg_asym_seed_greenfield.json
```

### Output Structure

Each run produces:
```
data/<run_name>/
├── config.json                # Experiment configuration
├── rounds/
│   ├── round_001.json         # Conversations + probe responses
│   ├── round_002.json
│   └── ...
├── memories/
│   ├── Alex.json              # Agent's social-relational memory
│   ├── Blake.json
│   └── ...
├── post_measurements.json     # Allocation, prediction, belief, conformity
└── summary.json               # Run summary statistics
```

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Temperature (conversations) | 0.7 |
| Temperature (probes/tests) | 0.0 |
| Max tokens (conversations) | 1024 |
| Max tokens (probes) | 512 |
| Max tokens (allocation) | 256 |
| Primary model | Gemini 2.5 Pro |
| Cross-check model | Qwen 3 122B |
| Agents per run | 8 (identically initialized) |
| Rounds per run | 30 |

## Agent Design

Agents are **identically initialized** with:
- Same base model and prompting template
- Same role structure (community member / employee)
- Brief name and role label
- Behavioral instruction: "Express your genuine thoughts honestly"
- **NO** personality, stance, ideology, or knowledge base injected

After each conversation, a **social-relational memory** is extracted via LLM:
```json
{
  "episodic": { "event": "...", "context": "...", "my_response": "..." },
  "social": { "participants": [...], "their_stances": {...}, "group_dynamics": "..." },
  "self_relevant": { "how_they_see_me": "...", "my_evolution": "..." }
}
```

## Measurement Battery

| When | Instrument | Source |
|------|-----------|--------|
| Pre/Post | BFI-10 personality | PsychoBench |
| Pre/Post | Conformity test (herd effect) | CogMir |
| Every 5 rounds | Belief probe | Own design |
| Every 5 rounds | Common ground probe | Own design |
| Post only | Allocation task (5-dim vector) | Own design |
| Post only | Prediction task (predict others) | Own design |
| Post only | Consistency check (3 rephrasings) | Own design |

## Schedule

| Week | Dates | Deliverable |
|------|-------|-------------|
| 1 | Mar 18-24 | Pipeline + validated extraction consistency |
| 2 | Mar 25-31 | Measurement system + pilot data + power analysis |
| 3 | Apr 1-7 | All experimental data collected (66 runs) |
| 4 | Apr 8-14 | All figures + tables + statistical results |
| 5 | Apr 15-21 | Complete paper draft |
| 6 | Apr 22-28 | Revision + NeurIPS formatting |
| 7 | Apr 29-May 6 | Abstract (May 4) + Full paper (May 6) |

## Budget

~$100-120 total API cost. Extra repeats use DeepSeek ($0.30/run) to stay within budget.

## License

Research code. To be released under MIT license upon submission.
