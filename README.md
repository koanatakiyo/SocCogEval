# CogClaw

**Computational Social Cognition Experiment Pipeline for LLM Agent Societies**

CogClaw is the experiment pipeline and benchmark toolkit for studying whether social interaction produces transferable behavioral shifts in identically initialized LLM agents. It implements the Social Cognitive Transfer (SCT) task, the SocCogBench evaluation benchmark, and three reusable methodology components (SRM, FSIP, S1/S2).

## Paper

**Beyond Opinion Divergence: Cross-Task Evidence for Socially Induced Behavioral Shifts in Identically Initialized LLM Agents**

- Deadline: Abstract May 4, Full Paper May 6, 2026 (AoE)
- Canonical plan: `working_plan_v3.md`

---

## Three Contributions

### C1: Social Cognitive Transfer (SCT) Task (Section 3)

A new task: given identically initialized LLM agents interacting under controlled social information structures for N rounds, does behavioral change transfer across tasks?

Three sub-tasks with increasing difficulty:

| Sub-task | Question | Difficulty | Why |
|----------|----------|------------|-----|
| **SCT-Decision** | Do allocation decisions systematically diverge after social interaction? | Easiest | Seed memory alone could explain this |
| **SCT-Prediction** | Do agents form group-specific other-models (in-group > out-group prediction)? | Medium | Requires social interaction to form mental models of others |
| **SCT-Transfer** | Does divergence transfer to unrelated tasks (conformity, personality)? | Hardest, most valuable | If yes, changes go beyond surface opinion |

**Why this is new:** Existing work either tests static social cognition (ToMBench, CogMir) or runs large-scale social simulations (OASIS, AgentSociety). Nobody has defined "does social interaction produce transferable behavioral change" as a task.

### C2: SocCogBench Evaluation Benchmark (Section 4)

Three-layer measurement battery:

**Layer 1 -- Task-Specific Behavioral Metrics:**
- 5-dim resource allocation vector (pre/post)
- Prediction accuracy of others' allocations (in-group vs out-group MAE)

**Layer 2 -- Cross-Task Transfer Metrics:**
- Conformity test (CogMir/Asch-inspired, 10 MCQ + majority pressure, unrelated domain)
- BFI-10 personality instrument (pre/post drift, from PsychoBench)
- Cross-prompt consistency (same question, 3 phrasings)

**Layer 3 -- Causal Decomposition Metrics:**
- 5-condition effect size comparison (seed vs interaction vs structure vs baseline noise)
- Cohen's d + 95% bootstrap CI

SocCogBench is the first benchmark specifically measuring "socially induced cross-task behavioral transfer." Others can use it to evaluate their own agent architectures.

### C3: Methodology -- SRM + FSIP + S1/S2 (Section 5)

**Component 1: Social-Relational Memory Protocol (SRM)**
- Hypergraph-structured memory: agents as entity nodes, interactions as hyperedges
- RRF (Reciprocal Rank Fusion) retrieval combining graph traversal + semantic search
- Validated >90% field-level extraction consistency in pilot

**Component 2: Factorial Social Interaction Protocol (FSIP)**
- 5 conditions crossing connectivity x seed x interaction
- Sub-topic rotation (prevents 30-round content exhaustion)
- Network-topology-aware pair selection algorithm

**Component 3: Dual-Process Ablation (S1/S2)**
- S1: direct response
- S2: structured deliberation using fork-based ReasoningGraph (position trajectory, relationship history, contradiction detection, convergence trend analysis)
- Tests whether deliberation modulates divergence rate and cross-task transfer strength

---

## Architecture

CogClaw integrates techniques from five reference codebases into a unified experiment pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│              CogClaw Experiment Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ HypergraphMem│  │ ReasoningGraph│ │ ExperienceLibrary  │   │
│  │ (memory.py)  │  │ (reasoning.py)│ │ (experience.py)    │   │
│  │              │  │              │  │                    │   │
│  │ • NetworkX   │  │ • Fork from  │  │ • Advantage-       │   │
│  │   DiGraph    │  │   memory     │  │   weighted         │   │
│  │ • Entity +   │  │ • Structured │  │   retrieval        │   │
│  │   hyperedge  │  │   context    │  │ • Token Jaccard    │   │
│  │   nodes      │  │   builder    │  │   dedup            │   │
│  │ • RRF        │  │ • Trend      │  │ • EMA advantage    │   │
│  │   retrieval  │  │   detection  │  │   updates          │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘   │
│         │                 │                    │               │
│         └────────┬────────┴────────────────────┘               │
│                  ▼                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent (agent.py)                                         │  │
│  │ • Identical initialization (no personality/stance/bias)  │  │
│  │ • respond() — S1 direct response                        │  │
│  │ • respond_s2() — S2 structured deliberation              │  │
│  │ • answer_probe() — measurement interface                 │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐  │
│  │ Orchestrator (orchestrator.py)                            │  │
│  │ • Pre-test battery → 30 rounds → Post-test battery       │  │
│  │ • Periodic probes every 5 rounds                         │  │
│  │ • Memory extraction after each conversation              │  │
│  │ • Incremental save (JSON per round)                      │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐  │
│  │ SocCogBench Probes (probes.py)                            │  │
│  │ • Allocation task (5-dim)    • Prediction task            │  │
│  │ • Conformity test (10 MCQ)   • BFI-10 personality         │  │
│  │ • Belief/common ground probes • Consistency check          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Adapted from:**

| Source | What We Took |
|--------|-------------|
| **HyperGraphRAG** (NeurIPS 2025) | Hypergraph storage model, RRF retrieval, entity merging |
| **FutureMap** | Fork-based ReasoningGraph, structured context builder, ExperienceLibrary with advantage-weighted retrieval |
| **KnowledgeClaw** | Entity extraction pipeline, ReACT multi-round reasoning pattern |
| **MiroFish** | Agent profile generation patterns, real-time graph memory updates |
| **OASIS** | Interview/probe action pattern, trace logging for reproducibility |

---

## Experimental Design

### 5 Conditions (x 2 scenarios x repeats)

| # | Condition | Network | Seed | Interaction | Repeats |
|---|-----------|---------|------|-------------|---------|
| 1 | Open+Seed | All-to-all | Same | 30 rounds | 7 |
| 2 | Segreg+AsymSeed | Two groups | Different | 30 rounds | 10+ |
| 3 | Segreg+NoSeed | Two groups | None | 30 rounds | 5 |
| 4 | SeedOnly+NoInteract | N/A | Different | None | 5 |
| 5 | NoSeed+NoInteract | N/A | None | None | 3 |

**Causal decomposition:**
- (5): Initialization noise baseline
- (4) vs (5): Pure seed effect
- (2) vs (4): Social interaction effect (given seed)
- (2) vs (3): Seed effect (given interaction)
- (2) vs (1): Segregation effect
- All conditions subtract (5) baseline variance

### 2 Scenarios
- **Greenfield**: $10M community grant, 5 investment areas
- **Riverton**: $5M AI transition budget, 5 budget areas

---

## Project Structure

```
cogsociety/
├── cogclaw/                         # Core pipeline package
│   ├── config.py                    # Configuration (API params, model strategy, paths)
│   ├── llm.py                       # Multi-provider LLM client (6 providers)
│   ├── agent.py                     # Agent class with HypergraphMemory + S2 reasoning
│   ├── memory.py                    # HypergraphMemory (graph + vector + RRF) + extraction
│   ├── reasoning.py                 # Fork-based ReasoningGraph for S2 deliberation
│   ├── experience.py                # Advantage-weighted ExperienceLibrary
│   ├── probes.py                    # SocCogBench measurement battery
│   ├── network.py                   # Network topology (all-to-all, segregated, none)
│   ├── orchestrator.py              # Conversation orchestrator with pre/post measurement
│   └── conditions.py                # 5-condition factorial design factory
├── scenarios/                       # Scenario prompts and measurement instruments
│   ├── greenfield.py                # Community grant (context, seeds A/B, probes, allocation)
│   └── riverton.py                  # AI transition (context, seeds A/B, probes, allocation)
├── scripts/
│   └── serve_local_models.sh        # vLLM launcher for local models (GPU 6/7)
├── configs/                         # Generated experiment configs (5 conditions x 2 scenarios)
├── data/                            # Experiment output data
├── benchmark_datasets/              # Downloaded reference datasets
│   ├── PsychoBench/                 # BFI personality instruments (Layer 2)
│   ├── CogMir/                      # Conformity/herd effect items (Layer 2)
│   ├── ToMBench/                    # Theory of Mind reference
│   └── CogBench/                    # Cognitive psychology (citation)
├── FutureMap-main/                  # Reference: reasoning graph, experience library
├── HyperGraphRAG/                   # Reference: hypergraph storage, RRF retrieval
├── KnowledgeClaw/                   # Reference: entity extraction, ReACT reasoning
├── MiroFish-main/                   # Reference: agent profiles, graph memory updates
├── oasis/                           # Reference: interview actions, trace logging
├── run_experiment.py                # Main experiment runner
├── toy_test.py                      # Quick validation (4 agents x 5 rounds)
├── requirements.txt                 # Python dependencies
├── working_plan_v3.md               # Canonical research plan (FROZEN)
└── benchmark_analysis.md            # Dataset analysis and evaluation strategy
```

---

## Quick Start

### Prerequisites

```bash
conda activate cogclaw
pip install -r requirements.txt
```

### API Keys

Place API key files in `APIkeys/`:

| File | Provider | When needed |
|------|----------|-------------|
| `api_key_openai.txt` | OpenAI | Cross-model validation |
| `api_key_deepseek.txt` | DeepSeek | Cross-model check |
| `api_key_gemini.txt` | Google Gemini | Fifth model family |
| `api_key_grok.txt` | xAI Grok | Backup |

Or set environment variables: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `GROK_API_KEY`.

### Local Models (vLLM)

Models stored at `/mnt/ssd1/yandan/transformer_cache/`:

```bash
# Serve Qwen3-122B-FP8 on GPU 6 (port 8000)
./scripts/serve_local_models.sh qwen

# Serve Llama4-Scout on GPU 7 (port 8001)
./scripts/serve_local_models.sh llama

# Both simultaneously
./scripts/serve_local_models.sh both
```

### Run Toy Test

```bash
# 4 agents, 5 rounds, local Qwen (default)
python toy_test.py

# Or specify a model explicitly
python toy_test.py --model qwen3-122b-fp8
python toy_test.py --model deepseek-chat

# Memory extraction consistency validation (target: >90%)
python toy_test.py --consistency-test
```

### Run Full Experiment

```bash
# Single condition
python run_experiment.py --condition 2 --scenario greenfield

# Cross-model validation
python run_experiment.py --condition 2 --scenario greenfield --model gpt-4o-mini
python run_experiment.py --condition 2 --scenario greenfield --model deepseek-chat

# With S2 deliberate reasoning ablation
python run_experiment.py --condition 2 --scenario greenfield --s2

# All 5 conditions for one scenario
python run_experiment.py --all --scenario greenfield

# Generate all config files
python run_experiment.py --generate-configs
```

### Output Structure

```
data/<run_name>/
├── config.json                # Experiment configuration
├── pre_measurements.json      # SocCogBench pre-test (BFI-10, conformity, allocation)
├── rounds/
│   ├── round_001.json         # Conversations + probe responses per round
│   ├── round_005.json         # Includes periodic belief/common-ground probes
│   └── ...
├── memories/
│   ├── Alex.json              # Agent's hypergraph memory (SRM output)
│   └── ...
├── post_measurements.json     # SocCogBench post-test (all instruments)
└── summary.json               # Run summary statistics
```

---

## Model Strategy

Six model families across local and API providers:

| Model | Type | Port/Endpoint | Role | Cost |
|-------|------|---------------|------|------|
| **Qwen3-122B-FP8** | Local vLLM (GPU 6) | `localhost:8000` | Primary (主力) | Free |
| **Llama4-Scout** | Local vLLM (GPU 7) | `localhost:8001` | Second local family | Free |
| **GPT-4o-mini** | OpenAI API | Direct | Cloud cross-model | ~$30-50 |
| **DeepSeek-chat** | DeepSeek API | Direct | Cheap cloud check | ~$10-20 |
| **Gemini-2.0-flash** | Google API | Direct | Fifth model family | Minimal |
| **Grok-2** | xAI API | Direct | Backup | -- |

Auto-routing: model name in config determines which provider client is used.

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Temperature (conversations) | 0.7 |
| Temperature (probes/tests) | 0.0 |
| Max tokens (conversations) | 1024 |
| Max tokens (probes) | 512 |
| Max tokens (allocation) | 256 |
| Agents per run | 8 (identically initialized) |
| Rounds per run | 30 |

## SocCogBench Measurement Battery

| Layer | When | Instrument | Source |
|-------|------|-----------|--------|
| 1 | Pre/Post | 5-dim allocation task | Own design |
| 1 | Post | Prediction of others' allocations | Own design |
| 2 | Pre/Post | BFI-10 personality instrument | PsychoBench |
| 2 | Pre/Post | Conformity test (10 MCQ + majority pressure) | CogMir |
| 2 | Post | Cross-prompt consistency (3 phrasings) | Own design |
| 3 | All conditions | Effect sizes + 95% CI | Statistical |
| -- | Every 5 rounds | Belief probe | Own design |
| -- | Every 5 rounds | Common ground probe | Own design |

---

## Schedule

| Week | Dates | Deliverable |
|------|-------|-------------|
| 1 | Mar 18-24 | Pipeline + validated extraction consistency |
| 2 | Mar 25-31 | SocCogBench measurement system + pilot data + power analysis |
| 3 | Apr 1-7 | All experimental data collected |
| 4 | Apr 8-14 | All figures + tables + statistical results |
| 5 | Apr 15-21 | Complete paper draft (SCT + SocCogBench + Methodology) |
| 6 | Apr 22-28 | Revision + formatting |
| 7 | Apr 29-May 6 | Abstract (May 4) + Full paper (May 6) |

## License

Research code. To be released under MIT license upon submission.
