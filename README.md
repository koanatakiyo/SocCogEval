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
- Standardized extraction: fixed prompt converts each conversation to 3-facet JSON (episodic / social / self-relevant)
- Validated >90% field-level extraction consistency in pilot
- Reusable template for any multi-agent simulation needing structured memory

**Component 2: Factorial Social Interaction Protocol (FSIP)**
- 5 conditions crossing connectivity x seed x interaction
- Sub-topic rotation (prevents 30-round content exhaustion)
- Network-topology-aware pair selection algorithm
- Reusable factorial design template for multi-agent social experiments

**Component 3: Dual-Process Ablation (S1/S2)**
- S1: direct response
- S2: internal deliberation before responding ("How does the other agent see me? How might they react?")
- Tests whether deliberation modulates divergence rate and cross-task transfer strength
- Connects to dual-process theory from cognitive science

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
cogclaw/
├── cogclaw/                         # Core pipeline package
│   ├── config.py                    # Configuration (API params, model strategy)
│   ├── llm.py                       # Multi-provider LLM client (Qwen/GPT/DeepSeek/Llama/apiyi)
│   ├── agent.py                     # Agent class + SocialRelationalMemory (SRM)
│   ├── memory.py                    # Memory extraction (LLM -> JSON) + consistency validation
│   ├── network.py                   # Network topology (all-to-all, segregated, none)
│   ├── orchestrator.py              # Conversation orchestrator (FSIP implementation)
│   └── conditions.py                # 5-condition factorial design factory
├── scenarios/                       # Scenario prompts and measurement instruments
│   ├── greenfield.py                # Community grant (context, seeds A/B, probes, allocation)
│   └── riverton.py                  # AI transition (context, seeds A/B, probes, allocation)
├── configs/                         # Generated experiment configs (5 conditions x 2 scenarios)
├── data/                            # Experiment output data
├── benchmark_datasets/              # Downloaded reference datasets
│   ├── PsychoBench/                 # BFI personality instruments (Layer 2)
│   ├── CogMir/                      # Conformity/herd effect items (Layer 2)
│   ├── ToMBench/                    # Theory of Mind reference
│   └── CogBench/                    # Cognitive psychology (citation)
├── run_experiment.py                # Main experiment runner
├── toy_test.py                      # Quick validation (4 agents x 5 rounds)
├── working_plan_v3.md               # Canonical research plan (FROZEN)
└── benchmark_analysis.md            # Dataset analysis and evaluation strategy
```

### Reference Codebases

| Directory | Project | Relevance |
|-----------|---------|-----------|
| `FutureMap-main/` | Prediction engine with LLM reasoning | LLM client patterns, JSON repair |
| `KnowledgeClaw/` | Knowledge graph engine | Storage, retrieval, graph patterns |
| `MiroFish-main/` | Multi-agent simulation orchestrator | Profile generation, OASIS integration |
| `oasis/` | OASIS social media simulation | Agent/environment/network patterns |
| `openclaw-main/` | AI assistant framework | Provider/model selection patterns |
| `HyperGraphRAG/` | Hypergraph-based RAG (NeurIPS 2025) | Future: hypergraph memory (Paper 2) |

---

## Quick Start

### Prerequisites

```bash
conda activate cogclaw
pip install -r requirements.txt
```

### API Keys

Place API key files in the project root (`/home/yandan/cogclaw/`):

| File | What | When needed |
|------|------|-------------|
| *(none needed)* | Qwen 3 122B + Llama 4 Scout run locally | Always (local models) |
| `api_key_openai.txt` | OpenAI API key for GPT-5.3 mini | Cross-model validation |
| `api_key_deepseek.txt` | DeepSeek API key | Cross-model check |
| `api_key_apiyi.txt` | Apiyi proxy key (for Gemini 3.1 Flash / Grok) | Fifth model + backup |

Or set environment variables instead:

```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export APIYI_API_KEY="sk-..."

# Local model servers (defaults shown):
export QWEN_BASE_URL="http://localhost:8000/v1"    # vLLM / Ollama for Qwen
export LLAMA_BASE_URL="http://localhost:8000/v1"    # vLLM / Ollama for Llama
```

### Run Toy Test

```bash
# 4 agents, 5 rounds, local Qwen (default)
python toy_test.py

# Or specify a model explicitly
python toy_test.py --model qwen3-122b
python toy_test.py --model deepseek-chat

# Memory extraction consistency validation (target: >90%)
python toy_test.py --consistency-test
```

### Run Full Experiment

```bash
# Single condition (uses primary model by default)
python run_experiment.py --condition 2 --scenario greenfield

# Cross-model validation run
python run_experiment.py --condition 2 --scenario greenfield --model gpt-5.3-mini
python run_experiment.py --condition 2 --scenario greenfield --model deepseek-chat
python run_experiment.py --condition 2 --scenario greenfield --model llama-4-scout

# Override round count (e.g., for pilot)
python run_experiment.py --condition 2 --scenario greenfield --n-rounds 10

# All 5 conditions for one scenario
python run_experiment.py --all --scenario greenfield

# With S2 deliberate reasoning ablation
python run_experiment.py --condition 2 --scenario greenfield --s2

# Generate all config files
python run_experiment.py --generate-configs

# Run from config file
python run_experiment.py --config-file configs/condition_2_segreg_asym_seed_greenfield.json
```

### Output Structure

```
data/<run_name>/
├── config.json                # Experiment configuration
├── rounds/
│   ├── round_001.json         # Conversations + probe responses per round
│   ├── round_002.json
│   └── ...
├── memories/
│   ├── Alex.json              # Agent's social-relational memory (SRM output)
│   ├── Blake.json
│   └── ...
├── post_measurements.json     # SocCogBench: allocation, prediction, belief, conformity
└── summary.json               # Run summary statistics
```

---

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

## Model Strategy

Five model families support cross-model robustness claims:

| Model | Type | Role | Routing | Cost |
|-------|------|------|---------|------|
| **Qwen 3 122B** | Local | Primary (主力). All pilot + main experiment | `localhost:8000/v1` | Free |
| **Llama 4 Scout** | Local | Second local model family | `localhost:8000/v1` | Free |
| **GPT-5.3 mini** | API | Cloud cross-model validation | OpenAI API | ~$30-50 |
| **DeepSeek v3.2** | API | Cloud cross-model check | DeepSeek API | ~$10-20 |
| **Gemini 3.1 Flash** | API | Fifth model family | Apiyi proxy | Minimal |
| Grok | Backup | If needed | Apiyi proxy | -- |

Key advantage: Two local models (Qwen + Llama) = unlimited repeats, no budget constraints. Main condition goes to 10+ repeats.

## Agent Design

Agents are **identically initialized** with:
- Same base model and prompting template
- Same role structure (community member / employee)
- Brief name and role label
- Behavioral instruction: "Express your genuine thoughts honestly"
- **NO** personality, stance, ideology, or knowledge base injected

After each conversation, **Social-Relational Memory (SRM)** extraction:
```json
{
  "episodic": { "event": "...", "context": "...", "my_response": "..." },
  "social": { "participants": [...], "their_stances": {...}, "group_dynamics": "..." },
  "self_relevant": { "how_they_see_me": "...", "my_evolution": "..." }
}
```

## SocCogBench Measurement Battery

| Layer | When | Instrument | Source |
|-------|------|-----------|--------|
| 1 | Post | 5-dim allocation task | Own design |
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
