# Benchmark Datasets: What You Have, What's Useful, What's Not

## Downloaded Datasets (in /home/claude/benchmark_datasets/)

| Dataset | Source | Size | Status |
|---|---|---|---|
| CogMir | ICLR 2025 | 7 bias types, prompt templates, MCQ datasets | ✅ Downloaded |
| ToMBench | ACL 2024 | 2,860 items across 15 sheets, bilingual (CN/EN) | ✅ Downloaded |
| CogBench | ICML 2024 | 7 cognitive psych experiments, 35+ LLM results | ✅ Downloaded |
| PsychoBench | ICLR 2024 | 14 questionnaires (BFI, Empathy, DTDD, etc.) | ✅ Downloaded |
| HyperGraphRAG | NeurIPS 2025 | Code + evaluation datasets | ✅ Downloaded |

---

## HONEST ASSESSMENT: Which datasets matter for YOUR paper?

### 🔴 NONE of these are your evaluation benchmark.

Your paper is NOT:
- Testing if LLMs have Theory of Mind (→ ToMBench)
- Testing if LLMs have human-like cognitive biases (→ CogMir, CogBench)
- Testing if LLMs match human personality profiles (→ PsychoBench)
- Testing if LLMs reflect cultural attitudes (→ SocioBench)

Your paper IS:
- Testing if social information structure causes belief divergence in identical LLM agents.

These are fundamentally different research questions. 
Trying to "evaluate on existing benchmarks" would be like testing 
a cancer drug using a diabetes diagnostic tool.

### What these datasets ARE useful for:

1. **Methodology reference** — how others structured their inputs/outputs
2. **Related work** — what patterns are already documented
3. **Discussion fodder** — "our findings parallel X from CogMir/CogBench"
4. **Possible pre/post tests** — run a SUBSET on your agents before and after social interaction

---

## Dataset-by-Dataset Analysis: What's Inside

### 1. CogMir (ICLR 2025) — Multi-Agent Cognitive Bias Framework

**What it contains:**
- 7 cognitive bias experiment templates:
  - Herd Effect (conformity under group pressure)
  - Authority Effect (conformity to authority)
  - Confirmation Bias
  - Halo Effect
  - Gambler's Fallacy
  - Ben Franklin Effect
  - Rumor Chain Effect
- Core datasets:
  - Known MCQ: trivial questions ("What color is an apple?")
  - Unknown MCQ: unanswerable questions ("Weather in Brooklyn on 2027/3/25?")
  - CogIdentity: agent profiles with personality traits
  - CogScene: scenario templates with placeholders
  - CogAction: simple actions ("lend a pen")
  - Inform: narrative stories for rumor experiments

**Input format:** Prompt templates with placeholders like [NUMBER], [Known/Unknown MCQ], [SCENARIO]
**Output format:** "Answer: A/B. Explanation: ***"

**Relevance to your paper: MEDIUM**
- The Herd Effect experiment is directly relevant — it tests conformity under group pressure
- You could run a simplified version as a PRE-TEST: before social interaction, 
  how susceptible is each agent to conformity? Then after 30 rounds, test again.
- This would give you a concrete metric: "social interaction changed conformity susceptibility"

**Practical suggestion:** 
Extract 10 Known MCQ + 10 Unknown MCQ items from CogMir.
Before round 1 and after round 30, give each agent a mini herd-effect test:
"7 others said B. What do you answer?"
Compare conformity rates pre vs post, segregated vs open.
This is ~20 extra API calls per agent per measurement point. Very cheap.


### 2. ToMBench (ACL 2024) — Theory of Mind Benchmark

**What it contains:**
- 2,860 items across 15 task types (Chinese + English bilingual):
  - Unexpected Outcome Test (302 items) — emotion inference
  - False Belief Task (601 items) — classic Sally-Anne style
  - Strange Story Task (408 items) — irony, sarcasm, misunderstanding
  - Faux-pas Recognition (561 items) — social inappropriateness detection
  - Hinting Task (104 items) — indirect communication
  - Persuasion Story (101 items) — desire/intention understanding
  - Hidden/Moral/Discrepant Emotions
  - Multiple/Discrepant Desires and Intentions
  - Prediction of Actions
  - Knowledge-Attention links

**Input format:** Story + MCQ question (A/B/C/D options)
**Output format:** Letter choice

**Relevance to your paper: LOW-MEDIUM**
- Your paper isn't about whether LLMs have ToM
- BUT: you could use a tiny subset (e.g., 10 False Belief items) as a 
  sanity check to show your agents have baseline ToM ability
- More relevant: the "Prediction of Actions" sheet (21 items) is conceptually 
  close to your pairwise prediction task
- The bilingual nature could be useful if you ever extend to multilingual conditions

**Practical suggestion:**
Don't run the full benchmark. Pick 5-10 items from "Prediction of Actions" and 
use them as a template for designing your own prediction prompts.


### 3. CogBench (ICML 2024) — Cognitive Psychology Experiments

**What it contains:**
- 7 cognitive experiments with human baselines:
  - Probabilistic Reasoning (Bayesian updating)
  - Horizon Task (exploration vs exploitation)
  - Instrumental Learning (reward learning)
  - BART (risk taking)
  - Temporal Discounting
  - Two-Step Task (model-based vs model-free)
  - Iterated Prisoner's Dilemma (cooperation)
- 10 behavioral metrics extracted from these experiments
- Data from 35+ LLMs already tested

**Input format:** Multi-round decision scenarios with structured prompts
**Output format:** Numerical choices, analyzed via computational cognitive models

**Relevance to your paper: LOW (but methodologically inspiring)**
- These experiments test individual decision-making, not social dynamics
- The Prisoner's Dilemma data IS social but it's dyadic, not group-level
- HOWEVER: the methodology of "phenotyping LLM behavior via psych experiments" 
  is exactly what you're doing at the social level

**Practical suggestion:**
Don't run any CogBench experiments. Cite it as methodological precedent:
"Following CogBench's approach of phenotyping LLM behavior through 
cognitive experiments, we apply social psychology paradigms to multi-agent settings."


### 4. PsychoBench (ICLR 2024) — Psychological Questionnaires

**What it contains:**
- 14 standardized psychological questionnaires:
  - Empathy (10 items)
  - BFI / Big Five Inventory (44 items)
  - BSRI / Bem Sex Role Inventory (60 items)
  - EPQ-R / Eysenck Personality (100 items)
  - LMS / Loneliness (9 items)
  - DTDD / Dark Triad (12 items)
  - ECR-R / Attachment Style (36 items)
  - GSE / Self-Efficacy (10 items)
  - ICB / Impulsivity (8 items)
  - LOT-R / Optimism (10 items)
  - EIS + WLEIS / Emotional Intelligence (33 + 16 items)
  - CABIN / 164 items
  - 16P / 16 Personalities (60 items)

**Input format:** Likert-scale self-report questions
**Output format:** Score on scale (1-5 typically)

**Relevance to your paper: MEDIUM-HIGH (as pre/post measurement)**
- You explicitly don't inject personality into your agents
- BUT: you could run a SHORT personality questionnaire (e.g., BFI-10, the 10-item 
  version of Big Five) on each agent BEFORE and AFTER 30 rounds of social interaction
- If personality scores CHANGE after social interaction, and change DIFFERENTLY 
  across conditions, that's extremely powerful evidence for your thesis
- This would be a concrete, validated, quantitative measure of "socially induced 
  personality drift" — much harder for reviewers to dismiss than embedding similarity

**Practical suggestion: THIS IS THE SINGLE MOST USEFUL THING FROM ALL THESE DATASETS.**

Do this:
1. Extract BFI-10 (only 10 items) from PsychoBench
2. Administer to each agent at round 0 and round 30
3. Compare: do personality profiles diverge across conditions?
4. This gives you Figure X: "Personality Profile Drift Under Different Social Structures"

Cost: 10 questions × 8 agents × 2 timepoints × 4 conditions = 640 extra API calls total. Trivial.

---

## RECOMMENDED EVALUATION STRATEGY

### Your evaluation has THREE layers:

**Layer 1: Your own metrics (PRIMARY — this is what the paper is about)**
- Allocation task divergence
- Belief embedding trajectory
- Belief entropy + inter-group KL
- Prediction accuracy
- Belief revision events

**Layer 2: Validated psychological instruments (SECONDARY — borrowed from PsychoBench)**
- BFI-10 pre/post: personality profile drift
- Optional: Empathy scale pre/post (10 items from PsychoBench)

**Layer 3: Social cognition probes (TERTIARY — inspired by CogMir)**
- Mini herd-effect test pre/post (10 items from CogMir Known MCQ)
- Conformity rate change across conditions

### What this gives you that you didn't have before:

| Before | After |
|---|---|
| "Belief embeddings diverged" (soft) | "Allocation decisions diverged" (hard) |
| "Self-descriptions changed" (soft) | "BFI-10 personality scores shifted" (validated instrument) |
| "Agents formed different views" (vague) | "Conformity susceptibility changed" (replicable from CogMir) |

The combination of your own metrics + validated instruments + behavioral tasks 
makes the measurement argument much more robust.

---

## What NOT to do

❌ Don't try to run the full ToMBench (2,860 items) on your agents
❌ Don't try to run CogBench's 7 experiments 
❌ Don't try to match SocioBench's 481K respondent records
❌ Don't try to run all 14 PsychoBench questionnaires
❌ Don't redesign your experiment to be a "benchmark evaluation paper"

Your paper is an EXPERIMENT, not a benchmark evaluation. 
The datasets are INSTRUMENTS you borrow a few items from, not targets you evaluate against.

---

## Files Available

```
/home/yandan/cogclaw/benchmark_datasets/
├── CogMir/                     # Cognitive bias framework (github.com/XuanL17/CogMir)
│   └── CogMir_Data/
│       ├── Core_Datasets/      # Known MCQ (99), Unknown MCQ (99), identities, scenarios, actions
│       └── Experimental_Prompts_Example/  # Templates for 7 bias types (incl. Herd_Effect)
├── ToMBench/                   # Theory of Mind benchmark (github.com/zhchen18/ToMBench)
│   └── data/                   # 2,860 items across 20 JSONL files (Prediction of Actions: 20 items)
├── CogBench/                   # Cognitive psychology experiments (github.com/juliancodaforno/CogBench)
│   └── Experiments/            # 7 experiments with LLM data
└── PsychoBench/                # Psychological questionnaires (github.com/CUHK-ARISE/PsychoBench)
    └── questionnaires.json     # 14 questionnaires, 562+ items total (BFI: 44, Empathy: 10, etc.)
```
