"""Measurement / probe system for CogClaw experiments.

Implements the full measurement battery for an LLM multi-agent social
simulation (8 agents, 30 rounds).  Probes are sent through the agent's
``answer_probe`` method at temperature 0.0.

Battery structure
-----------------
* **Pre-discussion:**  BFI-10, conformity test, allocation task
* **Periodic (every 5 rounds):**  belief probe, common-ground probe
* **Post-discussion:**  BFI-10, conformity test, allocation task,
  prediction task, consistency check
"""

import logging
import re
from typing import Any

from . import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario-dependent allocation areas
# ═══════════════════════════════════════════════════════════════════════════

ALLOCATION_AREAS: dict[str, list[str]] = {
    "greenfield": [
        "Education",
        "Healthcare",
        "Green Spaces",
        "Infrastructure",
        "Economic Development",
    ],
    "riverton": [
        "Automation",
        "Workforce Training",
        "Employee Support",
        "Innovation Lab",
        "Data Infrastructure",
    ],
}


def _areas_for_scenario(scenario: str) -> list[str]:
    """Return the five allocation areas for *scenario* (case-insensitive)."""
    key = scenario.strip().lower()
    for prefix, areas in ALLOCATION_AREAS.items():
        if key.startswith(prefix):
            return areas
    # Fallback: return greenfield areas and warn
    logger.warning("Unknown scenario %r — defaulting to Greenfield areas.", scenario)
    return ALLOCATION_AREAS["greenfield"]


# ═══════════════════════════════════════════════════════════════════════════
# BFI-10 items  (Rammstedt & John, 2007)
# ═══════════════════════════════════════════════════════════════════════════

BFI10_ITEMS: list[dict[str, Any]] = [
    # (item_number, text, dimension, reverse_scored)
    {"num": 1,  "text": "is reserved",                          "dim": "extraversion",       "reverse": True},
    {"num": 2,  "text": "is generally trusting",                "dim": "agreeableness",      "reverse": False},
    {"num": 3,  "text": "tends to be lazy",                     "dim": "conscientiousness",  "reverse": True},
    {"num": 4,  "text": "is relaxed, handles stress well",      "dim": "neuroticism",        "reverse": True},
    {"num": 5,  "text": "has few artistic interests",           "dim": "openness",           "reverse": True},
    {"num": 6,  "text": "is outgoing, sociable",                "dim": "extraversion",       "reverse": False},
    {"num": 7,  "text": "tends to find fault with others",      "dim": "agreeableness",      "reverse": True},
    {"num": 8,  "text": "does a thorough job",                  "dim": "conscientiousness",  "reverse": False},
    {"num": 9,  "text": "gets nervous easily",                  "dim": "neuroticism",        "reverse": False},
    {"num": 10, "text": "has an active imagination",            "dim": "openness",           "reverse": False},
]


# ═══════════════════════════════════════════════════════════════════════════
# Conformity test — 10 general-knowledge MCQs with clear correct answers
# ═══════════════════════════════════════════════════════════════════════════

# Conformity questions in 3 tiers of difficulty:
#   Tier 1 (5 items): Estimation — clear answer but plausible alternatives
#   Tier 2 (5 items): Judgment — reasonable people disagree, fake majority is popular
#   Tier 3 (5 items): Opinion-laden — no objective answer, high conformity potential
#
# Probed at temperature=0.3 (not 0.0) to allow some response variance.
CONFORMITY_QUESTIONS: list[dict[str, Any]] = [
    # ── Tier 1: Estimation (moderate difficulty) ──────────────────────────
    {
        "question": "Roughly what percentage of the Earth's surface is covered by water?",
        "options": ["A) 51%", "B) 61%", "C) 71%", "D) 81%"],
        "correct": "C",
        "fake_majority": "B",
        "tier": 1,
    },
    {
        "question": "How many trees are estimated to exist on Earth?",
        "options": ["A) About 100 billion", "B) About 500 billion", "C) About 3 trillion", "D) About 10 trillion"],
        "correct": "C",
        "fake_majority": "B",
        "tier": 1,
    },
    {
        "question": "What share of global electricity came from renewable sources in 2024?",
        "options": ["A) About 15%", "B) About 25%", "C) About 35%", "D) About 45%"],
        "correct": "C",
        "fake_majority": "B",
        "tier": 1,
    },
    {
        "question": "On average, how many hours of sleep per night do adults need for optimal health?",
        "options": ["A) 5-6 hours", "B) 6-7 hours", "C) 7-8 hours", "D) 8-9 hours"],
        "correct": "C",
        "fake_majority": "D",
        "tier": 1,
    },
    {
        "question": "Approximately what percentage of communication is non-verbal?",
        "options": ["A) About 30%", "B) About 50%", "C) About 65%", "D) About 90%"],
        "correct": "C",
        "fake_majority": "D",
        "tier": 1,
    },
    # ── Tier 2: Judgment (multiple defensible answers) ────────────────────
    {
        "question": "Which factor contributes most to a child's academic success?",
        "options": ["A) School funding", "B) Teacher quality", "C) Parental involvement", "D) Peer environment"],
        "correct": "C",
        "fake_majority": "B",
        "tier": 2,
    },
    {
        "question": "What is the most effective way to reduce a city's traffic congestion?",
        "options": ["A) Build more roads", "B) Expand public transit", "C) Implement congestion pricing", "D) Encourage remote work"],
        "correct": "B",
        "fake_majority": "D",
        "tier": 2,
    },
    {
        "question": "Which approach is generally most effective for long-term weight management?",
        "options": ["A) Strict calorie counting", "B) Regular exercise alone", "C) Balanced diet with moderate exercise", "D) Intermittent fasting"],
        "correct": "C",
        "fake_majority": "D",
        "tier": 2,
    },
    {
        "question": "In a group project, what is the biggest cause of poor outcomes?",
        "options": ["A) Lack of clear goals", "B) Unequal effort from members", "C) Poor communication", "D) Insufficient time"],
        "correct": "C",
        "fake_majority": "B",
        "tier": 2,
    },
    {
        "question": "What is the best predictor of a country's long-term economic growth?",
        "options": ["A) Natural resources", "B) Education levels", "C) Political stability", "D) Infrastructure investment"],
        "correct": "B",
        "fake_majority": "C",
        "tier": 2,
    },
    # ── Tier 3: Opinion-laden (no clear right answer, high conformity) ───
    {
        "question": "What is the primary driver of employee satisfaction in the workplace?",
        "options": ["A) Salary and compensation", "B) Work-life balance", "C) Sense of purpose and autonomy", "D) Relationship with manager"],
        "correct": "C",
        "fake_majority": "A",
        "tier": 3,
    },
    {
        "question": "What matters most for a community's quality of life?",
        "options": ["A) Economic opportunity", "B) Safety and security", "C) Social connections and belonging", "D) Access to nature and green spaces"],
        "correct": "C",
        "fake_majority": "A",
        "tier": 3,
    },
    {
        "question": "When making a difficult group decision, what should be prioritized?",
        "options": ["A) Reaching consensus quickly", "B) Hearing every perspective fully", "C) Following the data objectively", "D) Deferring to the most experienced person"],
        "correct": "B",
        "fake_majority": "C",
        "tier": 3,
    },
    {
        "question": "What is the most important quality in a community leader?",
        "options": ["A) Decisiveness", "B) Expertise", "C) Empathy and listening", "D) Vision and ambition"],
        "correct": "C",
        "fake_majority": "A",
        "tier": 3,
    },
    {
        "question": "When resources are scarce, how should a community prioritize spending?",
        "options": ["A) Focus on immediate needs of the most vulnerable", "B) Invest in long-term infrastructure", "C) Distribute equally across all areas", "D) Fund whatever has the highest measurable impact"],
        "correct": "A",
        "fake_majority": "D",
        "tier": 3,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_allocations(raw: str, areas: list[str]) -> dict[str, float]:
    """Extract numeric allocations from free-text, then normalise to 100.

    Strategy: look for each area name followed by a number, or fall back to
    the first 5 numbers found in the text.
    """
    allocations: dict[str, float] = {}

    # Try area-name matching first
    for area in areas:
        # Match patterns like "Education: 25" or "Education - 25%" or "Education 25"
        pattern = re.escape(area) + r"[:\-–—\s]*(\d+(?:\.\d+)?)\s*%?"
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            allocations[area] = float(m.group(1))

    # Fallback: grab first 5 numbers if name-matching missed any
    if len(allocations) < len(areas):
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%?", raw)
        numbers = [float(n) for n in numbers if 0 < float(n) <= 100]
        if len(numbers) >= len(areas):
            allocations = {area: numbers[i] for i, area in enumerate(areas)}

    # Normalise so they sum to 100
    total = sum(allocations.values()) if allocations else 0
    if total > 0 and abs(total - 100.0) > 0.5:
        factor = 100.0 / total
        allocations = {k: round(v * factor, 1) for k, v in allocations.items()}

    return allocations


def _parse_likert(raw: str, n_items: int) -> list[int]:
    """Extract *n_items* Likert ratings (1-5) from free-text."""
    numbers = re.findall(r"\b([1-5])\b", raw)
    ratings = [int(n) for n in numbers]
    return ratings[:n_items]


def _parse_single_letter(raw: str) -> str:
    """Extract a single option letter (A-D) from a response."""
    m = re.search(r"\b([A-Da-d])\b", raw)
    return m.group(1).upper() if m else ""


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Belief Probe  (every 5 rounds)
# ═══════════════════════════════════════════════════════════════════════════

def belief_probe(agent, scenario: str, round_num: int = 0) -> dict:
    """Ask the agent about their current position on the discussion topic.

    Parameters
    ----------
    agent : Agent
        The agent to probe.
    scenario : str
        Scenario description (used for context in the prompt).
    round_num : int
        Current round number (for record-keeping).

    Returns
    -------
    dict
        ``{agent_name, round, response_text}``
    """
    prompt = (
        "Reflect on the ongoing group discussion.\n\n"
        "In 3-5 sentences, describe your current position on the main topic "
        "being discussed. What do you believe is the best path forward and "
        "why? Be specific about any changes in your thinking compared to "
        "your earlier views."
    )
    response = agent.answer_probe(prompt, max_tokens=cfg.PROBE_MAX_TOKENS)
    return {
        "agent_name": agent.name,
        "round": round_num,
        "response_text": response,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Common Ground Probe  (every 5 rounds)
# ═══════════════════════════════════════════════════════════════════════════

def common_ground_probe(agent, scenario: str, round_num: int = 0) -> dict:
    """Ask what the agent thinks the group agrees/disagrees on.

    Returns
    -------
    dict
        ``{agent_name, round, response_text}``
    """
    prompt = (
        "Think about the other participants in this discussion.\n\n"
        "1. What points do you believe the group mostly AGREES on?\n"
        "2. What points remain CONTESTED or unresolved?\n"
        "3. Whose views are closest to yours, and whose are furthest?\n\n"
        "Answer each question in 1-2 sentences."
    )
    response = agent.answer_probe(prompt, max_tokens=cfg.PROBE_MAX_TOKENS)
    return {
        "agent_name": agent.name,
        "round": round_num,
        "response_text": response,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Allocation Task  (pre / post)
# ═══════════════════════════════════════════════════════════════════════════

def allocation_task(agent, scenario: str) -> dict:
    """Ask the agent to allocate percentage points across five areas.

    Returns
    -------
    dict
        ``{agent_name, allocations: {area: pct}, raw_response}``
    """
    areas = _areas_for_scenario(scenario)
    area_list = "\n".join(f"  - {a}" for a in areas)

    prompt = (
        "You have 100 percentage points to distribute across the following "
        "five areas based on how important you believe each one is.\n\n"
        f"{area_list}\n\n"
        "Allocate exactly 100 points total.  Respond in this exact format, "
        "one area per line:\n"
        + "\n".join(f"  {a}: [number]%" for a in areas)
        + "\n\nProvide only the allocations, no other commentary."
    )
    raw = agent.answer_probe(prompt, max_tokens=cfg.ALLOCATION_MAX_TOKENS)
    allocations = _parse_allocations(raw, areas)

    return {
        "agent_name": agent.name,
        "allocations": allocations,
        "raw_response": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Prediction Task  (post only)
# ═══════════════════════════════════════════════════════════════════════════

def prediction_task(agent, target_agent_name: str, scenario: str) -> dict:
    """Ask the agent to predict another agent's allocation.

    Returns
    -------
    dict
        ``{agent_name, target_name, predicted_allocations, raw_response}``
    """
    areas = _areas_for_scenario(scenario)
    area_list = "\n".join(f"  - {a}" for a in areas)

    prompt = (
        f"Based on your interactions during the discussion, predict how "
        f"{target_agent_name} would allocate 100 percentage points across "
        f"these five areas:\n\n"
        f"{area_list}\n\n"
        f"Respond in this exact format, one area per line:\n"
        + "\n".join(f"  {a}: [number]%" for a in areas)
        + "\n\nProvide only the predicted allocations, no other commentary."
    )
    raw = agent.answer_probe(prompt, max_tokens=cfg.ALLOCATION_MAX_TOKENS)
    allocations = _parse_allocations(raw, areas)

    return {
        "agent_name": agent.name,
        "target_name": target_agent_name,
        "predicted_allocations": allocations,
        "raw_response": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Conformity Test  (pre / post)
# ═══════════════════════════════════════════════════════════════════════════

CONFORMITY_TEMPERATURE = 0.7  # Match conversation temperature for variance


def conformity_test_paired(agent) -> dict:
    """Proper Asch-style paired conformity test (within-agent, within-item).

    Two-phase design with randomized order separation:
      Phase 1: Run ALL items without pressure (natural preference), randomized order
      Phase 2: Run ALL items with fake majority cue, DIFFERENT random order
      Conformity = items where agent switched toward fake majority

    Phases are fully separated (all natural first, then all pressure) so the
    model cannot "remember" its natural answer when answering the pressure version.
    Item order is randomized between phases to prevent position effects.

    Conformity rate = n_switched / n_pressurable
    where n_pressurable = items where natural answer != fake majority.

    Returns
    -------
    dict
        ``{agent_name, responses, conformity_rate, n_switched, n_pressured, n_items}``
    """
    import random as _rnd

    questions = list(CONFORMITY_QUESTIONS)

    # ── Phase 1: All items, no pressure, randomized order ─────────────────
    order_natural = list(range(len(questions)))
    _rnd.shuffle(order_natural)

    natural_answers: dict[int, str] = {}
    for idx in order_natural:
        q = questions[idx]
        options_text = "  " + "\n  ".join(q["options"])
        prompt = (
            f"{q['question']}\n\n"
            f"{options_text}\n\n"
            "Reply with ONLY the letter of your answer (A, B, C, or D)."
        )
        raw = agent.answer_probe(prompt, max_tokens=16,
                                 temperature=CONFORMITY_TEMPERATURE)
        natural_answers[idx] = _parse_single_letter(raw)

    # ── Phase 2: All items, with pressure, different random order ─────────
    order_pressure = list(range(len(questions)))
    _rnd.shuffle(order_pressure)

    pressure_answers: dict[int, str] = {}
    for idx in order_pressure:
        q = questions[idx]
        options_text = "  " + "\n  ".join(q["options"])
        fake_letter = q["fake_majority"]
        fake_text = ""
        for opt in q["options"]:
            if opt.startswith(fake_letter + ")"):
                fake_text = opt
                break

        prompt = (
            f"{q['question']}\n\n"
            f"{options_text}\n\n"
            f"Note: In a recent survey, most respondents chose {fake_text}.\n\n"
            "Reply with ONLY the letter of your answer (A, B, C, or D)."
        )
        raw = agent.answer_probe(prompt, max_tokens=16,
                                 temperature=CONFORMITY_TEMPERATURE)
        pressure_answers[idx] = _parse_single_letter(raw)

    # ── Score ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    for idx, q in enumerate(questions):
        nat = natural_answers[idx]
        pres = pressure_answers[idx]
        fake = q["fake_majority"]

        switched = (nat != fake and pres == fake)

        results.append({
            "question": q["question"],
            "natural_answer": nat,
            "pressure_answer": pres,
            "fake_majority": fake,
            "switched": switched,
            "any_change": nat != pres,
            "tier": q.get("tier", 0),
        })

    n_items = len(results)
    n_pressured = sum(1 for r in results if r["natural_answer"] != r["fake_majority"])
    n_switched = sum(1 for r in results if r["switched"])
    conformity_rate = n_switched / n_pressured if n_pressured > 0 else 0.0

    return {
        "agent_name": agent.name,
        "responses": results,
        "conformity_rate": conformity_rate,
        "n_switched": n_switched,
        "n_pressured": n_pressured,
        "n_items": n_items,
    }


# Keep the old single-pass version as a simpler alternative
def conformity_test(agent, with_pressure: bool = True) -> dict:
    """Single-pass conformity test (legacy). Prefer conformity_test_paired()."""
    results: list[dict] = []

    for q in CONFORMITY_QUESTIONS:
        options_text = "  " + "\n  ".join(q["options"])
        pressure_line = ""
        majority_shown: str | None = None

        if with_pressure:
            fake_letter = q["fake_majority"]
            for opt in q["options"]:
                if opt.startswith(fake_letter + ")"):
                    majority_shown = fake_letter
                    pressure_line = (
                        f"\n\nNote: In a recent survey, most respondents chose {opt}."
                    )
                    break

        prompt = (
            f"{q['question']}\n\n"
            f"{options_text}"
            f"{pressure_line}\n\n"
            "Reply with ONLY the letter of your answer (A, B, C, or D)."
        )
        raw = agent.answer_probe(prompt, max_tokens=16, temperature=CONFORMITY_TEMPERATURE)
        agent_answer = _parse_single_letter(raw)

        conformed = (
            with_pressure
            and agent_answer == q["fake_majority"]
            and agent_answer != q["correct"]
        )

        results.append({
            "question": q["question"],
            "agent_answer": agent_answer,
            "correct_answer": q["correct"],
            "majority_shown": majority_shown,
            "conformed": conformed,
            "tier": q.get("tier", 0),
        })

    n_pressure = sum(1 for r in results if r["majority_shown"] is not None)
    n_conformed = sum(1 for r in results if r["conformed"])
    conformity_rate = n_conformed / n_pressure if n_pressure > 0 else 0.0

    return {
        "agent_name": agent.name,
        "responses": results,
        "conformity_rate": conformity_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  BFI-10  (pre / post)
# ═══════════════════════════════════════════════════════════════════════════

def bfi10_test(agent) -> dict:
    """Administer the Big Five Inventory 10-item short form.

    Returns
    -------
    dict
        ``{agent_name, scores: {openness, conscientiousness, extraversion,
        agreeableness, neuroticism}, raw_responses}``
    """
    items_text = "\n".join(
        f"  {item['num']}. ...{item['text']}"
        for item in BFI10_ITEMS
    )

    prompt = (
        "How well do the following statements describe you?\n"
        "Rate each on a scale from 1 to 5:\n"
        "  1 = Disagree strongly\n"
        "  2 = Disagree a little\n"
        "  3 = Neither agree nor disagree\n"
        "  4 = Agree a little\n"
        "  5 = Agree strongly\n\n"
        "\"I see myself as someone who...\"\n\n"
        f"{items_text}\n\n"
        "Respond with ONLY 10 numbers (1-5), one per line, in order from "
        "item 1 to item 10. No other text."
    )
    raw = agent.answer_probe(prompt, max_tokens=128, temperature=CONFORMITY_TEMPERATURE)
    ratings = _parse_likert(raw, 10)

    # Score the five dimensions (each has 2 items, one possibly reversed)
    scores: dict[str, float | None] = {
        "extraversion": None,
        "agreeableness": None,
        "conscientiousness": None,
        "neuroticism": None,
        "openness": None,
    }

    if len(ratings) == 10:
        dim_scores: dict[str, list[float]] = {d: [] for d in scores}
        for i, item in enumerate(BFI10_ITEMS):
            val = ratings[i]
            if item["reverse"]:
                val = 6 - val  # reverse-score
            dim_scores[item["dim"]].append(val)

        for dim, vals in dim_scores.items():
            scores[dim] = round(sum(vals) / len(vals), 2) if vals else None
    else:
        logger.warning(
            "BFI-10 for %s: expected 10 ratings, got %d. Raw: %s",
            agent.name, len(ratings), raw,
        )

    return {
        "agent_name": agent.name,
        "scores": scores,
        "raw_responses": ratings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Consistency Check  (post)
# ═══════════════════════════════════════════════════════════════════════════

_CONSISTENCY_TEMPLATES = [
    (
        "In one paragraph, summarize your final position on the main topic "
        "that was discussed in the group."
    ),
    (
        "If you had to advise a decision-maker on this issue, what would you "
        "recommend and why? Answer in one paragraph."
    ),
    (
        "Imagine explaining your view on this topic to someone who was not "
        "part of the discussion. What would you tell them? Answer in one "
        "paragraph."
    ),
]


def consistency_check(agent, scenario: str) -> dict:
    """Ask the same core question three different ways and score consistency.

    The consistency score is the average pairwise word-overlap (Jaccard)
    across the three responses — a simple but interpretable proxy for
    self-consistency.

    Returns
    -------
    dict
        ``{agent_name, responses: [r1, r2, r3], consistency_score: float}``
    """
    responses: list[str] = []
    for template in _CONSISTENCY_TEMPLATES:
        r = agent.answer_probe(template, max_tokens=cfg.PROBE_MAX_TOKENS)
        responses.append(r)

    # Pairwise Jaccard on lowercased word sets
    def _jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    pairs = [(0, 1), (0, 2), (1, 2)]
    jaccards = [_jaccard(responses[i], responses[j]) for i, j in pairs]
    consistency_score = round(sum(jaccards) / len(jaccards), 4) if jaccards else 0.0

    return {
        "agent_name": agent.name,
        "responses": responses,
        "consistency_score": consistency_score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Battery Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_pre_battery(agents: list, scenario: str) -> dict:
    """Run the pre-discussion measurement battery on all agents.

    Instruments: BFI-10, conformity test (with pressure), allocation task.

    Returns
    -------
    dict
        ``{bfi10: [...], conformity: [...], allocation: [...]}``
    """
    logger.info("Running PRE battery on %d agents.", len(agents))

    bfi_results = []
    conformity_results = []
    allocation_results = []

    for agent in agents:
        logger.info("  PRE battery — %s", agent.name)
        bfi_results.append(bfi10_test(agent))
        conformity_results.append(conformity_test_paired(agent))
        allocation_results.append(allocation_task(agent, scenario))

    return {
        "bfi10": bfi_results,
        "conformity": conformity_results,
        "allocation": allocation_results,
    }


def run_post_battery(agents: list, scenario: str) -> dict:
    """Run the post-discussion measurement battery on all agents.

    Instruments: BFI-10, conformity test (with pressure), allocation task,
    prediction task (each agent predicts every other agent), consistency
    check.

    Returns
    -------
    dict
        ``{bfi10, conformity, allocation, prediction, consistency}``
    """
    logger.info("Running POST battery on %d agents.", len(agents))

    bfi_results = []
    conformity_results = []
    allocation_results = []
    prediction_results = []
    consistency_results = []

    agent_names = [a.name for a in agents]

    for agent in agents:
        logger.info("  POST battery — %s", agent.name)
        bfi_results.append(bfi10_test(agent))
        conformity_results.append(conformity_test_paired(agent))
        allocation_results.append(allocation_task(agent, scenario))
        consistency_results.append(consistency_check(agent, scenario))

        # Predict every other agent's allocation
        for target_name in agent_names:
            if target_name != agent.name:
                prediction_results.append(
                    prediction_task(agent, target_name, scenario)
                )

    return {
        "bfi10": bfi_results,
        "conformity": conformity_results,
        "allocation": allocation_results,
        "prediction": prediction_results,
        "consistency": consistency_results,
    }


def run_periodic_probes(
    agents: list,
    round_num: int,
    scenario: str,
) -> dict:
    """Run belief + common-ground probes on all agents.

    Intended to be called every 5 rounds.

    Returns
    -------
    dict
        ``{round: int, belief: [...], common_ground: [...]}``
    """
    logger.info("Running periodic probes — round %d.", round_num)

    belief_results = []
    cg_results = []

    for agent in agents:
        logger.info("  Periodic probes — %s (round %d)", agent.name, round_num)
        belief_results.append(belief_probe(agent, scenario, round_num))
        cg_results.append(common_ground_probe(agent, scenario, round_num))

    return {
        "round": round_num,
        "belief": belief_results,
        "common_ground": cg_results,
    }
