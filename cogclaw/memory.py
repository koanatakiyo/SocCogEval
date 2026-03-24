"""Memory extraction system: LLM call → structured JSON.

After each conversation, LLM extraction produces a social-relational
memory entry with three facets: episodic, social, self-relevant.

From working_plan_v3.md §4.4:
- Episodic: event, context, my_response
- Social: participants, their_stances, social_signals, group_dynamics
- Self-relevant: how_they_see_me, my_evolution
"""

import logging
from typing import Optional

from .llm import chat_json

logger = logging.getLogger(__name__)

# ── Extraction Prompt Template ──────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation between agents, extract a structured memory from the perspective of {agent_name}.

CONVERSATION (Round {round_num}):
{conversation_text}

Extract {agent_name}'s memory of this interaction as a JSON object with exactly this structure:

{{
  "episodic": {{
    "event": "<brief description of what was discussed>",
    "context": "<Round {round_num} discussion>",
    "my_response": "<what {agent_name} argued or expressed>"
  }},
  "social": {{
    "participants": [<list of other agent names in the conversation>],
    "their_stances": {{<agent_name>: "<their main position>" for each other participant}},
    "social_signals": "<notable social dynamics: agreement, disagreement, persuasion attempts>",
    "group_dynamics": "<overall group dynamic: consensus forming, polarization, debate>"
  }},
  "self_relevant": {{
    "how_they_see_me": "<how others perceived {agent_name} based on their reactions>",
    "my_evolution": "<how {agent_name}'s thinking shifted (or didn't) during this conversation>"
  }}
}}"""


# ── Memory Extraction Function ──────────────────────────────────────────────

def extract_memory(
    agent_name: str,
    conversation_text: str,
    round_num: int,
    model: str = "",
    max_tokens: int = 512,
) -> dict:
    """Extract a social-relational memory entry from a conversation.

    Returns a dict with keys: round, episodic, social, self_relevant.
    """
    prompt = EXTRACTION_PROMPT.format(
        agent_name=agent_name,
        round_num=round_num,
        conversation_text=conversation_text,
    )

    try:
        memory = chat_json(
            prompt=prompt,
            model=model,
            temperature=0.0,  # deterministic extraction
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Memory extraction failed for {agent_name} round {round_num}: {e}")
        memory = _empty_memory()

    # Ensure all required fields exist
    memory = _validate_memory(memory)
    memory["round"] = round_num
    return memory


def _empty_memory() -> dict:
    """Return an empty memory structure."""
    return {
        "episodic": {"event": "", "context": "", "my_response": ""},
        "social": {
            "participants": [],
            "their_stances": {},
            "social_signals": "",
            "group_dynamics": "",
        },
        "self_relevant": {"how_they_see_me": "", "my_evolution": ""},
    }


def _validate_memory(mem: dict) -> dict:
    """Ensure memory has all required fields, filling missing ones."""
    template = _empty_memory()
    for facet in ("episodic", "social", "self_relevant"):
        if facet not in mem:
            mem[facet] = template[facet]
        else:
            for key, default in template[facet].items():
                if key not in mem[facet]:
                    mem[facet][key] = default
    return mem


# ── Consistency Validation ──────────────────────────────────────────────────

def validate_extraction_consistency(
    conversation_text: str,
    agent_name: str,
    round_num: int,
    n_runs: int = 5,
    model: str = "",
) -> dict:
    """Run extraction N times on the same conversation, measure field-level agreement.

    Used during pilot to validate >90% consistency (working_plan_v3.md §4.4).
    Returns: {field_agreements: dict, overall_rate: float, extractions: list}
    """
    extractions = []
    for _ in range(n_runs):
        mem = extract_memory(agent_name, conversation_text, round_num, model=model)
        extractions.append(mem)

    # Compute field-level agreement
    fields = [
        ("episodic", "event"),
        ("episodic", "context"),
        ("episodic", "my_response"),
        ("social", "participants"),
        ("social", "social_signals"),
        ("social", "group_dynamics"),
        ("self_relevant", "how_they_see_me"),
        ("self_relevant", "my_evolution"),
    ]

    agreements = {}
    for facet, key in fields:
        values = [e[facet][key] for e in extractions]
        # For lists, compare sorted string representations
        if isinstance(values[0], list):
            values = [str(sorted(v)) for v in values]
        # Agreement = fraction of pairs that match the majority value
        from collections import Counter
        counter = Counter(values)
        most_common_count = counter.most_common(1)[0][1]
        agreements[f"{facet}.{key}"] = most_common_count / len(values)

    overall = sum(agreements.values()) / len(agreements)

    return {
        "field_agreements": agreements,
        "overall_rate": overall,
        "n_runs": n_runs,
        "extractions": extractions,
    }
