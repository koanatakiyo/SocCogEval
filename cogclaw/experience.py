"""Advantage-weighted experience library for social simulation agents.

Inspired by FutureMap's evolver.py, adapted for multi-agent social
interaction over repeated rounds.  Agents accumulate transferable
lessons about negotiation, persuasion, agreement, etc., and retrieve
the most relevant (and historically successful) experiences when
composing their next action.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from . import llm

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase split + strip punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _make_id(description: str, domain: str) -> str:
    return hashlib.md5(f"{description}|{domain}".encode()).hexdigest()


# ── ExperienceLibrary ──────────────────────────────────────────────────────

class ExperienceLibrary:
    """Stores, retrieves, and maintains advantage-weighted experiences."""

    def __init__(self):
        self._experiences: list[dict] = []

    # ── core mutation ──────────────────────────────────────────────────

    def add(
        self,
        description: str,
        domain: str,
        advantage: float,
        tags: list[str],
        source_round: int,
    ) -> dict:
        """Add an experience, merging with an existing one if similar.

        Similarity is measured by token Jaccard on the description field.
        If a match (>= 0.6) in the same domain exists, the advantage is
        EMA-merged and tags are unioned rather than creating a duplicate.
        """
        advantage = max(-1.0, min(1.0, advantage))
        new_tokens = _tokenize(description)

        # Try to merge with existing experience in the same domain
        for exp in self._experiences:
            if exp["domain"] != domain:
                continue
            if _jaccard(new_tokens, _tokenize(exp["description"])) >= 0.6:
                # Merge: EMA advantage, union tags, keep newer description
                exp["advantage"] = max(-1.0, min(1.0,
                    0.7 * exp["advantage"] + 0.3 * advantage
                ))
                exp["tags"] = sorted(set(exp["tags"]) | set(tags))
                exp["source_round"] = max(exp["source_round"], source_round)
                logger.debug("Merged experience %s (adv=%.2f)", exp["id"], exp["advantage"])
                return exp

        # No merge candidate — create new entry
        exp = {
            "id": _make_id(description, domain),
            "description": description,
            "domain": domain,
            "advantage": advantage,
            "tags": sorted(set(tags)),
            "source_round": source_round,
        }
        self._experiences.append(exp)
        logger.debug("Added experience %s domain=%s adv=%.2f", exp["id"], domain, advantage)
        return exp

    # ── retrieval ──────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Keyword-based retrieval with advantage weighting.

        score = keyword_overlap * (1 + (advantage + 1) / 2)
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for exp in self._experiences:
            text_tokens = _tokenize(
                f"{exp['description']} {exp['domain']} {' '.join(exp['tags'])}"
            )
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            if overlap <= 0:
                continue
            adv = exp["advantage"]
            score = overlap * (1 + (adv + 1) / 2)
            scored.append((score, exp))

        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    # ── advantage updates ──────────────────────────────────────────────

    def update_advantage(
        self,
        experience_id: str,
        signal: float,
        eta: float = 0.3,
    ) -> Optional[dict]:
        """EMA update: advantage = (1 - eta) * old + eta * signal."""
        signal = max(-1.0, min(1.0, signal))
        for exp in self._experiences:
            if exp["id"] == experience_id:
                old = exp["advantage"]
                exp["advantage"] = max(-1.0, min(1.0,
                    (1 - eta) * old + eta * signal
                ))
                logger.debug(
                    "Updated advantage %s: %.3f -> %.3f",
                    experience_id, old, exp["advantage"],
                )
                return exp
        return None

    # ── pruning ────────────────────────────────────────────────────────

    def prune(self, threshold: float = -0.5) -> list[str]:
        """Remove experiences whose advantage falls below *threshold*."""
        removed: list[str] = []
        kept: list[dict] = []
        for exp in self._experiences:
            if exp["advantage"] < threshold:
                removed.append(exp["id"])
            else:
                kept.append(exp)
        self._experiences = kept
        if removed:
            logger.info("Pruned %d low-advantage experiences", len(removed))
        return removed

    # ── prompt formatting ──────────────────────────────────────────────

    def format_for_prompt(self, query: str, top_k: int = 3) -> str:
        """Return a human-readable hint block for injection into an agent prompt."""
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["[Relevant past experiences]"]
        for i, exp in enumerate(hits, 1):
            adv_label = f"{exp['advantage']:+.2f}"
            lines.append(
                f"  {i}. [{adv_label}] ({exp['domain']}) {exp['description']}"
            )
        return "\n".join(lines)

    # ── persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._experiences, indent=2, ensure_ascii=False))

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            self._experiences = json.loads(p.read_text())
        else:
            logger.warning("Experience file not found: %s", path)

    # ── accessors ──────────────────────────────────────────────────────

    def to_list(self) -> list[dict]:
        """Return a copy of all experiences."""
        return list(self._experiences)

    def __len__(self) -> int:
        return len(self._experiences)

    def __repr__(self) -> str:
        return f"<ExperienceLibrary n={len(self._experiences)}>"


# ── Standalone extraction helper ───────────────────────────────────────────

def extract_experience(
    conversation_text: str,
    outcome: str,
    agent_name: str,
    round_num: int,
    model: str = "",
) -> dict:
    """Use an LLM to distil a transferable lesson from a conversation outcome.

    Returns a dict ready for ``ExperienceLibrary.add(**result)``.
    """
    outcome_lower = outcome.strip().lower()
    positive = outcome_lower in {
        "success", "agree", "agreed", "cooperation",
        "cooperate", "positive", "win", "accepted",
    }

    prompt = f"""You are analyzing a social interaction to extract a reusable lesson.

Agent: {agent_name}
Round: {round_num}
Outcome: {outcome}

Conversation:
{conversation_text[:2000]}

Extract ONE concise, transferable lesson that {agent_name} should remember for
future social interactions.  Focus on *strategy* (what worked or failed), not on
the specific topic discussed.

Return JSON only:
{{
  "description": "one-sentence actionable lesson",
  "domain": "negotiation | persuasion | agreement | conflict | coordination | trust",
  "tags": ["tag1", "tag2"]
}}"""

    try:
        result = llm.chat_json(prompt, model=model, max_tokens=256)
    except Exception as e:
        logger.warning("extract_experience LLM call failed: %s", e)
        # Fallback: return a generic entry derived from the outcome
        result = {
            "description": f"Round {round_num}: outcome was {outcome}.",
            "domain": "coordination",
            "tags": [outcome_lower],
        }

    # Normalise domain to a known set (fall back to "coordination")
    known_domains = {
        "negotiation", "persuasion", "agreement",
        "conflict", "coordination", "trust",
    }
    domain = result.get("domain", "coordination").lower().strip()
    if domain not in known_domains:
        domain = "coordination"

    return {
        "description": result.get("description", f"Lesson from round {round_num}"),
        "domain": domain,
        "tags": result.get("tags", []),
        "advantage": 0.2 if positive else -0.1,
        "source_round": round_num,
    }
