"""Fork-based reasoning graph for S2 deliberate reasoning.

Inspired by FutureMap's reasoning_graph.py, but adapted for social cognition:
instead of forecasting geopolitical events, we model an agent's structured
deliberation about another agent's beliefs, trajectory, and relationship
dynamics before responding in conversation.

The graph is lightweight — 8 agents, 30 rounds, no need for heavy infra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SpeculativeNode:
    """A speculative prediction about what another agent might think or do."""
    subject: str          # agent name being speculated about
    prediction: str       # what we think they'll do/think
    confidence: float     # 0.0 to 1.0
    round_added: int = 0  # which round this was added

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "round_added": self.round_added,
        }


@dataclass
class MemoryNode:
    """A fact node copied from an agent's SocialRelationalMemory."""
    round_num: int
    event: str
    my_response: str
    participants: List[str]
    their_stances: Dict[str, str]
    social_signals: str
    group_dynamics: str

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "event": self.event,
            "my_response": self.my_response,
            "participants": self.participants,
            "their_stances": self.their_stances,
            "social_signals": self.social_signals,
            "group_dynamics": self.group_dynamics,
        }


# ── Reasoning Graph ──────────────────────────────────────────────────────────

class ReasoningGraph:
    """A mutable working copy of an agent's memory, forked for deliberation.

    Unlike FutureMap's version (which wraps a networkx DiGraph for causal
    forecasting), this is a flat structure: memory nodes + speculative nodes.
    For 8 agents over 30 rounds, a list-based approach is simpler and
    sufficient.
    """

    def __init__(self):
        self.memory_nodes: List[MemoryNode] = []
        self.speculative_nodes: List[SpeculativeNode] = []
        self.owner_name: str = ""  # the agent who owns this reasoning graph

    @classmethod
    def fork_from(cls, memory, owner_name: str = "") -> "ReasoningGraph":
        """Create a reasoning graph from an agent's SocialRelationalMemory.

        Copies relevant fields from each memory entry into lightweight
        MemoryNode structures for structured reasoning.

        Parameters
        ----------
        memory : SocialRelationalMemory
            The agent's memory object (has .entries list).
        owner_name : str
            Name of the agent doing the reasoning.
        """
        graph = cls()
        graph.owner_name = owner_name

        for entry in memory.entries:
            ep = entry.get("episodic", {})
            soc = entry.get("social", {})
            node = MemoryNode(
                round_num=entry.get("round", 0),
                event=ep.get("event", ""),
                my_response=ep.get("my_response", ""),
                participants=soc.get("participants", []),
                their_stances=soc.get("their_stances", {}),
                social_signals=soc.get("social_signals", ""),
                group_dynamics=soc.get("group_dynamics", ""),
            )
            graph.memory_nodes.append(node)

        return graph

    def add_speculation(
        self,
        subject: str,
        prediction: str,
        confidence: float,
        round_num: int = 0,
    ) -> None:
        """Add a speculative node about what another agent might think/do."""
        confidence = max(0.0, min(1.0, confidence))
        self.speculative_nodes.append(
            SpeculativeNode(
                subject=subject,
                prediction=prediction,
                confidence=confidence,
                round_added=round_num,
            )
        )

    # ── Context Building ──────────────────────────────────────────────────

    def _extract_agent_narrative(self, target: str) -> List[str]:
        """Extract the trajectory of a target agent's stated positions.

        Walks through memory nodes chronologically, pulling out what the
        target agent said in each round they appeared in.
        """
        narrative = []
        for node in sorted(self.memory_nodes, key=lambda n: n.round_num):
            if target in node.participants or target in node.their_stances:
                stance = node.their_stances.get(target, "")
                if stance:
                    narrative.append(f"Round {node.round_num}: {stance}")
        return narrative

    def _extract_relationship_summary(self, target: str) -> List[str]:
        """Summarize how the owner and target have interacted."""
        interactions = []
        for node in sorted(self.memory_nodes, key=lambda n: n.round_num):
            if target not in node.participants and target not in node.their_stances:
                continue
            parts = []
            if node.social_signals:
                parts.append(node.social_signals)
            if node.group_dynamics:
                parts.append(f"dynamics: {node.group_dynamics}")
            if parts:
                interactions.append(f"Round {node.round_num}: {'; '.join(parts)}")
        return interactions

    def _detect_contradictions(self, target: str) -> List[str]:
        """Find conflicting stances between the owner and the target.

        A contradiction is when the owner's response and the target's stance
        in the same round express opposing positions. We use simple keyword
        heuristics rather than an LLM call to keep this fast.
        """
        contradictions = []
        opposition_signals = [
            "disagree", "opposed", "conflict", "against", "reject",
            "but", "however", "unlike", "contrary", "different",
        ]
        for node in sorted(self.memory_nodes, key=lambda n: n.round_num):
            target_stance = node.their_stances.get(target, "")
            if not target_stance or not node.my_response:
                continue
            # Check social signals for explicit disagreement
            signals_lower = node.social_signals.lower()
            combined = f"{node.my_response} {target_stance}".lower()
            if any(s in signals_lower or s in combined for s in opposition_signals):
                contradictions.append(
                    f"Round {node.round_num}: "
                    f"You said \"{node.my_response}\" "
                    f"while {target} held \"{target_stance}\""
                )
        return contradictions

    def _detect_trend(self, target: str) -> str:
        """Determine if the target is moving toward or away from the owner.

        Returns one of: CONVERGING, DIVERGING, STABLE.

        Logic: look at the last few rounds of social signals. If recent
        interactions show increasing agreement, return CONVERGING. If
        increasing disagreement, return DIVERGING. Otherwise STABLE.
        """
        relevant_signals = []
        for node in sorted(self.memory_nodes, key=lambda n: n.round_num):
            if target not in node.participants and target not in node.their_stances:
                continue
            relevant_signals.append(node.social_signals.lower())

        if not relevant_signals:
            return "STABLE"

        # Weight recent signals more heavily — look at last 3
        recent = relevant_signals[-3:]
        converge_words = ["agree", "consensus", "align", "common ground",
                          "persuad", "convinc", "come around", "shifted toward"]
        diverge_words = ["disagree", "polariz", "conflict", "opposition",
                         "reject", "resist", "pushed back", "further apart"]

        converge_score = sum(
            1 for sig in recent
            for w in converge_words
            if w in sig
        )
        diverge_score = sum(
            1 for sig in recent
            for w in diverge_words
            if w in sig
        )

        if converge_score > diverge_score:
            return "CONVERGING"
        elif diverge_score > converge_score:
            return "DIVERGING"
        return "STABLE"

    def build_structured_context(self, target_agent: str) -> str:
        """Build a structured prompt context for reasoning about a target agent.

        Returns a formatted string containing:
        - Agent narrative (target's position trajectory)
        - Relationship summary (interaction history)
        - Contradictions (conflicting stances)
        - Trend (CONVERGING / DIVERGING / STABLE)
        - Speculations (if any were added for this target)
        """
        sections = []

        # Header
        sections.append(
            f"=== Structured Reasoning Context: {self.owner_name} about {target_agent} ==="
        )

        # Agent narrative
        narrative = self._extract_agent_narrative(target_agent)
        sections.append(f"\n[{target_agent}'s Position Trajectory]")
        if narrative:
            sections.extend(narrative)
        else:
            sections.append(f"No recorded positions from {target_agent}.")

        # Relationship summary
        relationship = self._extract_relationship_summary(target_agent)
        sections.append(f"\n[Relationship History]")
        if relationship:
            sections.extend(relationship)
        else:
            sections.append(f"No prior interactions with {target_agent}.")

        # Contradictions
        contradictions = self._detect_contradictions(target_agent)
        sections.append(f"\n[Contradictions Detected]")
        if contradictions:
            sections.extend(contradictions)
        else:
            sections.append("No direct contradictions detected.")

        # Trend
        trend = self._detect_trend(target_agent)
        sections.append(f"\n[Trend]")
        sections.append(f"Trajectory with {target_agent}: {trend}")

        # Speculations about this target
        target_specs = [
            s for s in self.speculative_nodes if s.subject == target_agent
        ]
        if target_specs:
            sections.append(f"\n[Speculations]")
            for spec in target_specs:
                sections.append(
                    f"- {spec.prediction} (confidence: {spec.confidence:.0%})"
                )

        return "\n".join(sections)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializable output of the reasoning graph."""
        return {
            "owner": self.owner_name,
            "memory_nodes": [n.to_dict() for n in self.memory_nodes],
            "speculative_nodes": [n.to_dict() for n in self.speculative_nodes],
        }


# ── Convenience Function ─────────────────────────────────────────────────────

def build_s2_context(agent_memory, target_agent_name: str, owner_name: str = "") -> str:
    """Convenience: fork a reasoning graph and build structured context.

    Parameters
    ----------
    agent_memory : SocialRelationalMemory
        The reasoning agent's memory.
    target_agent_name : str
        Name of the agent being reasoned about.
    owner_name : str
        Name of the agent doing the reasoning.

    Returns
    -------
    str
        Formatted structured context, ready to inject into a system prompt.
    """
    graph = ReasoningGraph.fork_from(agent_memory, owner_name=owner_name)
    return graph.build_structured_context(target_agent_name)
