"""Agent class with social-relational memory for CogClaw experiments.

Each agent is identically initialized with:
- Same base model
- Same prompting template
- Same role structure ("You are a resident of Greenfield..." / "You are an employee of Riverton...")
- Brief name and role label
- Behavioral instruction: "Express your genuine thoughts honestly"
- NO personality, NO stance, NO ideology, NO knowledge base
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from . import config as cfg
from .llm import chat

logger = logging.getLogger(__name__)


@dataclass
class SocialRelationalMemory:
    """Three-facet social-relational memory (episodic/social/self-relevant).

    This is ONE memory system organized in three facets, mapping to the
    computational social cognition pipeline framework.
    """
    entries: list = field(default_factory=list)

    def add(self, entry: dict):
        """Add a memory entry (extracted from conversation)."""
        self.entries.append(entry)

    def format_for_prompt(self, max_entries: int = 10) -> str:
        """Format recent memories for inclusion in agent's prompt."""
        if not self.entries:
            return "You have no prior interaction memories."

        recent = self.entries[-max_entries:]
        parts = ["Your memories from past interactions:"]
        for i, entry in enumerate(recent, 1):
            ep = entry.get("episodic", {})
            soc = entry.get("social", {})
            self_ref = entry.get("self_relevant", {})

            parts.append(f"\n--- Memory {i} (Round {entry.get('round', '?')}) ---")
            if ep.get("event"):
                parts.append(f"Event: {ep['event']}")
            if ep.get("context"):
                parts.append(f"Context: {ep['context']}")
            if ep.get("my_response"):
                parts.append(f"My position: {ep['my_response']}")
            if soc.get("participants"):
                parts.append(f"Discussed with: {', '.join(soc['participants'])}")
            if soc.get("their_stances"):
                stances = "; ".join(f"{k}: {v}" for k, v in soc["their_stances"].items())
                parts.append(f"Their views: {stances}")
            if soc.get("group_dynamics"):
                parts.append(f"Group dynamics: {soc['group_dynamics']}")
            if self_ref.get("how_they_see_me"):
                parts.append(f"How others see me: {self_ref['how_they_see_me']}")
            if self_ref.get("my_evolution"):
                parts.append(f"My evolving view: {self_ref['my_evolution']}")

        return "\n".join(parts)

    def to_list(self) -> list:
        return list(self.entries)


# ── Agent Names and Roles ───────────────────────────────────────────────────

AGENT_NAMES = [
    "Alex", "Blake", "Casey", "Dana",
    "Ellis", "Finley", "Gray", "Harper",
]

AGENT_ROLES = [
    "community member",
    "community member",
    "community member",
    "community member",
    "community member",
    "community member",
    "community member",
    "community member",
]


# ── Agent Class ─────────────────────────────────────────────────────────────

class Agent:
    """An identically initialized LLM agent for the social cognition experiment."""

    def __init__(
        self,
        agent_id: int,
        name: str,
        scenario_context: str,
        model: str = "",
        seed_info: str = "",
    ):
        self.agent_id = agent_id
        self.name = name
        self.model = model or cfg.PILOT_MODEL
        self.scenario_context = scenario_context
        self.seed_info = seed_info
        self.memory = SocialRelationalMemory()
        self.conversation_history: list[dict] = []  # all conversations
        self.group_id: Optional[int] = None  # assigned group in segregated conditions

    @property
    def system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        parts = [
            f"You are {self.name}, a {AGENT_ROLES[self.agent_id]}.",
            "",
            self.scenario_context,
            "",
            "Express your genuine thoughts honestly. Engage thoughtfully with "
            "others' perspectives. You may change your mind if you find their "
            "arguments compelling, or maintain your position if you disagree.",
        ]

        if self.seed_info:
            parts.extend([
                "",
                "Background information you have received:",
                self.seed_info,
            ])

        memory_text = self.memory.format_for_prompt()
        if self.memory.entries:
            parts.extend(["", memory_text])

        return "\n".join(parts)

    def respond(
        self,
        conversation_prompt: str,
        temperature: float = cfg.CONVERSATION_TEMPERATURE,
        max_tokens: int = cfg.CONVERSATION_MAX_TOKENS,
    ) -> str:
        """Generate a response to a conversation prompt."""
        response = chat(
            prompt=conversation_prompt,
            system=self.system_prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response

    def respond_s2(
        self,
        conversation_prompt: str,
        other_agent_name: str,
        temperature: float = cfg.CONVERSATION_TEMPERATURE,
        max_tokens: int = cfg.CONVERSATION_MAX_TOKENS,
    ) -> str:
        """S2 deliberate reasoning: internal reflection before responding.

        Agent gets an internal prompt to reason about the other agent's
        perspective before formulating a response. The S2 output is NOT
        shared with other agents.
        """
        # Step 1: Internal deliberation (one extra LLM call)
        s2_prompt = (
            f"Based on your past interactions with {other_agent_name}, "
            f"reflect on the following before you respond:\n"
            f"1. What is {other_agent_name}'s likely perspective on this topic?\n"
            f"2. How have they reacted to your views in the past?\n"
            f"3. If you express your current thinking, how might they respond?\n\n"
            f"Think through this briefly."
        )
        _deliberation = chat(
            prompt=s2_prompt,
            system=self.system_prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=512,
        )

        # Step 2: Respond with deliberation context
        augmented_system = (
            self.system_prompt
            + f"\n\n[Internal reflection — not shared with others]\n{_deliberation}"
        )
        response = chat(
            prompt=conversation_prompt,
            system=augmented_system,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response

    def answer_probe(
        self,
        probe_prompt: str,
        max_tokens: int = cfg.PROBE_MAX_TOKENS,
    ) -> str:
        """Answer a measurement probe (belief, personality, conformity, etc.)."""
        return chat(
            prompt=probe_prompt,
            system=self.system_prompt,
            model=self.model,
            temperature=cfg.PROBE_TEMPERATURE,
            max_tokens=max_tokens,
        )

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.name}, group={self.group_id})"
