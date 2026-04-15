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
from typing import Optional

from . import config as cfg
from .llm import chat
from .memory import HypergraphMemory
from .reasoning import build_s2_context

logger = logging.getLogger(__name__)


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
        self.memory = HypergraphMemory()
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
        """S2 deliberate reasoning: structured reflection before responding.

        Uses the ReasoningGraph to build structured context about the other
        agent (position trajectory, relationship history, contradictions,
        trend), then has the agent deliberate before responding.
        The S2 output is NOT shared with other agents.
        """
        # Step 1: Build structured context from memory graph
        structured_ctx = build_s2_context(
            self.memory, other_agent_name, self.name
        )

        # Step 2: Internal deliberation with structured context (one extra LLM call)
        s2_prompt = (
            f"You are about to respond to {other_agent_name}. "
            f"Here is structured analysis of your relationship:\n\n"
            f"{structured_ctx}\n\n"
            f"Based on this analysis, briefly reflect:\n"
            f"1. What is {other_agent_name}'s likely perspective?\n"
            f"2. How might they respond to your current thinking?\n"
            f"3. What approach would be most effective?"
        )
        _deliberation = chat(
            prompt=s2_prompt,
            system=self.system_prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=512,
        )

        # Step 3: Respond with deliberation context
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
        temperature: float = cfg.PROBE_TEMPERATURE,
    ) -> str:
        """Answer a measurement probe (belief, personality, conformity, etc.)."""
        return chat(
            prompt=probe_prompt,
            system=self.system_prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.name}, group={self.group_id})"
