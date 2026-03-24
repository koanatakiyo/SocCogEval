"""Conversation orchestrator: manages multi-round agent discussions.

Handles:
- Pair selection via network topology
- Multi-turn conversation within each pair
- Memory extraction after each conversation
- Probe scheduling (every 5 rounds)
- Data collection and saving
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .agent import Agent
from .config import ExperimentConfig, DATA_DIR
from .memory import extract_memory
from .network import Network

logger = logging.getLogger(__name__)


@dataclass
class ConversationRecord:
    """Record of a single conversation between agents."""
    round_num: int
    agent_a_id: int
    agent_b_id: int
    agent_a_name: str
    agent_b_name: str
    turns: list[dict] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class RoundRecord:
    """Record of all conversations in one round."""
    round_num: int
    conversations: list[ConversationRecord] = field(default_factory=list)
    probe_responses: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


class Orchestrator:
    """Manages the full experiment lifecycle for a single run."""

    def __init__(
        self,
        config: ExperimentConfig,
        agents: list[Agent],
        network: Network,
    ):
        self.config = config
        self.agents = agents
        self.network = network
        self.rounds: list[RoundRecord] = []
        self.run_dir: Optional[Path] = None

    def setup_run_dir(self, base_dir: Optional[Path] = None) -> Path:
        """Create output directory for this run."""
        base = base_dir or DATA_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.condition_name}_{self.config.scenario}_{timestamp}"
        if self.config.run_id:
            run_name = f"{self.config.run_id}_{run_name}"
        self.run_dir = base / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(self.run_dir / "config.json")
        return self.run_dir

    def run_conversation(
        self,
        agent_a: Agent,
        agent_b: Agent,
        round_num: int,
        n_turns: int = 3,
    ) -> ConversationRecord:
        """Run a multi-turn conversation between two agents.

        Each agent takes turns responding. The conversation is formatted
        as a group discussion context.
        """
        record = ConversationRecord(
            round_num=round_num,
            agent_a_id=agent_a.agent_id,
            agent_b_id=agent_b.agent_id,
            agent_a_name=agent_a.name,
            agent_b_name=agent_b.name,
            timestamp=datetime.now().isoformat(),
        )

        conversation_so_far = []

        for turn in range(n_turns):
            # Agent A speaks
            prompt_a = self._build_conversation_prompt(
                agent_a, agent_b, conversation_so_far, round_num
            )
            if self.config.use_s2:
                response_a = agent_a.respond_s2(prompt_a, agent_b.name)
            else:
                response_a = agent_a.respond(prompt_a)

            conversation_so_far.append({"speaker": agent_a.name, "text": response_a})
            record.turns.append({
                "turn": turn * 2,
                "speaker": agent_a.name,
                "speaker_id": agent_a.agent_id,
                "text": response_a,
            })

            # Agent B speaks
            prompt_b = self._build_conversation_prompt(
                agent_b, agent_a, conversation_so_far, round_num
            )
            if self.config.use_s2:
                response_b = agent_b.respond_s2(prompt_b, agent_a.name)
            else:
                response_b = agent_b.respond(prompt_b)

            conversation_so_far.append({"speaker": agent_b.name, "text": response_b})
            record.turns.append({
                "turn": turn * 2 + 1,
                "speaker": agent_b.name,
                "speaker_id": agent_b.agent_id,
                "text": response_b,
            })

        return record

    def _build_conversation_prompt(
        self,
        speaker: Agent,
        partner: Agent,
        conversation_so_far: list[dict],
        round_num: int,
    ) -> str:
        """Build the conversation prompt for an agent."""
        parts = [
            f"This is round {round_num} of the committee discussion.",
            f"You are having a conversation with {partner.name}.",
        ]

        if conversation_so_far:
            parts.append("\nConversation so far:")
            for msg in conversation_so_far:
                parts.append(f"\n{msg['speaker']}: {msg['text']}")
            parts.append(f"\nNow it's your turn, {speaker.name}. Respond thoughtfully.")
        else:
            parts.append(
                f"\nStart the conversation by sharing your thoughts on the topic."
            )

        return "\n".join(parts)

    def extract_memories(
        self, conversation: ConversationRecord, agents: list[Agent]
    ):
        """Extract and store memories for each agent in the conversation."""
        # Format conversation as text
        conv_text = "\n".join(
            f"{t['speaker']}: {t['text']}" for t in conversation.turns
        )

        for agent in agents:
            if agent.agent_id in (conversation.agent_a_id, conversation.agent_b_id):
                memory = extract_memory(
                    agent_name=agent.name,
                    conversation_text=conv_text,
                    round_num=conversation.round_num,
                    model=self.config.model,
                )
                agent.memory.add(memory)

    def run_round(self, round_num: int, n_turns: int = 3) -> RoundRecord:
        """Run one full round of conversations."""
        t0 = time.time()
        record = RoundRecord(round_num=round_num)

        # Get conversation pairs from network
        n_pairs = self.config.interaction.conversations_per_round
        pairs = self.network.get_conversation_pairs(n_pairs)

        if not pairs:
            logger.info(f"Round {round_num}: No conversations (no-interaction condition)")
            record.duration_seconds = time.time() - t0
            return record

        logger.info(
            f"Round {round_num}: {len(pairs)} conversations "
            f"({', '.join(f'{a.name}-{b.name}' for a, b in pairs)})"
        )

        for agent_a, agent_b in pairs:
            conv = self.run_conversation(agent_a, agent_b, round_num, n_turns)
            record.conversations.append(conv)

            # Extract memories
            self.extract_memories(conv, self.agents)

        record.duration_seconds = time.time() - t0
        self.rounds.append(record)
        return record

    def run_experiment(
        self,
        n_rounds: Optional[int] = None,
        n_turns_per_conversation: int = 3,
        probe_interval: int = 5,
        probe_fn=None,
    ) -> list[RoundRecord]:
        """Run the full experiment for all rounds.

        Args:
            n_rounds: Number of rounds (default from config)
            n_turns_per_conversation: Turns per pair conversation
            probe_interval: Run probes every N rounds
            probe_fn: Optional function(agents, round_num) -> dict for probes
        """
        n_rounds = n_rounds or self.config.interaction.n_rounds

        if self.config.interaction.n_rounds == 0:
            logger.info("No-interaction condition: skipping all rounds")
            return self.rounds

        if self.run_dir is None:
            self.setup_run_dir()

        logger.info(
            f"Starting experiment: {self.config.condition_name} | "
            f"{self.config.scenario} | {n_rounds} rounds | "
            f"model={self.config.model}"
        )

        for round_num in range(1, n_rounds + 1):
            record = self.run_round(round_num, n_turns_per_conversation)

            # Run probes at intervals
            if probe_fn and round_num % probe_interval == 0:
                logger.info(f"Running probes at round {round_num}")
                record.probe_responses = probe_fn(self.agents, round_num)

            # Save incrementally
            self._save_round(record)

            logger.info(
                f"Round {round_num}/{n_rounds} complete "
                f"({record.duration_seconds:.1f}s, "
                f"{len(record.conversations)} conversations)"
            )

        # Save final state
        self._save_final()
        return self.rounds

    def _save_round(self, record: RoundRecord):
        """Save a round's data to disk."""
        if self.run_dir is None:
            return
        round_dir = self.run_dir / "rounds"
        round_dir.mkdir(exist_ok=True)

        data = {
            "round_num": record.round_num,
            "duration_seconds": record.duration_seconds,
            "conversations": [
                {
                    "agent_a": c.agent_a_name,
                    "agent_b": c.agent_b_name,
                    "turns": c.turns,
                    "timestamp": c.timestamp,
                }
                for c in record.conversations
            ],
            "probe_responses": record.probe_responses,
        }

        with open(round_dir / f"round_{record.round_num:03d}.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_final(self):
        """Save final experiment state."""
        if self.run_dir is None:
            return

        # Save agent memories
        memories_dir = self.run_dir / "memories"
        memories_dir.mkdir(exist_ok=True)
        for agent in self.agents:
            mem_data = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "group_id": agent.group_id,
                "memories": agent.memory.to_list(),
            }
            with open(memories_dir / f"{agent.name}.json", "w") as f:
                json.dump(mem_data, f, indent=2, ensure_ascii=False)

        # Save summary
        summary = {
            "config": self.config.to_dict(),
            "n_rounds_completed": len(self.rounds),
            "total_conversations": sum(len(r.conversations) for r in self.rounds),
            "total_duration_seconds": sum(r.duration_seconds for r in self.rounds),
            "agents": [
                {
                    "id": a.agent_id,
                    "name": a.name,
                    "group_id": a.group_id,
                    "n_memories": len(a.memory.entries),
                }
                for a in self.agents
            ],
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Experiment saved to {self.run_dir}")
