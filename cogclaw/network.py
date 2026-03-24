"""Network topology and pair selection for conversation orchestration.

Supports:
- All-to-all (open): any agent can talk to any other
- Segregated: two groups, agents only talk within their group
- None: no interaction (for seed-only and pure baseline conditions)
"""

import random
import logging
from typing import Optional

from .agent import Agent

logger = logging.getLogger(__name__)


class Network:
    """Social network managing agent connectivity and pair selection."""

    def __init__(self, topology: str, agents: list[Agent], n_groups: int = 1):
        """
        Args:
            topology: "all_to_all", "segregated", or "none"
            agents: list of Agent instances
            n_groups: number of groups (2 for segregated)
        """
        self.topology = topology
        self.agents = agents
        self.n_groups = n_groups
        self.groups: dict[int, list[Agent]] = {}

        if topology == "segregated" and n_groups >= 2:
            self._assign_groups(n_groups)
        elif topology == "all_to_all":
            # All agents in one group
            self.groups[0] = list(agents)
            for a in agents:
                a.group_id = 0
        elif topology == "none":
            # No groups, no interaction
            pass

    def _assign_groups(self, n_groups: int):
        """Evenly distribute agents into groups."""
        group_size = len(self.agents) // n_groups
        for i, agent in enumerate(self.agents):
            gid = min(i // group_size, n_groups - 1)
            agent.group_id = gid
            if gid not in self.groups:
                self.groups[gid] = []
            self.groups[gid].append(agent)

    def get_conversation_pairs(self, n_pairs: int) -> list[tuple[Agent, Agent]]:
        """Select pairs of agents for conversation this round.

        Args:
            n_pairs: number of pairs to select

        Returns:
            List of (agent_a, agent_b) tuples
        """
        if self.topology == "none":
            return []

        if self.topology == "all_to_all":
            return self._select_pairs_from_pool(self.agents, n_pairs)

        if self.topology == "segregated":
            # Select pairs within each group
            pairs = []
            pairs_per_group = max(1, n_pairs // self.n_groups)
            for gid, group_agents in self.groups.items():
                group_pairs = self._select_pairs_from_pool(group_agents, pairs_per_group)
                pairs.extend(group_pairs)
            return pairs[:n_pairs]

        return []

    def _select_pairs_from_pool(
        self, pool: list[Agent], n_pairs: int
    ) -> list[tuple[Agent, Agent]]:
        """Select random non-repeating pairs from a pool of agents."""
        if len(pool) < 2:
            return []

        # Generate all possible pairs
        all_pairs = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                all_pairs.append((pool[i], pool[j]))

        random.shuffle(all_pairs)
        return all_pairs[:n_pairs]

    def get_group_members(self, agent: Agent) -> list[Agent]:
        """Get all agents in the same group as the given agent."""
        if agent.group_id is None:
            return list(self.agents)
        return [a for a in self.groups.get(agent.group_id, []) if a != agent]

    def get_all_agents(self) -> list[Agent]:
        return list(self.agents)
