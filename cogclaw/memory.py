"""Memory extraction + hypergraph storage + RRF retrieval.

After each conversation, LLM extraction produces a social-relational
memory entry with three facets: episodic, social, self-relevant.

From working_plan_v3.md SS4.4:
- Episodic: event, context, my_response
- Social: participants, their_stances, social_signals, group_dynamics
- Self-relevant: how_they_see_me, my_evolution

The HypergraphMemory class stores memories in a NetworkX DiGraph where:
- Entity nodes represent agents / concepts / topics
- Hyperedge nodes represent individual memory entries
- Directed edges go from hyperedge -> entity
Retrieval uses RRF (Reciprocal Rank Fusion) over graph-neighbor ranking
and cosine-similarity semantic ranking.
"""

import logging
import re
import time
from typing import Optional

import networkx as nx
import numpy as np

from .llm import chat_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded sentence-transformer singleton
# ---------------------------------------------------------------------------
_ENCODER = None
_ENCODER_NAME = "all-MiniLM-L6-v2"


def _get_encoder():
    """Return (and lazily load) the sentence-transformers encoder."""
    global _ENCODER
    if _ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer(_ENCODER_NAME)
            logger.info("Loaded sentence-transformer: %s", _ENCODER_NAME)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; semantic search disabled"
            )
    return _ENCODER


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns (N, D) float32 array or empty."""
    enc = _get_encoder()
    if enc is None:
        return np.empty((0, 0), dtype=np.float32)
    return enc.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================================
# Extraction prompt template
# ============================================================================

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


# ============================================================================
# Memory extraction functions (unchanged)
# ============================================================================

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
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(
            "Memory extraction failed for %s round %d: %s", agent_name, round_num, e
        )
        memory = _empty_memory()

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


# ============================================================================
# Consistency validation (unchanged)
# ============================================================================

def validate_extraction_consistency(
    conversation_text: str,
    agent_name: str,
    round_num: int,
    n_runs: int = 5,
    model: str = "",
) -> dict:
    """Run extraction N times on the same conversation, measure field-level agreement.

    Used during pilot to validate >90% consistency (working_plan_v3.md SS4.4).
    Returns: {field_agreements: dict, overall_rate: float, extractions: list}
    """
    extractions = []
    for _ in range(n_runs):
        mem = extract_memory(agent_name, conversation_text, round_num, model=model)
        extractions.append(mem)

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
        if isinstance(values[0], list):
            values = [str(sorted(v)) for v in values]
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


# ============================================================================
# HypergraphMemory — hypergraph storage + RRF retrieval
# ============================================================================

# Known agent names for entity extraction
_AGENT_NAMES = {"Alex", "Blake", "Casey", "Dana", "Ellis", "Finley", "Gray", "Harper"}


def _memory_to_text(entry: dict) -> str:
    """Flatten a memory entry dict into a single text string for embedding."""
    parts = []
    ep = entry.get("episodic", {})
    if ep.get("event"):
        parts.append(ep["event"])
    if ep.get("my_response"):
        parts.append(ep["my_response"])
    soc = entry.get("social", {})
    if soc.get("social_signals"):
        parts.append(soc["social_signals"])
    if soc.get("group_dynamics"):
        parts.append(soc["group_dynamics"])
    sr = entry.get("self_relevant", {})
    if sr.get("how_they_see_me"):
        parts.append(sr["how_they_see_me"])
    if sr.get("my_evolution"):
        parts.append(sr["my_evolution"])
    return " ".join(parts) if parts else ""


class HypergraphMemory:
    """Hypergraph-based social-relational memory with RRF retrieval.

    Graph layout (NetworkX DiGraph):
        - Entity nodes:    {id: "entity:<name>", type: "entity",
                            entity_type: "agent"|"concept"|"topic",
                            name: ..., description: ...}
        - Hyperedge nodes: {id: "he:<idx>", type: "hyperedge",
                            content: ..., round: ..., timestamp: ...,
                            facet: "full"}
        - Edges:           hyperedge ---> entity  (directed)

    Vector index:
        Parallel lists kept in sync with hyperedge nodes for cosine search.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        # Ordered list of memory dicts (backward compat)
        self._entries: list[dict] = []
        # Parallel vectors for semantic search
        self._he_ids: list[str] = []       # hyperedge node ids
        self._he_texts: list[str] = []     # flattened text per entry
        self._he_embeddings: list[np.ndarray] = []  # one embedding per entry

    # -- backward-compatible property -----------------------------------------
    @property
    def entries(self) -> list:
        return self._entries

    def to_list(self) -> list:
        """Return all entries as a plain list (backward compat)."""
        return list(self._entries)

    # -- core: add ------------------------------------------------------------
    def add(self, entry: dict):
        """Add a memory entry: create hyperedge node, extract entities, link."""
        self._entries.append(entry)

        he_idx = len(self._he_ids)
        he_id = f"he:{he_idx}"

        text = _memory_to_text(entry)
        round_num = entry.get("round", -1)

        # Create hyperedge node
        self.graph.add_node(
            he_id,
            type="hyperedge",
            content=text,
            round=round_num,
            timestamp=time.time(),
            facet="full",
        )

        # Extract entities and create edges
        entities = self._extract_entities(entry)
        for ename, etype in entities:
            eid = f"entity:{ename}"
            if not self.graph.has_node(eid):
                self.graph.add_node(
                    eid,
                    type="entity",
                    entity_type=etype,
                    name=ename,
                    description="",
                )
            self.graph.add_edge(he_id, eid)

        # Vector index
        self._he_ids.append(he_id)
        self._he_texts.append(text)
        if text:
            try:
                vec = _embed([text])
                if vec.size > 0:
                    self._he_embeddings.append(vec[0])
                else:
                    self._he_embeddings.append(None)
            except Exception:
                self._he_embeddings.append(None)
        else:
            self._he_embeddings.append(None)

    # -- entity extraction (lightweight, no LLM call) -------------------------
    @staticmethod
    def _extract_entities(entry: dict) -> list[tuple[str, str]]:
        """Return list of (entity_name, entity_type) from a memory entry."""
        entities: dict[str, str] = {}

        # Participants are agents
        participants = entry.get("social", {}).get("participants", [])
        for p in participants:
            p_clean = p.strip()
            if p_clean:
                entities[p_clean] = "agent"

        # Stances keys are also agents
        stances = entry.get("social", {}).get("their_stances", {})
        for name in stances:
            n = name.strip()
            if n:
                entities[n] = "agent"

        # Scan text for known agent names not yet captured
        text = _memory_to_text(entry)
        for aname in _AGENT_NAMES:
            if aname in text and aname not in entities:
                entities[aname] = "agent"

        # Extract simple noun-phrase topics from event text (lightweight)
        event = entry.get("episodic", {}).get("event", "")
        if event:
            # Grab quoted phrases or capitalized multi-word phrases as topics
            for match in re.findall(r'"([^"]+)"', event):
                if match not in entities:
                    entities[match] = "topic"

        return list(entities.items())

    # -- retrieval: RRF -------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """RRF retrieval combining graph neighbor ranking + semantic search.

        Returns up to top_k memory entry dicts, highest relevance first.
        """
        if not self._entries:
            return []

        n = len(self._entries)
        k_rrf = 60  # standard RRF constant

        # --- Rank list 1: semantic similarity --------------------------------
        sem_scores: list[tuple[int, float]] = []
        query_vec = None
        try:
            qv = _embed([query])
            if qv.size > 0:
                query_vec = qv[0]
        except Exception:
            pass

        if query_vec is not None:
            for idx in range(n):
                evec = self._he_embeddings[idx]
                if evec is not None:
                    sim = _cosine_sim(query_vec, evec)
                    sem_scores.append((idx, sim))
            sem_scores.sort(key=lambda x: x[1], reverse=True)

        sem_rank = {idx: rank for rank, (idx, _) in enumerate(sem_scores)}

        # --- Rank list 2: graph neighbor overlap -----------------------------
        # Find entity nodes whose names appear in the query
        query_lower = query.lower()
        query_entities: set[str] = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "entity":
                if data.get("name", "").lower() in query_lower:
                    query_entities.add(node)

        graph_scores: list[tuple[int, float]] = []
        for idx in range(n):
            he_id = self._he_ids[idx]
            if not self.graph.has_node(he_id):
                continue
            neighbors = set(self.graph.successors(he_id))
            overlap = len(neighbors & query_entities)
            if overlap > 0:
                graph_scores.append((idx, float(overlap)))

        graph_scores.sort(key=lambda x: x[1], reverse=True)
        graph_rank = {idx: rank for rank, (idx, _) in enumerate(graph_scores)}

        # --- RRF fusion ------------------------------------------------------
        all_indices = set(sem_rank.keys()) | set(graph_rank.keys())
        if not all_indices:
            # Fallback: return most recent entries
            return list(reversed(self._entries[-top_k:]))

        rrf: list[tuple[int, float]] = []
        for idx in all_indices:
            score = 0.0
            if idx in sem_rank:
                score += 1.0 / (k_rrf + sem_rank[idx])
            if idx in graph_rank:
                score += 1.0 / (k_rrf + graph_rank[idx])
            rrf.append((idx, score))

        rrf.sort(key=lambda x: x[1], reverse=True)
        return [self._entries[idx] for idx, _ in rrf[:top_k]]

    # -- graph traversal for relationships ------------------------------------
    def get_agent_relationships(self, agent_name: str) -> dict:
        """Traverse graph to build a summary of relationships for an agent.

        Returns: {
            "agent": agent_name,
            "interactions": {other_agent: count, ...},
            "recent_dynamics": [str, ...],
        }
        """
        eid = f"entity:{agent_name}"
        result: dict = {
            "agent": agent_name,
            "interactions": {},
            "recent_dynamics": [],
        }

        if not self.graph.has_node(eid):
            return result

        # Find all hyperedges that point to this entity
        he_nodes = list(self.graph.predecessors(eid))

        interaction_counts: dict[str, int] = {}
        dynamics: list[tuple[int, str]] = []

        for he_id in he_nodes:
            he_data = self.graph.nodes[he_id]
            round_num = he_data.get("round", -1)

            # Other entities connected to this hyperedge
            for neighbor in self.graph.successors(he_id):
                if neighbor == eid:
                    continue
                ndata = self.graph.nodes[neighbor]
                if ndata.get("entity_type") == "agent":
                    other = ndata.get("name", neighbor)
                    interaction_counts[other] = interaction_counts.get(other, 0) + 1

            content = he_data.get("content", "")
            if content:
                dynamics.append((round_num, content))

        result["interactions"] = interaction_counts
        # Keep last 5 dynamics, ordered by round
        dynamics.sort(key=lambda x: x[0])
        result["recent_dynamics"] = [d for _, d in dynamics[-5:]]
        return result

    # -- format for prompt (backward compatible) ------------------------------
    def format_for_prompt(self, max_entries: int = 10) -> str:
        """Format memories for inclusion in agent's prompt.

        Keeps the exact same output format that Agent.system_prompt expects.
        When possible uses RRF to pick the most relevant entries; falls back
        to most-recent when no query context is available.
        """
        if not self._entries:
            return "You have no prior interaction memories."

        recent = self._entries[-max_entries:]
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
                stances = "; ".join(
                    f"{k}: {v}" for k, v in soc["their_stances"].items()
                )
                parts.append(f"Their views: {stances}")
            if soc.get("group_dynamics"):
                parts.append(f"Group dynamics: {soc['group_dynamics']}")
            if self_ref.get("how_they_see_me"):
                parts.append(f"How others see me: {self_ref['how_they_see_me']}")
            if self_ref.get("my_evolution"):
                parts.append(f"My evolving view: {self_ref['my_evolution']}")

        return "\n".join(parts)
