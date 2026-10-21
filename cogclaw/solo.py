"""C6 matched solo-generation control.

This module implements the rebuttal protocol frozen in
``rebuttal/2026-07-25/c6_frozen_protocol.md``.  It deliberately lives outside
the C1--C5 conversation orchestrator so the submitted experiment path remains
unchanged.

C6 preserves the C2 exposure budget:

* 24 content generations per round (3 per agent);
* 8 SRM extraction calls per round (1 per agent);
* the same 30-round horizon and five-round probe cadence.

The treatment removed from C6 is partner exposure: prompts contain no partner
identity, partner message, simulated dialogue, or adaptive feedback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from . import config as cfg
from .agent import Agent
from .config import (
    DATA_DIR,
    ExperimentConfig,
    InteractionConfig,
    NetworkConfig,
    SeedConfig,
)
from .llm import _parse_json, chat
from .memory import EXTRACTION_PROMPT, _empty_memory, _validate_memory
from .probes import run_periodic_probes, run_post_battery, run_pre_battery

logger = logging.getLogger(__name__)

JSON_ONLY_SUFFIX = "\n\nRespond with valid JSON only. No markdown, no explanation."

SOLO_PROMPTS = (
    (
        "This is round {round_num} of a private reflection exercise about the "
        "committee topic. Develop your current reasoning about the allocation "
        "problem. Consider the values, evidence, and tradeoffs that matter to "
        "you. Do not address, quote, predict, or simulate another person. "
        "Write only your own reflection."
    ),
    (
        "Continue your private reflection. Reconsider your reasoning from "
        "another angle, identify a genuine tension or uncertainty, and explain "
        "how you would resolve it. Do not address, quote, predict, or simulate "
        "another person.\n\n"
        "Your reflection so far:\n"
        "{own_turn_1}"
    ),
    (
        "Complete this round's private reflection by stating your current best "
        "allocation reasoning and what, if anything, changed in your own view. "
        "Do not address, quote, predict, or simulate another person.\n\n"
        "Your reflection so far:\n"
        "{own_turn_1}\n"
        "{own_turn_2}"
    ),
)


def sha_text(text: str) -> str:
    """Return a stable SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def make_c6_config(
    scenario: str,
    model: str = "",
    n_rounds: int = 30,
) -> ExperimentConfig:
    """Build the frozen C6 condition configuration."""
    if scenario not in {"greenfield", "riverton"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    if n_rounds < 1:
        raise ValueError("C6 requires at least one round")

    return ExperimentConfig(
        condition_id=6,
        condition_name="solo_generation",
        scenario=scenario,
        n_agents=8,
        # No network is instantiated for C6.  Two groups are retained only as
        # an audit label for the A/B brief assignment.
        network=NetworkConfig(topology="none", n_groups=2, group_size=4),
        seed=SeedConfig(use_seed=True, asymmetric=True),
        interaction=InteractionConfig(
            n_rounds=n_rounds,
            agents_per_conversation=1,
            conversations_per_round=0,
        ),
        model=model or cfg.PRIMARY_MODEL,
        n_repeats=5,
        use_s2=False,
        memory_mode="srm",
    )


class TokenCounter:
    """Count request/output tokens with the exact local tokenizer."""

    def __init__(self, tokenizer_path: Path):
        from transformers import AutoTokenizer

        self.path = tokenizer_path.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path),
            trust_remote_code=True,
            local_files_only=True,
        )

    def count_request(self, system: str, prompt: str) -> int:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return len(token_ids)
        except (TypeError, ValueError):
            # Older tokenizer templates may not accept enable_thinking.
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(token_ids)

    def count_output(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class ExposureRecorder:
    """Append one auditable JSON record per C6 content/extraction call."""

    def __init__(
        self,
        path: Path,
        token_counter: TokenCounter,
        *,
        run_id: str,
        scenario: str,
        replica: int,
        model: str,
        python_seed: int,
    ):
        self.path = path
        self.token_counter = token_counter
        self.run_id = run_id
        self.scenario = scenario
        self.replica = replica
        self.model = model
        self.python_seed = python_seed
        self.call_index = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.content_calls = 0
        self.extraction_calls = 0
        self.provider_usage_records = 0
        self.provider_prompt_tokens = 0
        self.provider_completion_tokens = 0
        self.provider_total_tokens = 0
        self.serving_metadata_sha256 = ""
        self.serving_context = {
            "serving_endpoint": "not_recorded",
            "served_precision": "not_recorded",
            "tensor_parallel_size": None,
            "client_batch_policy": "not_recorded",
            "client_batch_size": None,
            "server_max_num_seqs": None,
            "vllm_command_sha256": "not_recorded",
        }
        self._record_lock = threading.Lock()

    def set_serving_context(self, metadata: dict) -> None:
        """Attach stable run-level serving configuration to every call row."""
        configuration = metadata.get("serving_configuration", {})
        self.serving_context = {
            "serving_endpoint": configuration.get("endpoint", "not_recorded"),
            "served_precision": configuration.get("precision", "not_recorded"),
            "tensor_parallel_size": configuration.get("tensor_parallel_size"),
            "client_batch_policy": configuration.get(
                "client_batch_policy", "not_recorded"
            ),
            "client_batch_size": configuration.get("client_batch_size"),
            "server_max_num_seqs": configuration.get("max_num_seqs"),
            "vllm_command_sha256": configuration.get(
                "vllm_command_sha256", "not_recorded"
            ),
        }
        self.serving_metadata_sha256 = sha_text(canonical_json(metadata))

    def record_threadsafe(self, **kwargs: Any) -> None:
        """Serialize token counting, counters, and JSONL writes."""
        with self._record_lock:
            self.record(**kwargs)

    def record(
        self,
        *,
        kind: str,
        agent: Agent,
        round_num: int,
        turn: Optional[int],
        system: str,
        prompt: str,
        output_text: str,
        temperature: float,
        max_tokens: int,
        duration_seconds: float,
        provider_metadata: Optional[dict] = None,
    ) -> None:
        self.call_index += 1
        input_tokens = self.token_counter.count_request(system, prompt)
        output_tokens = self.token_counter.count_output(output_text)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if kind == "content":
            self.content_calls += 1
        elif kind == "memory_extraction":
            self.extraction_calls += 1
        else:
            raise ValueError(f"Unknown exposure kind: {kind}")

        provider_metadata = provider_metadata or {}
        usage = provider_metadata.get("usage", {}) or {}
        provider_prompt = usage.get("prompt_tokens")
        provider_completion = usage.get("completion_tokens")
        provider_total = usage.get("total_tokens")
        if any(isinstance(value, int) for value in usage.values()):
            self.provider_usage_records += 1
        if isinstance(provider_prompt, int):
            self.provider_prompt_tokens += provider_prompt
        if isinstance(provider_completion, int):
            self.provider_completion_tokens += provider_completion
        if isinstance(provider_total, int):
            self.provider_total_tokens += provider_total

        row = {
            "call_index": self.call_index,
            "kind": kind,
            "run_id": self.run_id,
            "scenario": self.scenario,
            "replica": self.replica,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "round": round_num,
            "turn": turn,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # The submitted C2 path did not send per-request sampling seeds.
            # C6 matches that behavior and records the absence explicitly.
            "sampling_seed": None,
            "python_replica_seed": self.python_seed,
            "system_sha256": sha_text(system),
            "user_prompt_sha256": sha_text(prompt),
            "output_sha256": sha_text(output_text),
            "input_tokens_local": input_tokens,
            "output_tokens_local": output_tokens,
            "input_tokens_provider": provider_prompt,
            "output_tokens_provider": provider_completion,
            "total_tokens_provider": provider_total,
            "input_token_delta_provider_minus_local": (
                provider_prompt - input_tokens
                if isinstance(provider_prompt, int) else None
            ),
            "output_token_delta_provider_minus_local": (
                provider_completion - output_tokens
                if isinstance(provider_completion, int) else None
            ),
            "provider_response_metadata": provider_metadata,
            "serving_metadata_sha256": self.serving_metadata_sha256,
            **self.serving_context,
            "output_chars": len(output_text),
            "duration_seconds": round(duration_seconds, 6),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tokenizer_path": str(self.token_counter.path),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def totals(self) -> dict[str, int]:
        return {
            "content_calls": self.content_calls,
            "memory_extraction_calls": self.extraction_calls,
            "input_tokens_local": self.total_input_tokens,
            "output_tokens_local": self.total_output_tokens,
            "provider_usage_records": self.provider_usage_records,
            "input_tokens_provider": self.provider_prompt_tokens,
            "output_tokens_provider": self.provider_completion_tokens,
            "total_tokens_provider": self.provider_total_tokens,
        }

@dataclass
class SoloReflectionRecord:
    """All three private reflections produced by one agent in one round."""

    agent_id: int
    agent_name: str
    turns: list[dict] = field(default_factory=list)


@dataclass
class SoloRoundRecord:
    """Saved C6 state for one round."""

    round_num: int
    reflections: list[SoloReflectionRecord] = field(default_factory=list)
    probe_responses: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


class SoloOrchestrator:
    """Run a single C6 replica."""

    def __init__(
        self,
        config: ExperimentConfig,
        agents: list[Agent],
        recorder: ExposureRecorder,
        *,
        replica: int,
        python_seed: int,
        output_dir: Path = DATA_DIR,
        batch_size: int = 1,
    ):
        if config.condition_id != 6:
            raise ValueError("SoloOrchestrator requires a C6 config")
        if config.memory_mode != "srm":
            raise ValueError("Frozen C6 requires memory_mode='srm'")
        if batch_size < 1 or batch_size > len(agents):
            raise ValueError("batch_size must be between 1 and the agent count")
        self.config = config
        self.agents = agents
        self.recorder = recorder
        self.replica = replica
        self.python_seed = python_seed
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.rounds: list[SoloRoundRecord] = []
        self.run_dir: Optional[Path] = None

    def setup_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.config.condition_name}_{self.config.scenario}_{timestamp}"
        if self.config.run_id:
            name = f"{self.config.run_id}_{name}"
        self.run_dir = self.output_dir / name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.config.save(self.run_dir / "config.json")
        # The recorder is created before the timestamped run directory exists.
        self.recorder.path = self.run_dir / "exposure_calls.jsonl"
        return self.run_dir

    @staticmethod
    def build_solo_prompt(
        round_num: int,
        turn: int,
        prior_outputs: list[str],
    ) -> str:
        if turn == 1:
            return SOLO_PROMPTS[0].format(round_num=round_num)
        if turn == 2:
            if len(prior_outputs) != 1:
                raise ValueError("Turn 2 requires exactly one prior output")
            return SOLO_PROMPTS[1].format(
                round_num=round_num,
                own_turn_1=prior_outputs[0],
            )
        if turn == 3:
            if len(prior_outputs) != 2:
                raise ValueError("Turn 3 requires exactly two prior outputs")
            return SOLO_PROMPTS[2].format(
                round_num=round_num,
                own_turn_1=prior_outputs[0],
                own_turn_2=prior_outputs[1],
            )
        raise ValueError(f"C6 has exactly three turns, got turn={turn}")

    def _content_call(
        self,
        agent: Agent,
        round_num: int,
        turn: int,
        prior_outputs: list[str],
    ) -> tuple[str, str]:
        prompt = self.build_solo_prompt(round_num, turn, prior_outputs)
        system = agent.system_prompt
        provider_metadata: dict = {}
        t0 = time.time()
        output = chat(
            prompt=prompt,
            system=system,
            model=agent.model,
            temperature=cfg.CONVERSATION_TEMPERATURE,
            response_metadata=provider_metadata,
            max_tokens=cfg.CONVERSATION_MAX_TOKENS,
        )
        self.recorder.record_threadsafe(
            kind="content",
            agent=agent,
            round_num=round_num,
            turn=turn,
            system=system,
            prompt=prompt,
            output_text=output,
            temperature=cfg.CONVERSATION_TEMPERATURE,
            provider_metadata=provider_metadata,
            max_tokens=cfg.CONVERSATION_MAX_TOKENS,
            duration_seconds=time.time() - t0,
        )
        return prompt, output

    def _extract_solo_memory(
        self,
        agent: Agent,
        round_num: int,
        outputs: list[str],
    ) -> dict:
        # Keep the submitted SRM extractor unchanged.  The input contains only
        # this agent's private reflections and therefore exposes no partner
        # identity or partner message.
        reflection_text = "\n".join(f"{agent.name}: {text}" for text in outputs)
        extraction_prompt = EXTRACTION_PROMPT.format(
            agent_name=agent.name,
            round_num=round_num,
            conversation_text=reflection_text,
        )
        assembled_prompt = extraction_prompt + JSON_ONLY_SUFFIX
        provider_metadata: dict = {}
        t0 = time.time()
        try:
            raw = chat(
                prompt=assembled_prompt,
                model=agent.model,
                temperature=0.0,
                response_metadata=provider_metadata,
                max_tokens=1024,
            )
            memory = _parse_json(raw)
        except Exception as exc:
            logger.error(
                "C6 memory extraction failed for %s round %d: %s",
                agent.name,
                round_num,
                exc,
            )
            raw = canonical_json(_empty_memory())
            memory = _empty_memory()

        memory = _validate_memory(memory)
        memory["round"] = round_num
        self.recorder.record_threadsafe(
            kind="memory_extraction",
            agent=agent,
            round_num=round_num,
            turn=None,
            system="",
            prompt=assembled_prompt,
            output_text=raw,
            temperature=0.0,
            max_tokens=1024,
            provider_metadata=provider_metadata,
            duration_seconds=time.time() - t0,
        )
        agent.memory.add(memory)
        return memory

    def run_round(self, round_num: int) -> SoloRoundRecord:
        t0 = time.time()
        record = SoloRoundRecord(round_num=round_num)
        logger.info(
            "C6 round %d: %d agents x 3 private reflections (batch=%d)",
            round_num,
            len(self.agents),
            self.batch_size,
        )

        reflections_by_id = {
            agent.agent_id: SoloReflectionRecord(
                agent_id=agent.agent_id,
                agent_name=agent.name,
            )
            for agent in self.agents
        }
        outputs_by_id = {agent.agent_id: [] for agent in self.agents}

        # Each turn is a barrier: the eight agents are independent within a
        # turn, while turn 2/3 retain each agent's own earlier output.
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            for turn in (1, 2, 3):
                futures = {
                    agent.agent_id: executor.submit(
                        self._content_call,
                        agent,
                        round_num,
                        turn,
                        list(outputs_by_id[agent.agent_id]),
                    )
                    for agent in self.agents
                }
                for agent in self.agents:
                    prompt, output = futures[agent.agent_id].result()
                    outputs_by_id[agent.agent_id].append(output)
                    reflections_by_id[agent.agent_id].turns.append(
                        {
                            "turn": turn,
                            "prompt_sha256": sha_text(prompt),
                            "text": output,
                        }
                    )

            extraction_futures = {
                agent.agent_id: executor.submit(
                    self._extract_solo_memory,
                    agent,
                    round_num,
                    list(outputs_by_id[agent.agent_id]),
                )
                for agent in self.agents
            }
            for agent in self.agents:
                extraction_futures[agent.agent_id].result()

        record.reflections = [
            reflections_by_id[agent.agent_id]
            for agent in self.agents
        ]

        record.duration_seconds = time.time() - t0
        self.rounds.append(record)
        return record

    def run_experiment(
        self,
        *,
        probe_interval: int = 5,
    ) -> list[SoloRoundRecord]:
        if self.run_dir is None:
            self.setup_run_dir()

        logger.info("Running C6 pre-test battery")
        self._save_json(
            run_pre_battery(self.agents, self.config.scenario),
            "pre_measurements.json",
        )

        n_rounds = self.config.interaction.n_rounds
        for round_num in range(1, n_rounds + 1):
            record = self.run_round(round_num)
            if round_num % probe_interval == 0:
                logger.info("Running C6 periodic probes at round %d", round_num)
                record.probe_responses = run_periodic_probes(
                    self.agents,
                    round_num,
                    self.config.scenario,
                )
            self._save_round(record)
            logger.info(
                "C6 round %d/%d complete (%.1fs)",
                round_num,
                n_rounds,
                record.duration_seconds,
            )

        # Prediction requires observations of another agent and is therefore
        # not applicable to C6.
        logger.info("Running C6 post-test battery without prediction")
        self._save_json(
            run_post_battery(
                self.agents,
                self.config.scenario,
                include_prediction=False,
            ),
            "post_measurements.json",
        )
        self._save_final()
        return self.rounds

    def _save_json(self, data: Any, filename: str) -> None:
        if self.run_dir is None:
            raise RuntimeError("run directory is not initialized")
        with (self.run_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _save_round(self, record: SoloRoundRecord) -> None:
        if self.run_dir is None:
            raise RuntimeError("run directory is not initialized")
        round_dir = self.run_dir / "rounds"
        round_dir.mkdir(exist_ok=True)
        data = {
            "round_num": record.round_num,
            "duration_seconds": record.duration_seconds,
            "reflections": [
                {
                    "agent_id": reflection.agent_id,
                    "agent_name": reflection.agent_name,
                    "turns": reflection.turns,
                }
                for reflection in record.reflections
            ],
            "probe_responses": record.probe_responses,
        }
        with (round_dir / f"round_{record.round_num:03d}.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _save_final(self) -> None:
        if self.run_dir is None:
            raise RuntimeError("run directory is not initialized")
        memories_dir = self.run_dir / "memories"
        memories_dir.mkdir(exist_ok=True)
        for agent in self.agents:
            payload = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "group_id": agent.group_id,
                "memories": agent.memory.to_list(),
            }
            with (memories_dir / f"{agent.name}.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

        expected_content = (
            self.config.interaction.n_rounds * self.config.n_agents * 3
        )
        expected_extractions = (
            self.config.interaction.n_rounds * self.config.n_agents
        )
        totals = self.recorder.totals()
        if totals["content_calls"] != expected_content:
            raise RuntimeError(
                f"C6 content-call mismatch: {totals['content_calls']} "
                f"!= {expected_content}"
            )
        if totals["memory_extraction_calls"] != expected_extractions:
            raise RuntimeError(
                "C6 memory-extraction mismatch: "
                f"{totals['memory_extraction_calls']} != {expected_extractions}"
            )

        summary = {
            "config": self.config.to_dict(),
            "protocol": "c6_matched_solo_generation_batch8_amended_2026-07-28",
            "protocol_amendment": "throughput-only barriered request scheduling",
            "batch_size": self.batch_size,
            "replica": self.replica,
            "python_seed": self.python_seed,
            "sampling_seed": None,
            "n_rounds_completed": len(self.rounds),
            "total_conversations": 0,
            "total_solo_reflections": sum(
                len(record.reflections) for record in self.rounds
            ),
            "total_duration_seconds": sum(
                record.duration_seconds for record in self.rounds
            ),
            "exposure": totals,
            "agents": [
                {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "group_id": agent.group_id,
                    "n_memories": len(agent.memory.entries),
                }
                for agent in self.agents
            ],
        }
        self._save_json(summary, "summary.json")
        logger.info("C6 experiment saved to %s", self.run_dir)


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

def command_option(command_text: str, option: str) -> Optional[str]:
    """Extract a ``--flag value`` or ``--flag=value`` from a command line."""
    match = re.search(
        rf"(?:^|\s){re.escape(option)}(?:=|\s+)(\S+)",
        command_text,
    )
    return match.group(1) if match else None


def select_vllm_process(
    process_listing: str,
    *,
    endpoint: str,
    model: str,
) -> tuple[str, str]:
    """Select the vLLM process serving the requested C6 endpoint."""
    lines = [
        line.strip()
        for line in process_listing.splitlines()
        if "vllm.entrypoints.openai.api_server" in line
    ]
    try:
        endpoint_port = urlparse(endpoint).port
    except ValueError:
        endpoint_port = None

    if endpoint_port is not None:
        port_matches = [
            line
            for line in lines
            if command_option(line, "--port") == str(endpoint_port)
        ]
        if len(port_matches) == 1:
            return port_matches[0], "matched_endpoint_port"
        if len(port_matches) > 1:
            model_matches = [
                line
                for line in port_matches
                if command_option(line, "--served-model-name") == model
            ]
            if len(model_matches) == 1:
                return model_matches[0], "matched_endpoint_port_and_model"
            return "", "ambiguous_endpoint_port"

    model_matches = [
        line
        for line in lines
        if command_option(line, "--served-model-name") == model
    ]
    if len(model_matches) == 1:
        return model_matches[0], "matched_served_model"
    if len(model_matches) > 1:
        return "", "ambiguous_served_model"
    return "", "no_matching_vllm_process"



def validate_c6_serving_metadata(metadata: dict) -> list[str]:
    """Return protocol violations that must block a C6 run."""
    configuration = metadata.get("serving_configuration", {})
    selected = metadata.get("selected_vllm_process", "")
    errors: list[str] = []

    if configuration.get("vllm_process_match_status") not in {
        "matched_endpoint_port",
        "matched_endpoint_port_and_model",
    }:
        errors.append("no unique vLLM process matched the configured endpoint")
    if command_option(selected, "--served-model-name") != metadata.get("model"):
        errors.append("served model name does not match the requested C6 model")
    if configuration.get("tensor_parallel_size") != 1:
        errors.append("tensor_parallel_size must equal 1")
    if str(configuration.get("max_model_len")) != "8192":
        errors.append("max_model_len must equal 8192")
    if str(configuration.get("max_num_seqs")) != "8":
        errors.append("amended C6 max_num_seqs must equal 8")
    if configuration.get("client_batch_size") != 8:
        errors.append("amended C6 client batch size must equal 8")
    if configuration.get("dtype_argument") != "auto":
        errors.append("dtype must equal auto")
    if str(configuration.get("gpu_memory_utilization")) != "0.92":
        errors.append("gpu_memory_utilization must equal 0.92")
    if configuration.get("gdn_prefill_backend") != "triton":
        errors.append("gdn_prefill_backend must equal triton")
    if configuration.get("enforce_eager") is not True:
        errors.append("enforce_eager must be enabled")
    if str(configuration.get("precision", "")).lower() != "fp8":
        errors.append("served checkpoint precision must be fp8")
    try:
        endpoint_port = urlparse(configuration.get("endpoint", "")).port
    except ValueError:
        endpoint_port = None
    if endpoint_port not in {9500, 9501}:
        errors.append("C6 endpoint port must equal 9500 or 9501")
    return errors


def build_serving_metadata(
    *,
    tokenizer_path: Path,
    model: str,
    scenario: str,
    replica: int,
    python_seed: int,
    batch_size: int,
    argv: list[str],
) -> dict:
    """Capture prospective C6 serving/code metadata before model calls."""
    root = Path(__file__).resolve().parent.parent
    model_files = {}
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        path = tokenizer_path / filename
        model_files[filename] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }

    code_files = {}
    for relative in (
        "cogclaw/solo.py",
        "cogclaw/llm.py",
        "cogclaw/agent.py",
        "cogclaw/memory.py",
        "cogclaw/probes.py",
        "cogclaw/config.py",
        "run_c6.py",
        "scripts/launch_c6.sh",
        "scripts/serve_c6_qwen.sh",
    ):
        path = root / relative
        code_files[relative] = file_sha256(path)

    package_versions = {}
    for package in ("vllm", "torch", "transformers", "openai"):
        try:
            module = __import__(package)
            package_versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            package_versions[package] = "not_installed"

    torch_module = sys.modules.get("torch")
    package_versions["cuda"] = (
        getattr(getattr(torch_module, "version", None), "cuda", None)
        if torch_module is not None
        else None
    )

    model_config_path = tokenizer_path / "config.json"
    try:
        with model_config_path.open(encoding="utf-8") as handle:
            model_config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        model_config = {}
    quantization = model_config.get("quantization_config", {}) or {}
    precision = (
        quantization.get("quant_method")
        or model_config.get("torch_dtype")
        or "not_recorded"
    )
    vllm_processes = command_output(
        [
            "pgrep",
            "-af",
            "vllm.entrypoints.openai.api_server",
        ]
    )
    selected_vllm_process, process_match_status = select_vllm_process(
        vllm_processes,
        endpoint=cfg.QWEN_BASE_URL,
        model=model,
    )
    tp_value = command_option(
        selected_vllm_process,
        "--tensor-parallel-size",
    )
    try:
        tensor_parallel_size = int(tp_value) if tp_value is not None else None
    except ValueError:
        tensor_parallel_size = None
    serving_configuration = {
        "endpoint": cfg.QWEN_BASE_URL,
        "precision": precision,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype_argument": command_option(selected_vllm_process, "--dtype"),
        "max_model_len": command_option(
            selected_vllm_process,
            "--max-model-len",
        ),
        "gpu_memory_utilization": command_option(
            selected_vllm_process,
            "--gpu-memory-utilization",
        ),
        "gdn_prefill_backend": command_option(
            selected_vllm_process,
            "--gdn-prefill-backend",
        ),
        "max_num_seqs": command_option(
            selected_vllm_process,
            "--max-num-seqs",
        ),
        "enforce_eager": "--enforce-eager" in selected_vllm_process,
        "client_batch_policy": (
            "one C6 process; four barriers per round; up to eight "
            "agent-independent requests issued concurrently per barrier"
        ),
        "client_batch_size": batch_size,
        "server_batch_policy": "vLLM dynamic batching; max_num_seqs=8",
        "co_tenant_requests": "dedicated endpoint; one C6 process only",
        "vllm_process_match_status": process_match_status,
        "vllm_command_sha256": (
            sha_text(selected_vllm_process) if selected_vllm_process else ""
        ),
    }

    return {
        "protocol": "c6_matched_solo_generation_batch8_amended_2026-07-28",
        "protocol_amendment": "throughput-only barriered request scheduling",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": argv,
        "scenario": scenario,
        "replica": replica,
        "python_seed": python_seed,
        "sampling_seed": None,
        "model": model,
        "base_url": cfg.QWEN_BASE_URL,
        "serving_configuration": serving_configuration,
        "token_count_policy": (
            "provider usage plus exact local-tokenizer recount for cross-check"
        ),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "model_files": model_files,
        "code_files": code_files,
        "packages": package_versions,
        "git_head": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status": command_output(
            ["git", "-C", str(root), "status", "--short"]
        ),
        "vllm_processes": vllm_processes,
        "selected_vllm_process": selected_vllm_process,
        "gpu_snapshot": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "decoding": {
            "content_temperature": cfg.CONVERSATION_TEMPERATURE,
            "content_max_tokens": cfg.CONVERSATION_MAX_TOKENS,
            "memory_temperature": 0.0,
            "memory_max_tokens": 1024,
            "thinking_enabled": False,
        },
    }
