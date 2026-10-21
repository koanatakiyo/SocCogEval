"""Unit tests for the frozen C6 solo-generation control."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_c6 import create_c6_agents
from cogclaw.llm import _chat_openai_compat
from cogclaw.solo import (
    ExposureRecorder,
    SoloOrchestrator,
    make_c6_config,
    select_vllm_process,
    validate_c6_serving_metadata,
)


class FakeTokenCounter:
    path = Path("/fake/tokenizer")

    @staticmethod
    def count_request(system: str, prompt: str) -> int:
        return len((system + prompt).split())

    @staticmethod
    def count_output(text: str) -> int:
        return len(text.split())


MEMORY_JSON = json.dumps(
    {
        "episodic": {
            "event": "Private reflection",
            "context": "Solo",
            "my_response": "I reconsidered the allocation.",
        },
        "social": {
            "participants": [],
            "their_stances": {},
            "social_signals": "None",
            "group_dynamics": "None",
        },
        "self_relevant": {
            "how_they_see_me": "",
            "my_evolution": "My allocation reasoning became clearer.",
        },
    }
)


class C6Test(unittest.TestCase):
    def test_frozen_config(self):
        config = make_c6_config("greenfield")
        self.assertEqual(config.condition_id, 6)
        self.assertEqual(config.interaction.n_rounds, 30)
        self.assertEqual(config.n_agents, 8)
        self.assertTrue(config.seed.use_seed)
        self.assertTrue(config.seed.asymmetric)
        self.assertEqual(config.memory_mode, "srm")

    def test_prompt_has_no_partner_language(self):
        first = SoloOrchestrator.build_solo_prompt(1, 1, [])
        second = SoloOrchestrator.build_solo_prompt(1, 2, ["first"])
        third = SoloOrchestrator.build_solo_prompt(1, 3, ["first", "second"])
        for prompt in (first, second, third):
            lowered = prompt.lower()
            self.assertNotIn("partner", lowered)
            self.assertNotIn("conversation with", lowered)
            self.assertIn("do not address", lowered)

    def test_openai_compat_provider_usage(self):
        choice = SimpleNamespace(
            message=SimpleNamespace(content="hello"),
            finish_reason="stop",
        )
        response = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
            id="response-1",
            model="qwen3-122b-fp8",
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        )
        metadata = {}
        text = _chat_openai_compat(
            client, "prompt", "system", "qwen3-122b-fp8", 0.7, 32,
            response_metadata=metadata,
        )
        self.assertEqual(text, "hello")
        self.assertEqual(metadata["response_id"], "response-1")
        self.assertEqual(metadata["usage"]["total_tokens"], 14)

    def test_selects_vllm_process_by_endpoint_port(self):
        listing = "\n".join(
            (
                "101 python -m vllm.entrypoints.openai.api_server "
                "--served-model-name Qwen/Qwen3-8B-W4A16 --port 8267 "
                "--tensor-parallel-size 1",
                "202 python -m vllm.entrypoints.openai.api_server "
                "--served-model-name qwen3-122b-fp8 --port 9500 "
                "--tensor-parallel-size 1",
            )
        )
        selected, status = select_vllm_process(
            listing,
            endpoint="http://localhost:9500/v1",
            model="qwen3-122b-fp8",
        )
        self.assertIn("--port 9500", selected)
        self.assertIn("--tensor-parallel-size 1", selected)
        self.assertEqual(status, "matched_endpoint_port")

    def test_frozen_serving_validation(self):
        command = (
            "202 python -m vllm.entrypoints.openai.api_server "
            "--served-model-name qwen3-122b-fp8 --port 9500 "
            "--tensor-parallel-size 1 --dtype auto --max-model-len 8192 "
            "--max-num-seqs 8 --gpu-memory-utilization 0.92 "
            "--enforce-eager --gdn-prefill-backend triton"
        )
        metadata = {
            "model": "qwen3-122b-fp8",
            "selected_vllm_process": command,
            "serving_configuration": {
                "endpoint": "http://localhost:9500/v1",
                "precision": "fp8",
                "tensor_parallel_size": 1,
                "dtype_argument": "auto",
                "max_model_len": "8192",
                "max_num_seqs": "8",
                "client_batch_size": 8,
                "gpu_memory_utilization": "0.92",
                "gdn_prefill_backend": "triton",
                "enforce_eager": True,
                "vllm_process_match_status": "matched_endpoint_port",
            },
        }
        self.assertEqual(validate_c6_serving_metadata(metadata), [])
        metadata["selected_vllm_process"] = command.replace("9500", "9501")
        metadata["serving_configuration"]["endpoint"] = (
            "http://localhost:9501/v1"
        )
        metadata["serving_configuration"]["tensor_parallel_size"] = 2
        errors = validate_c6_serving_metadata(metadata)
        self.assertIn("tensor_parallel_size must equal 1", errors)

    def test_one_round_call_budget_and_artifacts(self):
        config = make_c6_config("greenfield", n_rounds=1)
        # Keep the test fast while exercising the per-agent budget.
        config.n_agents = 2
        config.run_id = "smoke_c6test"
        agents = create_c6_agents(config)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            recorder = ExposureRecorder(
                output_dir / "placeholder.jsonl",
                FakeTokenCounter(),
                run_id=config.run_id,
                scenario=config.scenario,
                replica=0,
                model=config.model,
                python_seed=6200,
            )
            recorder.set_serving_context({"serving_configuration": {"endpoint": "http://localhost:9500/v1", "precision": "fp8", "tensor_parallel_size": 1, "client_batch_policy": "serial", "vllm_command_sha256": "abc123"}})
            orchestrator = SoloOrchestrator(
                config,
                agents,
                recorder,
                replica=0,
                python_seed=6200,
                output_dir=output_dir,
                batch_size=2,
            )
            run_dir = orchestrator.setup_run_dir()

            call_count = {"content": 0, "memory": 0}

            def fake_chat(*, prompt, **_kwargs):
                metadata = _kwargs.get("response_metadata")
                if metadata is not None:
                    metadata.update(
                        {"provider": "openai_compatible", "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}
                    )
                if "Respond with valid JSON only" in prompt:
                    call_count["memory"] += 1
                    return MEMORY_JSON
                call_count["content"] += 1
                return f"private reflection {call_count['content']}"

            with (
                patch("cogclaw.solo.chat", side_effect=fake_chat),
                patch("cogclaw.memory._get_encoder", return_value=None),
                patch("cogclaw.solo.run_pre_battery", return_value={"bfi10": []}),
                patch(
                    "cogclaw.solo.run_post_battery",
                    return_value={
                        "bfi10": [],
                        "prediction": [],
                        "prediction_status": {"status": "not_applicable"},
                    },
                ),
                patch("cogclaw.solo.run_periodic_probes", return_value={}),
            ):
                orchestrator.run_experiment(probe_interval=5)

            self.assertEqual(call_count, {"content": 6, "memory": 2})
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["exposure"]["content_calls"], 6)
            self.assertEqual(
                summary["exposure"]["memory_extraction_calls"],
                2,
            )
            self.assertEqual(summary["exposure"]["provider_usage_records"], 8)
            self.assertEqual(summary["exposure"]["total_tokens_provider"], 112)
            lines = (run_dir / "exposure_calls.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 8)
            first_call = json.loads(lines[0])
            self.assertEqual(first_call["total_tokens_provider"], 14)
            self.assertEqual(first_call["serving_endpoint"], "http://localhost:9500/v1")
            self.assertEqual(first_call["served_precision"], "fp8")
            self.assertEqual(first_call["tensor_parallel_size"], 1)
            self.assertTrue(first_call["serving_metadata_sha256"])
            round_data = json.loads(
                (run_dir / "rounds" / "round_001.json").read_text()
            )
            self.assertEqual(len(round_data["reflections"]), 2)
            self.assertEqual(len(round_data["reflections"][0]["turns"]), 3)


if __name__ == "__main__":
    unittest.main()
