"""Tests for the frozen objective mathematics/logic specificity probe."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cogclaw.objective_probe import (
    build_prompt,
    normalize_boolean,
    normalize_choice,
    normalize_numeric,
    ordered_items,
    score_response,
    validate_manifest,
)


def manifest() -> dict:
    items = [
        {
            "id": "math_1",
            "domain": "math",
            "task": "gsm8k",
            "source": "fixture",
            "source_index": 0,
            "prompt": "What is 6 times 7?",
            "reference_answer": "42",
            "answer_type": "numeric",
        },
        {
            "id": "logic_1",
            "domain": "logic",
            "task": "boolean_expressions",
            "source": "fixture",
            "source_index": 0,
            "prompt": "True and not False is",
            "reference_answer": "True",
            "answer_type": "boolean",
        },
        {
            "id": "logic_2",
            "domain": "logic",
            "task": "logical_deduction_three_objects",
            "source": "fixture",
            "source_index": 0,
            "prompt": "Choose one. (A) first (B) second",
            "reference_answer": "(B)",
            "answer_type": "choice",
        },
    ]
    return {
        "n_items": 3,
        "domains": {"math": 1, "logic": 2},
        "items": items,
    }


class ObjectiveProbeTest(unittest.TestCase):
    def test_manifest_and_fixed_order(self):
        payload = manifest()
        validate_manifest(payload)
        self.assertEqual(
            [item["id"] for item in ordered_items(payload, 0)],
            ["math_1", "logic_1", "logic_2"],
        )
        self.assertEqual(
            [item["id"] for item in ordered_items(payload, 1)],
            ["logic_2", "logic_1", "math_1"],
        )

    def test_prompt_requests_json_without_reasoning(self):
        prompt = build_prompt(manifest()["items"])
        self.assertIn('{"answers":', prompt)
        self.assertIn("Do not provide explanations or reasoning", prompt)
        for item in manifest()["items"]:
            self.assertEqual(prompt.count(f"[{item['id']}]"), 1)

    def test_normalizers(self):
        self.assertEqual(str(normalize_numeric("$1,200.00")), "1200.00")
        self.assertEqual(str(normalize_numeric("1/2")), "0.5")
        self.assertEqual(normalize_boolean(" TRUE. "), "true")
        self.assertEqual(normalize_choice("(b)"), "B")
        self.assertIsNone(normalize_choice("second"))

    def test_score_response_accepts_fenced_json(self):
        text = (
            "```json\n"
            '{"answers":{"math_1":"42.0","logic_1":"true","logic_2":"B"}}'
            "\n```"
        )
        result = score_response(manifest()["items"], text)
        self.assertTrue(result["response_json_parsed"])
        self.assertEqual(result["n_present"], 3)
        self.assertEqual(result["n_correct"], 3)

    def test_parse_failure_counts_all_items_incorrect(self):
        result = score_response(manifest()["items"], "not json")
        self.assertFalse(result["response_json_parsed"])
        self.assertEqual(result["n_present"], 0)
        self.assertEqual(result["n_correct"], 0)

    def test_saved_memory_reconstruction_does_not_embed(self):
        from scripts.run_objective_specificity import load_agent

        memory_payload = {
            "group_id": 0,
            "memories": [
                {
                    "round": 1,
                    "episodic": {"event": "A discussion", "context": "test"},
                    "social": {},
                    "self_relevant": {},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            memory_dir = run_dir / "memories"
            memory_dir.mkdir()
            (memory_dir / "Alex.json").write_text(json.dumps(memory_payload))
            with patch(
                "cogclaw.memory._get_encoder",
                side_effect=AssertionError("embedding should not be called"),
            ):
                pre = load_agent(
                    run_dir, "greenfield", 0, False, "qwen3-122b-fp8"
                )
                post = load_agent(
                    run_dir, "greenfield", 0, True, "qwen3-122b-fp8"
                )
            self.assertNotIn("Your memories from past interactions", pre.system_prompt)
            self.assertIn("A discussion", post.system_prompt)


if __name__ == "__main__":
    unittest.main()

