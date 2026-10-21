"""Fixed objective mathematics/logic probe utilities.

This module contains no experiment selection or statistical decisions. It
only validates a frozen item manifest, formats the fixed prompt, parses a
response, and scores answers deterministically.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ALLOWED_DOMAINS = {"math", "logic"}
ALLOWED_ANSWER_TYPES = {"numeric", "boolean", "choice"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    validate_manifest(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def validate_manifest(manifest: dict) -> None:
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty items list")
    if manifest.get("n_items") != len(items):
        raise ValueError("Manifest n_items does not match items")

    ids: set[str] = set()
    counts = {"math": 0, "logic": 0}
    for item in items:
        required = {
            "id",
            "domain",
            "task",
            "source",
            "source_index",
            "prompt",
            "reference_answer",
            "answer_type",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"Manifest item is missing fields: {sorted(missing)}")
        if item["id"] in ids:
            raise ValueError(f"Duplicate item id: {item['id']}")
        ids.add(item["id"])
        if item["domain"] not in ALLOWED_DOMAINS:
            raise ValueError(f"Unknown domain: {item['domain']}")
        if item["answer_type"] not in ALLOWED_ANSWER_TYPES:
            raise ValueError(f"Unknown answer type: {item['answer_type']}")
        if not str(item["prompt"]).strip():
            raise ValueError(f"Empty item prompt: {item['id']}")
        counts[item["domain"]] += 1

    if manifest.get("domains") != counts:
        raise ValueError(
            f"Manifest domain counts {manifest.get('domains')} do not match {counts}"
        )


def ordered_items(manifest: dict, agent_id: int) -> list[dict]:
    items = list(manifest["items"])
    return items if agent_id % 2 == 0 else list(reversed(items))


def build_prompt(items: list[dict]) -> str:
    lines = [
        "Complete the following fixed objective check. These questions are "
        "unrelated to the committee decision.",
        "",
        "Return final answers only. Do not provide explanations or reasoning.",
        "Return exactly one JSON object with this form:",
        '{"answers": {"ITEM_ID": "FINAL_ANSWER"}}',
        "",
        "For mathematics items, give only the final number. For Boolean items, "
        'give "True" or "False". For multiple-choice logic items, give only '
        'the option label such as "(A)".',
        "",
        "Items:",
    ]
    for item in items:
        lines.extend(
            [
                "",
                f"[{item['id']}]",
                item["prompt"],
            ]
        )
    lines.extend(
        [
            "",
            "Remember: output one JSON object only, with one answer for every "
            "item ID.",
        ]
    )
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object")
    answers = payload.get("answers", payload)
    if not isinstance(answers, dict):
        raise ValueError("Response answers must be an object")
    return {str(key): value for key, value in answers.items()}


def parse_answers(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        return _extract_json_object(text), None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _strip_wrappers(value: Any) -> str:
    text = str(value).strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    return text.strip().rstrip(".")


def normalize_numeric(value: Any) -> Decimal | None:
    text = _strip_wrappers(value)
    text = text.replace(",", "").replace("$", "").replace("£", "")
    text = re.sub(r"\s+", "", text)
    if text.endswith("%"):
        text = text[:-1]
    if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", text):
        numerator, denominator = text.split("/")
        try:
            denominator_decimal = Decimal(denominator)
            if denominator_decimal == 0:
                return None
            return Decimal(numerator) / denominator_decimal
        except InvalidOperation:
            return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def normalize_boolean(value: Any) -> str | None:
    text = _strip_wrappers(value).lower()
    if text == "true":
        return "true"
    if text == "false":
        return "false"
    return None


def normalize_choice(value: Any) -> str | None:
    text = _strip_wrappers(value).upper()
    match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", text)
    return match.group(1) if match else None


def score_answer(item: dict, answer: Any) -> tuple[bool, str | None, str | None]:
    answer_type = item["answer_type"]
    reference = item["reference_answer"]
    if answer_type == "numeric":
        normalized = normalize_numeric(answer)
        expected = normalize_numeric(reference)
    elif answer_type == "boolean":
        normalized = normalize_boolean(answer)
        expected = normalize_boolean(reference)
    elif answer_type == "choice":
        normalized = normalize_choice(answer)
        expected = normalize_choice(reference)
    else:
        raise ValueError(f"Unsupported answer type: {answer_type}")
    return (
        normalized is not None and expected is not None and normalized == expected,
        None if normalized is None else str(normalized),
        None if expected is None else str(expected),
    )


def score_response(items: list[dict], response_text: str) -> dict:
    answers, parse_error = parse_answers(response_text)
    item_scores = []
    for item in items:
        present = item["id"] in answers
        raw_answer = answers.get(item["id"])
        correct, normalized, expected = (
            score_answer(item, raw_answer)
            if present
            else (False, None, str(item["reference_answer"]))
        )
        item_scores.append(
            {
                "id": item["id"],
                "domain": item["domain"],
                "task": item["task"],
                "present": present,
                "raw_answer": raw_answer,
                "normalized_answer": normalized,
                "normalized_reference": expected,
                "correct": bool(correct),
            }
        )

    return {
        "response_json_parsed": parse_error is None,
        "parse_error": parse_error,
        "n_expected": len(items),
        "n_present": sum(row["present"] for row in item_scores),
        "n_correct": sum(row["correct"] for row in item_scores),
        "item_scores": item_scores,
    }

