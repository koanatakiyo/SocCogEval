"""Build the deterministic frozen GSM8K/BBH objective-probe manifest.

The source revisions, tasks, selection sizes, and random seed are frozen in
``objective_specificity_frozen_plan.md``. The output is deterministic: the
script verifies an identical existing file and refuses to overwrite a
different one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "rebuttal" / "2026-07-25" / "objective_specificity_items.json"
)
SELECTION_SEED = 20260728
GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"

SOURCES = {
    "gsm8k_test": {
        "revision": GSM8K_REVISION,
        "url": (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            f"{GSM8K_REVISION}/grade_school_math/data/test.jsonl"
        ),
        "n_select": 12,
        "selection_offset": 0,
    },
    "bbh_boolean_expressions": {
        "revision": BBH_REVISION,
        "url": (
            "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/"
            f"{BBH_REVISION}/bbh/boolean_expressions.json"
        ),
        "n_select": 6,
        "selection_offset": 1,
    },
    "bbh_logical_deduction_three_objects": {
        "revision": BBH_REVISION,
        "url": (
            "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/"
            f"{BBH_REVISION}/bbh/logical_deduction_three_objects.json"
        ),
        "n_select": 6,
        "selection_offset": 2,
    },
}


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CogSociety frozen objective probe"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer)
    if not match:
        raise ValueError(f"GSM8K answer has no final delimiter: {answer!r}")
    return match.group(1)


def _selected_indices(n_items: int, n_select: int, offset: int) -> list[int]:
    rng = random.Random(SELECTION_SEED + offset)
    return sorted(rng.sample(range(n_items), n_select))


def build_manifest(raw_sources: dict[str, bytes]) -> dict:
    source_metadata: dict[str, dict] = {}
    items: list[dict] = []

    for source_name, spec in SOURCES.items():
        raw = raw_sources[source_name]
        source_metadata[source_name] = {
            "url": spec["url"],
            "revision": spec["revision"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

        if source_name == "gsm8k_test":
            examples = [
                json.loads(line)
                for line in raw.decode("utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(raw.decode("utf-8"))
            examples = payload["examples"]

        indices = _selected_indices(
            len(examples),
            spec["n_select"],
            spec["selection_offset"],
        )
        for index in indices:
            example = examples[index]
            if source_name == "gsm8k_test":
                item = {
                    "id": f"gsm8k_test_{index:04d}",
                    "domain": "math",
                    "task": "gsm8k",
                    "source": source_name,
                    "source_index": index,
                    "prompt": example["question"].strip(),
                    "reference_answer": _gsm8k_answer(example["answer"]),
                    "answer_type": "numeric",
                }
            elif source_name == "bbh_boolean_expressions":
                item = {
                    "id": f"bbh_boolean_{index:03d}",
                    "domain": "logic",
                    "task": "boolean_expressions",
                    "source": source_name,
                    "source_index": index,
                    "prompt": example["input"].strip(),
                    "reference_answer": example["target"].strip(),
                    "answer_type": "boolean",
                }
            else:
                item = {
                    "id": f"bbh_logical_deduction_3_{index:03d}",
                    "domain": "logic",
                    "task": "logical_deduction_three_objects",
                    "source": source_name,
                    "source_index": index,
                    "prompt": example["input"].strip(),
                    "reference_answer": example["target"].strip(),
                    "answer_type": "choice",
                }
            items.append(item)

    return {
        "protocol": "objective_specificity_frozen_plan.md",
        "frozen_date": "2026-07-28",
        "selection_seed": SELECTION_SEED,
        "selection": {
            "gsm8k_test": 12,
            "bbh_boolean_expressions": 6,
            "bbh_logical_deduction_three_objects": 6,
        },
        "sources": source_metadata,
        "n_items": len(items),
        "domains": {"math": 12, "logic": 12},
        "items": items,
    }


def encoded_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_sources = {
        source_name: download(spec["url"])
        for source_name, spec in SOURCES.items()
    }
    manifest = build_manifest(raw_sources)
    encoded = encoded_manifest(manifest)

    if args.output.exists():
        existing = args.output.read_bytes()
        if existing != encoded:
            raise RuntimeError(
                f"Refusing to replace a different frozen manifest: {args.output}"
            )
        print(
            f"Verified existing frozen manifest {args.output} "
            f"sha256={hashlib.sha256(existing).hexdigest()}"
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        f"Wrote frozen manifest {args.output} "
        f"sha256={hashlib.sha256(encoded).hexdigest()}"
    )


if __name__ == "__main__":
    main()

