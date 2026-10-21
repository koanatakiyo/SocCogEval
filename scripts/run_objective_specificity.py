"""Run the frozen objective mathematics/logic probe on saved C2/C4 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cogclaw import config as cfg
from cogclaw.agent import AGENT_NAMES, Agent
from cogclaw.llm import chat
from cogclaw.objective_probe import (
    build_prompt,
    load_manifest,
    ordered_items,
    score_response,
    sha256_text,
)
from scenarios import get_scenario


DATA = ROOT / "data"
EVIDENCE = ROOT / "rebuttal" / "2026-07-25"
DEFAULT_MANIFEST = EVIDENCE / "objective_specificity_items.json"
CONDITION_SPECS = {
    "C2": ("segreg_asym_seed", 8),
    "C4": ("seed_only_no_interact", 5),
}
WRITE_LOCK = threading.Lock()


def selected_runs(
    scenario: str,
    conditions: list[str],
    replica_limit_per_condition: int | None = None,
) -> list[tuple[str, int, Path]]:
    selected: list[tuple[str, int, Path]] = []
    for condition in conditions:
        condition_name, n_reps = CONDITION_SPECS[condition]
        if replica_limit_per_condition is not None:
            n_reps = min(n_reps, replica_limit_per_condition)
        for replica in range(n_reps):
            pattern = f"rep{replica:02d}_{condition_name}_{scenario}_*"
            matches = sorted(
                path
                for path in DATA.glob(pattern)
                if (path / "summary.json").exists()
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one complete {scenario} {condition} "
                    f"replica {replica}; found {matches}"
                )
            selected.append((condition, replica, matches[0]))
    return selected


def load_agent(
    run_dir: Path,
    scenario_name: str,
    agent_id: int,
    post: bool,
    model: str,
) -> Agent:
    scenario = get_scenario(scenario_name)
    memory_path = run_dir / "memories" / f"{AGENT_NAMES[agent_id]}.json"
    payload = json.loads(memory_path.read_text())
    group_id = payload.get("group_id")
    if group_id not in (0, 1):
        group_id = 0 if agent_id < 4 else 1
    seed_info = scenario.SEED_A if group_id == 0 else scenario.SEED_B
    agent = Agent(
        agent_id=agent_id,
        name=AGENT_NAMES[agent_id],
        scenario_context=scenario.CONTEXT,
        model=model,
        seed_info=seed_info,
    )
    agent.group_id = group_id
    if post:
        # Only Agent.system_prompt -> HypergraphMemory.format_for_prompt is
        # needed. Loading the already serialized entries directly preserves
        # that text while avoiding irrelevant embedding recomputation.
        agent.memory._entries.extend(payload.get("memories", []))
    return agent


def run_probe_call(
    agent: Agent,
    items: list[dict],
    prompt: str,
    max_tokens: int,
) -> dict:
    metadata: dict = {}
    started = time.time()
    response = chat(
        prompt=prompt,
        system=agent.system_prompt,
        model=agent.model,
        temperature=0.0,
        max_tokens=max_tokens,
        response_metadata=metadata,
    )
    duration = time.time() - started
    scored = score_response(items, response)
    return {
        "response_text": response,
        "response_sha256": sha256_text(response),
        "duration_seconds": duration,
        "provider_metadata": metadata,
        **scored,
    }


def probe_one(
    condition: str,
    replica: int,
    run_dir: Path,
    scenario_name: str,
    agent_id: int,
    model: str,
    manifest: dict,
    manifest_sha256: str,
    max_tokens: int,
) -> dict:
    items = ordered_items(manifest, agent_id)
    prompt = build_prompt(items)
    pre_agent = load_agent(run_dir, scenario_name, agent_id, False, model)
    post_agent = load_agent(run_dir, scenario_name, agent_id, True, model)

    pre = run_probe_call(pre_agent, items, prompt, max_tokens)
    post = run_probe_call(post_agent, items, prompt, max_tokens)
    return {
        "schema_version": 1,
        "protocol": "objective_specificity_frozen_plan.md",
        "scenario": scenario_name,
        "condition": condition,
        "replica": replica,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "agent_id": agent_id,
        "agent_name": AGENT_NAMES[agent_id],
        "model": model,
        "endpoint": cfg.QWEN_BASE_URL,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "order": "manifest" if agent_id % 2 == 0 else "reverse_manifest",
        "ordered_item_ids": [item["id"] for item in items],
        "manifest_sha256": manifest_sha256,
        "prompt_sha256": sha256_text(prompt),
        "pre_system_sha256": sha256_text(pre_agent.system_prompt),
        "post_system_sha256": sha256_text(post_agent.system_prompt),
        "pre": pre,
        "post": post,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def task_key(row: dict) -> tuple[str, int, int]:
    return row["condition"], int(row["replica"]), int(row["agent_id"])


def load_completed(output: Path, manifest_sha256: str) -> dict[tuple, dict]:
    rows: dict[tuple, dict] = {}
    if not output.exists():
        return rows
    for line_number, line in enumerate(output.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed checkpoint line {line_number} in {output}"
            ) from exc
        if row.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(
                f"Checkpoint manifest mismatch on line {line_number}: {output}"
            )
        key = task_key(row)
        if key in rows:
            raise RuntimeError(f"Duplicate completed task {key} in {output}")
        rows[key] = row
    return rows


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".metadata.json")


def write_metadata(
    args: argparse.Namespace,
    manifest_sha256: str,
    runs: list[tuple[str, int, Path]],
    expected_keys: set[tuple[str, int, int]],
    n_complete: int,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": "objective_specificity_frozen_plan.md",
        "scenario": args.scenario,
        "conditions": args.conditions,
        "model": args.model,
        "endpoint": cfg.QWEN_BASE_URL,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit(),
        "selected_runs": [
            {
                "condition": condition,
                "replica": replica,
                "run_dir": str(run_dir.relative_to(ROOT)),
            }
            for condition, replica, run_dir in runs
        ],
        "agents_per_run": args.agent_limit,
        "expected_agent_rows": len(expected_keys),
        "complete_agent_rows": n_complete,
        "complete": n_complete == len(expected_keys),
        "independent_unit": "run",
        "created_or_updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "qwen_base_url": os.environ.get("QWEN_BASE_URL", ""),
        },
    }
    path = metadata_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("greenfield", "riverton"), required=True
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITION_SPECS),
        default=["C2", "C4"],
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=cfg.PRIMARY_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--replica-limit-per-condition",
        type=int,
        default=None,
        help="Smoke only. Formal runs must leave this unset.",
    )
    parser.add_argument(
        "--agent-limit",
        type=int,
        choices=range(1, 9),
        default=8,
        help="Smoke only. Formal runs must use 8.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    manifest, manifest_sha256 = load_manifest(args.manifest)
    runs = selected_runs(
        args.scenario,
        args.conditions,
        args.replica_limit_per_condition,
    )
    tasks = [
        (
            condition,
            replica,
            run_dir,
            args.scenario,
            agent_id,
            args.model,
            manifest,
            manifest_sha256,
            args.max_tokens,
        )
        for condition, replica, run_dir in runs
        for agent_id in range(args.agent_limit)
    ]
    expected_keys = {
        (condition, replica, agent_id)
        for condition, replica, _run_dir in runs
        for agent_id in range(args.agent_limit)
    }
    completed = load_completed(args.output, manifest_sha256)
    unexpected = set(completed) - expected_keys
    if unexpected:
        raise RuntimeError(
            f"Output contains tasks outside this invocation: {sorted(unexpected)}"
        )
    pending = [task for task in tasks if task[:2] + (task[4],) not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_metadata(
        args, manifest_sha256, runs, expected_keys, len(completed)
    )
    print(
        f"{args.scenario}: {len(completed)}/{len(tasks)} agent rows complete; "
        f"{len(pending)} pending; endpoint={cfg.QWEN_BASE_URL}",
        flush=True,
    )
    if not pending:
        print("Nothing to do; checkpoint is complete.", flush=True)
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(probe_one, *task) for task in pending]
        for future in as_completed(futures):
            row = future.result()
            key = task_key(row)
            with WRITE_LOCK:
                with args.output.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed[key] = row
                write_metadata(
                    args,
                    manifest_sha256,
                    runs,
                    expected_keys,
                    len(completed),
                )
            print(
                f"COMPLETE {row['scenario']} {row['condition']} "
                f"rep{row['replica']:02d} {row['agent_name']} "
                f"pre_json={row['pre']['response_json_parsed']} "
                f"post_json={row['post']['response_json_parsed']}",
                flush=True,
            )

    if set(completed) != expected_keys:
        raise RuntimeError(
            f"Incomplete output: {len(completed)}/{len(expected_keys)} rows"
        )
    print(f"COMPLETE all {len(completed)} agent rows: {args.output}", flush=True)


if __name__ == "__main__":
    main()

