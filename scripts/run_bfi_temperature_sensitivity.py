"""Frozen T=0 BFI-10 sensitivity on saved primary C2/C4 agent states.

This script re-probes every selected primary C2/C4 artifact at temperature
zero.  It reconstructs the pre state from the scenario and assigned brief,
and the post state from the serialized SRM entries.  The independent unit
remains the run; no new interaction trajectories are generated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from cogclaw.agent import AGENT_NAMES, Agent
from cogclaw import probes
from scenarios import get_scenario


DATA = ROOT / "data"


def selected_runs(scenario: str) -> list[tuple[str, int, Path]]:
    specs = (("C2", "segreg_asym_seed", 8), ("C4", "seed_only_no_interact", 5))
    selected: list[tuple[str, int, Path]] = []
    for condition, condition_name, n_reps in specs:
        for replica in range(n_reps):
            pattern = f"rep{replica:02d}_{condition_name}_{scenario}_*"
            matches = sorted(
                p for p in DATA.glob(pattern) if (p / "summary.json").exists()
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
        model="qwen3-122b-fp8",
        seed_info=seed_info,
    )
    agent.group_id = group_id
    if post:
        for entry in payload.get("memories", []):
            agent.memory.add(entry)
    return agent


def probe_one(
    condition: str,
    replica: int,
    run_dir: Path,
    scenario_name: str,
    agent_id: int,
) -> dict:
    pre_agent = load_agent(run_dir, scenario_name, agent_id, post=False)
    post_agent = load_agent(run_dir, scenario_name, agent_id, post=True)
    pre = probes.bfi10_test(pre_agent)
    post = probes.bfi10_test(post_agent)
    dims = sorted(pre["scores"])
    l1 = sum(abs(post["scores"][d] - pre["scores"][d]) for d in dims)
    return {
        "condition": condition,
        "replica": replica,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "scenario": scenario_name,
        "agent_id": agent_id,
        "agent_name": AGENT_NAMES[agent_id],
        "pre": pre,
        "post": post,
        "bfi_l1": l1,
    }


def exact_permutation_p(a: list[float], b: list[float]) -> float:
    from itertools import combinations

    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = a + b
    n_a = len(a)
    hits = total = 0
    indices = range(len(pooled))
    for chosen in combinations(indices, n_a):
        chosen_set = set(chosen)
        aa = [pooled[i] for i in indices if i in chosen_set]
        bb = [pooled[i] for i in indices if i not in chosen_set]
        hits += abs(statistics.mean(aa) - statistics.mean(bb)) >= observed - 1e-12
        total += 1
    return hits / total


def bootstrap_ci(a: list[float], b: list[float], draws: int = 50_000) -> list[float]:
    rng = np.random.default_rng(20260728)
    values = np.empty(draws)
    aa = np.asarray(a)
    bb = np.asarray(b)
    for i in range(draws):
        values[i] = (
            rng.choice(aa, size=len(aa), replace=True).mean()
            - rng.choice(bb, size=len(bb), replace=True).mean()
        )
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("greenfield", "riverton"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probes.CONFORMITY_TEMPERATURE = 0.0
    runs = selected_runs(args.scenario)
    tasks = [
        (condition, replica, run_dir, args.scenario, agent_id)
        for condition, replica, run_dir in runs
        for agent_id in range(8)
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(probe_one, *task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['scenario']} {row['condition']} rep{row['replica']:02d} "
                f"{row['agent_name']} L1={row['bfi_l1']:.3f}",
                flush=True,
            )

    run_means: dict[str, list[dict]] = {"C2": [], "C4": []}
    for condition in run_means:
        for replica in sorted({r["replica"] for r in rows if r["condition"] == condition}):
            vals = [
                r["bfi_l1"]
                for r in rows
                if r["condition"] == condition and r["replica"] == replica
            ]
            run_means[condition].append(
                {"replica": replica, "mean_bfi_l1": statistics.mean(vals)}
            )
    c2 = [r["mean_bfi_l1"] for r in run_means["C2"]]
    c4 = [r["mean_bfi_l1"] for r in run_means["C4"]]
    summary = {
        "frozen_design": {
            "scenario": args.scenario,
            "bfi_temperature": 0.0,
            "conditions": {"C2": 8, "C4": 5},
            "agents_per_run": 8,
            "independent_unit": "run",
            "selection": "exactly one complete primary rep00..07 C2 and rep00..04 C4",
            "endpoint": os.environ.get("QWEN_BASE_URL", ""),
        },
        "run_means": run_means,
        "contrast": {
            "c2_mean": statistics.mean(c2),
            "c4_mean": statistics.mean(c4),
            "c2_minus_c4": statistics.mean(c2) - statistics.mean(c4),
            "descriptive_bootstrap_ci": bootstrap_ci(c2, c4),
            "exact_two_sided_permutation_p": exact_permutation_p(c2, c4),
        },
        "rows": sorted(
            rows, key=lambda r: (r["condition"], r["replica"], r["agent_id"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["contrast"], indent=2))


if __name__ == "__main__":
    main()
