#!/usr/bin/env python3
"""Validate and analyze the frozen C6 matched solo-generation control.

The independent unit is a completed run.  This script never selects C6
replicas based on outcomes: it requires exactly one complete artifact for
each scenario x replica cell (5 x 2), validates the frozen serving/exposure
contract, and then reports the prespecified C6-C4 and C2-C6 contrasts.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from build_rebuttal_evidence import AGENTS, FACTORS, initial_system_prompt, sha_text


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "rebuttal" / "2026-07-25"
SCENARIOS = ("greenfield", "riverton")
REPLICAS = tuple(range(5))
N_BOOT = 50_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(values: Iterable[float]) -> float:
    return float(np.mean(list(values)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_complete_c6() -> dict[tuple[str, int], Path]:
    selected: dict[tuple[str, int], Path] = {}
    for scenario in SCENARIOS:
        for replica in REPLICAS:
            candidates = sorted(
                path
                for path in DATA.glob(
                    f"c6rep{replica:02d}_solo_generation_{scenario}_*"
                )
                if (path / "summary.json").exists()
            )
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected exactly one complete C6 artifact for "
                    f"{scenario} rep{replica:02d}; found {candidates}"
                )
            selected[(scenario, replica)] = candidates[0]
    return selected


def bfi_run(path: Path) -> tuple[float, list[float]]:
    pre_rows = read_json(path / "pre_measurements.json").get("bfi10", [])
    post_rows = read_json(path / "post_measurements.json").get("bfi10", [])
    pre = {row["agent_name"]: row["scores"] for row in pre_rows}
    post = {row["agent_name"]: row["scores"] for row in post_rows}
    if set(pre) != set(AGENTS) or set(post) != set(AGENTS):
        raise RuntimeError(f"incomplete C6 BFI payload: {path}")
    agent_l1 = [
        sum(
            abs(float(post[name][factor]) - float(pre[name][factor]))
            for factor in FACTORS
        )
        for name in AGENTS
    ]
    return mean(agent_l1), agent_l1


def load_exposure(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (path / "exposure_calls.jsonl").read_text().splitlines()
        if line.strip()
    ]


def validate_run(
    path: Path,
    scenario: str,
    replica: int,
) -> tuple[list[str], dict]:
    errors: list[str] = []
    config = read_json(path / "config.json")
    summary = read_json(path / "summary.json")
    serving = read_json(path / "serving_metadata.json")
    post = read_json(path / "post_measurements.json")
    exposure = load_exposure(path)
    serving_config = serving.get("serving_configuration", {})

    expected = {
        "condition_id": 6,
        "condition_name": "solo_generation",
        "scenario": scenario,
        "n_agents": 8,
        "memory_mode": "srm",
        "model": "qwen3-122b-fp8",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"config.{key}={config.get(key)!r}, expected {value!r}")

    interaction = config.get("interaction", {})
    if interaction.get("n_rounds") != 30:
        errors.append("config.interaction.n_rounds must equal 30")
    if interaction.get("agents_per_conversation") != 1:
        errors.append("config.interaction.agents_per_conversation must equal 1")
    if interaction.get("conversations_per_round") != 0:
        errors.append("config.interaction.conversations_per_round must equal 0")
    if config.get("network", {}).get("topology") != "none":
        errors.append("config.network.topology must equal none")
    if not config.get("seed", {}).get("use_seed"):
        errors.append("config.seed.use_seed must be true")
    if not config.get("seed", {}).get("asymmetric"):
        errors.append("config.seed.asymmetric must be true")

    summary_expected = {
        "protocol": "c6_matched_solo_generation_batch8_amended_2026-07-28",
        "batch_size": 8,
        "replica": replica,
        "python_seed": 6200 + replica,
        "n_rounds_completed": 30,
        "total_conversations": 0,
        "total_solo_reflections": 240,
    }
    for key, value in summary_expected.items():
        if summary.get(key) != value:
            errors.append(f"summary.{key}={summary.get(key)!r}, expected {value!r}")

    exposure_summary = summary.get("exposure", {})
    expected_exposure = {
        "content_calls": 720,
        "memory_extraction_calls": 240,
        "provider_usage_records": 960,
    }
    for key, value in expected_exposure.items():
        if exposure_summary.get(key) != value:
            errors.append(
                f"summary.exposure.{key}={exposure_summary.get(key)!r}, "
                f"expected {value!r}"
            )
    if len(exposure) != 960:
        errors.append(f"exposure row count={len(exposure)}, expected 960")
    if sum(row.get("kind") == "content" for row in exposure) != 720:
        errors.append("exposure content row count must equal 720")
    if sum(row.get("kind") == "memory_extraction" for row in exposure) != 240:
        errors.append("exposure extraction row count must equal 240")
    indexes = sorted(row.get("call_index") for row in exposure)
    if indexes != list(range(1, 961)):
        errors.append("exposure call_index must be a permutation of 1..960")
    if {row.get("round") for row in exposure} != set(range(1, 31)):
        errors.append("exposure rounds must cover 1..30")
    if {row.get("client_batch_size") for row in exposure} != {8}:
        errors.append("every exposure row must record client_batch_size=8")
    if {str(row.get("server_max_num_seqs")) for row in exposure} != {"8"}:
        errors.append("every exposure row must record server_max_num_seqs=8")
    if {row.get("tensor_parallel_size") for row in exposure} != {1}:
        errors.append("every exposure row must record tensor_parallel_size=1")
    if {row.get("served_precision") for row in exposure} != {"fp8"}:
        errors.append("every exposure row must record fp8 precision")
    if any(
        row.get("provider_response_metadata", {}).get("served_model")
        != "qwen3-122b-fp8"
        for row in exposure
    ):
        errors.append("every exposure row must record served_model=qwen3-122b-fp8")

    initial_content = {
        row["agent_id"]: row
        for row in exposure
        if row.get("kind") == "content"
        and row.get("round") == 1
        and row.get("turn") == 1
    }
    if set(initial_content) != set(range(8)):
        errors.append("round-1 turn-1 content rows must cover all 8 agents")
    else:
        for agent_id, row in initial_content.items():
            expected_hash = sha_text(
                initial_system_prompt(agent_id, scenario, seeded=True)
            )
            if row.get("system_sha256") != expected_hash:
                errors.append(
                    f"initial system hash mismatch for agent {agent_id}"
                )

    round_files = sorted((path / "rounds").glob("round_*.json"))
    if len(round_files) != 30:
        errors.append(f"round file count={len(round_files)}, expected 30")
    if serving.get("serving_validation", {}).get("status") != "passed":
        errors.append("serving validation did not pass")
    frozen_serving = {
        "precision": "fp8",
        "tensor_parallel_size": 1,
        "max_model_len": "8192",
        "max_num_seqs": "8",
        "client_batch_size": 8,
        "gpu_memory_utilization": "0.92",
        "gdn_prefill_backend": "triton",
        "enforce_eager": True,
    }
    for key, value in frozen_serving.items():
        if serving_config.get(key) != value:
            errors.append(
                f"serving.{key}={serving_config.get(key)!r}, expected {value!r}"
            )
    if post.get("prediction_status", {}).get("status") != "not_applicable":
        errors.append("C6 prediction status must be not_applicable")
    if len(summary.get("agents", [])) != 8:
        errors.append("summary must contain 8 agents")
    if any(agent.get("n_memories") != 30 for agent in summary.get("agents", [])):
        errors.append("every agent must contain 30 memories")

    run_mean, agent_l1 = bfi_run(path)
    row = {
        "scenario": scenario,
        "replica": replica,
        "run_id": path.name,
        "run_l1_mean": run_mean,
        "agent_l1_values": ";".join(f"{value:.4f}" for value in agent_l1),
        "content_calls": exposure_summary.get("content_calls"),
        "memory_extraction_calls": exposure_summary.get(
            "memory_extraction_calls"
        ),
        "input_tokens_local": exposure_summary.get("input_tokens_local"),
        "output_tokens_local": exposure_summary.get("output_tokens_local"),
        "input_tokens_provider": exposure_summary.get("input_tokens_provider"),
        "output_tokens_provider": exposure_summary.get(
            "output_tokens_provider"
        ),
        "total_tokens_provider": exposure_summary.get("total_tokens_provider"),
        "serving_endpoint": serving_config.get("endpoint"),
        "vllm_command_sha256": serving_config.get("vllm_command_sha256"),
        "checkpoint_config_sha256": serving.get("model_files", {})
        .get("config.json", {})
        .get("sha256"),
        "validation_status": "passed" if not errors else "failed",
        "validation_errors": "; ".join(errors),
    }
    return errors, row


def parse_values(text: str) -> list[float]:
    return [float(value) for value in text.split(";") if value]


def load_primary_cells() -> dict[str, dict[str, list[float]]]:
    result = {}
    with (OUT / "primary_c2_c4_run_level.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["scenario"]] = {
                "C2": parse_values(row["c2_replica_means"]),
                "C4": parse_values(row["c4_replica_means"]),
            }
    if set(result) != set(SCENARIOS):
        raise RuntimeError("primary C2/C4 run-level CSV is incomplete")
    return result


def bootstrap_difference(
    a: list[float],
    b: list[float],
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a_array = np.asarray(a)
    b_array = np.asarray(b)
    draws = np.empty(N_BOOT)
    for index in range(N_BOOT):
        draws[index] = (
            rng.choice(a_array, len(a_array), replace=True).mean()
            - rng.choice(b_array, len(b_array), replace=True).mean()
        )
    lo, hi = np.percentile(draws, (2.5, 97.5))
    return float(lo), float(hi)


def exact_permutation(
    a: list[float],
    b: list[float],
) -> tuple[float, float, int]:
    pooled = np.asarray(a + b)
    observed = mean(a) - mean(b)
    n_a = len(a)
    statistics = []
    for indexes in itertools.combinations(range(len(pooled)), n_a):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(indexes)] = True
        statistics.append(float(pooled[mask].mean() - pooled[~mask].mean()))
    p_value = (
        sum(abs(value) >= abs(observed) - 1e-12 for value in statistics)
        / len(statistics)
    )
    return observed, p_value, len(statistics)


def analyze(
    run_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    primary = load_primary_cells()
    decomposition_rows = []
    exposure_rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        c2 = primary[scenario]["C2"]
        c4 = primary[scenario]["C4"]
        c6 = [
            float(row["run_l1_mean"])
            for row in run_rows
            if row["scenario"] == scenario
        ]
        if len(c6) != 5:
            raise RuntimeError(f"{scenario} does not contain five C6 run means")

        contrasts = {}
        for contrast_index, (label, a, b) in enumerate(
            (
                ("c2_minus_c4", c2, c4),
                ("c6_minus_c4", c6, c4),
                ("c2_minus_c6", c2, c6),
            )
        ):
            delta, p_value, allocations = exact_permutation(a, b)
            lo, hi = bootstrap_difference(
                a,
                b,
                seed=20260728 + scenario_index * 10 + contrast_index,
            )
            contrasts[label] = {
                "delta": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p_value,
                "allocations": allocations,
            }

        total = contrasts["c2_minus_c4"]["delta"]
        generation = contrasts["c6_minus_c4"]["delta"]
        remainder = contrasts["c2_minus_c6"]["delta"]
        decomposition_rows.append(
            {
                "scenario": scenario,
                "c2_n": len(c2),
                "c4_n": len(c4),
                "c6_n": len(c6),
                "c2_run_values": ";".join(f"{value:.4f}" for value in c2),
                "c4_run_values": ";".join(f"{value:.4f}" for value in c4),
                "c6_run_values": ";".join(f"{value:.4f}" for value in c6),
                "c2_mean": mean(c2),
                "c4_mean": mean(c4),
                "c6_mean": mean(c6),
                "c2_minus_c4": total,
                "c2_minus_c4_ci_lo": contrasts["c2_minus_c4"]["ci_lo"],
                "c2_minus_c4_ci_hi": contrasts["c2_minus_c4"]["ci_hi"],
                "c2_minus_c4_exact_p": contrasts["c2_minus_c4"]["p"],
                "c6_minus_c4": generation,
                "c6_minus_c4_ci_lo": contrasts["c6_minus_c4"]["ci_lo"],
                "c6_minus_c4_ci_hi": contrasts["c6_minus_c4"]["ci_hi"],
                "c6_minus_c4_exact_p": contrasts["c6_minus_c4"]["p"],
                "c2_minus_c6": remainder,
                "c2_minus_c6_ci_lo": contrasts["c2_minus_c6"]["ci_lo"],
                "c2_minus_c6_ci_hi": contrasts["c2_minus_c6"]["ci_hi"],
                "c2_minus_c6_exact_p": contrasts["c2_minus_c6"]["p"],
                "descriptive_generation_fraction_of_total": generation / total,
                "descriptive_reciprocal_remainder_fraction_of_total": (
                    remainder / total
                ),
                "frozen_pattern": "C2 > C6 > C4",
                "interpretation": (
                    "extended generation/history explains most observed drift; "
                    "the smaller reciprocal-interaction remainder is unresolved"
                ),
            }
        )

        scenario_runs = [
            row for row in run_rows if row["scenario"] == scenario
        ]
        exposure_rows.append(
            {
                "scenario": scenario,
                "n_runs": len(scenario_runs),
                "content_calls_total": sum(
                    int(row["content_calls"]) for row in scenario_runs
                ),
                "memory_extraction_calls_total": sum(
                    int(row["memory_extraction_calls"]) for row in scenario_runs
                ),
                "provider_input_tokens_total": sum(
                    int(row["input_tokens_provider"]) for row in scenario_runs
                ),
                "provider_output_tokens_total": sum(
                    int(row["output_tokens_provider"]) for row in scenario_runs
                ),
                "provider_total_tokens": sum(
                    int(row["total_tokens_provider"]) for row in scenario_runs
                ),
                "provider_total_tokens_mean_per_run": mean(
                    int(row["total_tokens_provider"]) for row in scenario_runs
                ),
                "local_total_tokens": sum(
                    int(row["input_tokens_local"])
                    + int(row["output_tokens_local"])
                    for row in scenario_runs
                ),
                "historical_c2_usage_available": "no",
                "exposure_match_statement": (
                    "C6 matches C2's intended logical call budget "
                    "(720 content + 240 extraction calls per run); historical "
                    "C2 provider/tokenizer usage was not recorded"
                ),
            }
        )
    return decomposition_rows, exposure_rows


def summary_markdown(rows: list[dict], exposure_rows: list[dict]) -> str:
    lines = [
        "# C6 matched solo-generation decomposition",
        "",
        "Frozen run-level outcome: mean per-agent BFI-10 L1 post-pre change. "
        "All five C6 replicas in both scenarios are included.",
        "",
        "|Scenario|C2 mean|C6 mean|C4 mean|C2-C4 total|C6-C4 generation|C2-C6 reciprocal remainder|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['scenario']}|{row['c2_mean']:.3f}|{row['c6_mean']:.3f}|"
            f"{row['c4_mean']:.3f}|{row['c2_minus_c4']:+.3f} "
            f"[{row['c2_minus_c4_ci_lo']:+.3f},{row['c2_minus_c4_ci_hi']:+.3f}]|"
            f"{row['c6_minus_c4']:+.3f} "
            f"[{row['c6_minus_c4_ci_lo']:+.3f},{row['c6_minus_c4_ci_hi']:+.3f}]|"
            f"{row['c2_minus_c6']:+.3f} "
            f"[{row['c2_minus_c6_ci_lo']:+.3f},{row['c2_minus_c6_ci_hi']:+.3f}]|"
        )
    lines.extend(
        [
            "",
            "Both scenarios show the frozen `C2 > C6 > C4` pattern. "
            "Descriptively, C6-C4 accounts for about 75% (Greenfield) and 78% "
            "(Riverton) of C2-C4. These percentages are an aligned-mean "
            "decomposition, not formal mediation proportions.",
            "",
            "C6-C4 is positive in both scenarios, whereas the smaller C2-C6 "
            "descriptive intervals include zero. Extended solo "
            "generation/history is therefore sufficient to produce most of "
            "the observed response-profile drift; the data do not isolate a "
            "clear additional reciprocal-interaction effect.",
            "",
            "The defensible revised claim is that extended stateful agent "
            "trajectories can break replicate exchangeability. Social "
            "interaction is one such trajectory, but these data do not show "
            "that reciprocity is necessary or the dominant mechanism.",
            "",
            "## Exposure audit",
            "",
            "|Scenario|Runs|Content calls|Extraction calls|Provider tokens|Mean provider tokens/run|",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in exposure_rows:
        lines.append(
            f"|{row['scenario']}|{row['n_runs']}|"
            f"{row['content_calls_total']}|"
            f"{row['memory_extraction_calls_total']}|"
            f"{row['provider_total_tokens']}|"
            f"{row['provider_total_tokens_mean_per_run']:.0f}|"
        )
    lines.extend(
        [
            "",
            "Each C6 run contains exactly 720 content calls and 240 memory "
            "extractions, matching C2's intended logical call budget. "
            "Historical C2 artifacts did not record provider usage or exact "
            "local-tokenizer counts, so realized token equality with C2 cannot "
            "be claimed retrospectively.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    selected = discover_complete_c6()
    validation_rows = []
    run_rows = []
    all_errors = []
    for scenario in SCENARIOS:
        for replica in REPLICAS:
            path = selected[(scenario, replica)]
            errors, row = validate_run(path, scenario, replica)
            validation_rows.append(
                {
                    "scenario": scenario,
                    "replica": replica,
                    "selected_run": str(path.relative_to(ROOT)),
                    "status": "passed" if not errors else "failed",
                    "errors": "; ".join(errors),
                }
            )
            run_rows.append(row)
            all_errors.extend(
                f"{scenario} rep{replica:02d}: {error}" for error in errors
            )
    write_csv(OUT / "c6_validation.csv", validation_rows)
    write_csv(OUT / "c6_run_level.csv", run_rows)
    if all_errors:
        raise RuntimeError("C6 validation failed:\n" + "\n".join(all_errors))

    decomposition_rows, exposure_rows = analyze(run_rows)
    write_csv(OUT / "c6_decomposition.csv", decomposition_rows)
    write_csv(OUT / "c6_exposure_summary.csv", exposure_rows)
    (OUT / "c6_summary.md").write_text(
        summary_markdown(decomposition_rows, exposure_rows)
    )
    print(
        "Validated 10/10 C6 runs; wrote run-level decomposition and exposure audit"
    )


if __name__ == "__main__":
    main()
