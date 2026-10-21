"""Run-level analysis for the frozen objective specificity probe."""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "rebuttal" / "2026-07-25"
EXPECTED_RUNS = {"C2": 8, "C4": 5}
DOMAINS = ("combined", "math", "logic")
BOOTSTRAP_SEED = 20260728
EQUIVALENCE_MARGIN = 0.05


def load_complete_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (row["condition"], row["replica"], row["agent_id"])
        if key in seen:
            raise RuntimeError(f"{path}: duplicate row {key}")
        seen.add(key)
    expected = sum(EXPECTED_RUNS.values()) * 8
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} rows, found {len(rows)}")
    manifest_hashes = {row["manifest_sha256"] for row in rows}
    if len(manifest_hashes) != 1:
        raise RuntimeError(f"{path}: multiple manifest hashes")
    for condition, n_runs in EXPECTED_RUNS.items():
        replicas = {
            row["replica"] for row in rows if row["condition"] == condition
        }
        if replicas != set(range(n_runs)):
            raise RuntimeError(
                f"{path}: {condition} replicas {sorted(replicas)} are incomplete"
            )
        for replica in replicas:
            agents = {
                row["agent_id"]
                for row in rows
                if row["condition"] == condition
                and row["replica"] == replica
            }
            if agents != set(range(8)):
                raise RuntimeError(
                    f"{path}: {condition} rep{replica:02d} agents incomplete"
                )
    return rows


def accuracy(probe: dict, domain: str) -> float:
    scores = probe["item_scores"]
    if domain != "combined":
        scores = [row for row in scores if row["domain"] == domain]
    if not scores:
        raise RuntimeError(f"No items for domain {domain}")
    return sum(row["correct"] for row in scores) / len(scores)


def run_table(rows: list[dict], domain: str) -> list[dict]:
    result = []
    for condition, n_runs in EXPECTED_RUNS.items():
        for replica in range(n_runs):
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["replica"] == replica
            ]
            pre = [accuracy(row["pre"], domain) for row in selected]
            post = [accuracy(row["post"], domain) for row in selected]
            deltas = [b - a for a, b in zip(pre, post)]
            result.append(
                {
                    "condition": condition,
                    "replica": replica,
                    "pre_accuracy": statistics.mean(pre),
                    "post_accuracy": statistics.mean(post),
                    "delta": statistics.mean(deltas),
                }
            )
    return result


def exact_permutation_p(a: list[float], b: list[float]) -> float:
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = a + b
    indices = range(len(pooled))
    hits = total = 0
    for chosen in combinations(indices, len(a)):
        chosen_set = set(chosen)
        aa = [pooled[index] for index in indices if index in chosen_set]
        bb = [pooled[index] for index in indices if index not in chosen_set]
        difference = abs(statistics.mean(aa) - statistics.mean(bb))
        hits += difference >= observed - 1e-12
        total += 1
    return hits / total


def bootstrap_interval(
    a: list[float],
    b: list[float],
    quantiles: tuple[float, float],
    seed_offset: int,
    draws: int = 50_000,
) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    aa = np.asarray(a)
    bb = np.asarray(b)
    values = np.empty(draws)
    for index in range(draws):
        values[index] = (
            rng.choice(aa, size=len(aa), replace=True).mean()
            - rng.choice(bb, size=len(bb), replace=True).mean()
        )
    return [float(value) for value in np.quantile(values, quantiles)]


def summarize_domain(
    rows: list[dict],
    domain: str,
    seed_offset: int,
) -> dict:
    table = run_table(rows, domain)
    c2_rows = [row for row in table if row["condition"] == "C2"]
    c4_rows = [row for row in table if row["condition"] == "C4"]
    c2 = [row["delta"] for row in c2_rows]
    c4 = [row["delta"] for row in c4_rows]
    contrast = statistics.mean(c2) - statistics.mean(c4)
    ci95 = bootstrap_interval(c2, c4, (0.025, 0.975), seed_offset)
    ci90 = bootstrap_interval(c2, c4, (0.05, 0.95), seed_offset + 100)
    return {
        "run_values": table,
        "c2_pre_mean": statistics.mean(
            row["pre_accuracy"] for row in c2_rows
        ),
        "c2_post_mean": statistics.mean(
            row["post_accuracy"] for row in c2_rows
        ),
        "c4_pre_mean": statistics.mean(
            row["pre_accuracy"] for row in c4_rows
        ),
        "c4_post_mean": statistics.mean(
            row["post_accuracy"] for row in c4_rows
        ),
        "c2_delta_mean": statistics.mean(c2),
        "c4_delta_mean": statistics.mean(c4),
        "c2_minus_c4_delta": contrast,
        "descriptive_bootstrap_ci95": ci95,
        "descriptive_bootstrap_ci90": ci90,
        "exact_two_sided_permutation_p": exact_permutation_p(c2, c4),
        "equivalence_margin": EQUIVALENCE_MARGIN if domain == "combined" else None,
        "combined_practical_stability": (
            ci90[0] > -EQUIVALENCE_MARGIN
            and ci90[1] < EQUIVALENCE_MARGIN
            if domain == "combined"
            else None
        ),
    }


def parse_failure_summary(rows: list[dict]) -> dict:
    responses = [
        row[state]
        for row in rows
        for state in ("pre", "post")
    ]
    item_scores = [
        score
        for response in responses
        for score in response["item_scores"]
    ]
    return {
        "n_responses": len(responses),
        "response_json_parse_failures": sum(
            not response["response_json_parsed"] for response in responses
        ),
        "n_expected_item_answers": len(item_scores),
        "missing_item_answers": sum(
            not score["present"] for score in item_scores
        ),
    }


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Objective mathematics and logic specificity probe",
        "",
        "Frozen-state analysis of all primary C2 and C4 artifacts. The saved "
        "run is the independent unit. Values are raw accuracy proportions.",
        "",
        "|Scenario|Domain|C2 pre|C2 post|C4 pre|C4 post|C2-C4 change|95% run-bootstrap CI|Exact p|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("greenfield", "riverton"):
        for domain in DOMAINS:
            result = summary["scenarios"][scenario]["domains"][domain]
            ci = result["descriptive_bootstrap_ci95"]
            lines.append(
                f"|{scenario}|{domain}|{result['c2_pre_mean']:.3f}|"
                f"{result['c2_post_mean']:.3f}|{result['c4_pre_mean']:.3f}|"
                f"{result['c4_post_mean']:.3f}|"
                f"{result['c2_minus_c4_delta']:+.3f}|"
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}]|"
                f"{result['exact_two_sided_permutation_p']:.4f}|"
            )
    lines.extend(["", "## Frozen practical-stability check", ""])
    for scenario in ("greenfield", "riverton"):
        result = summary["scenarios"][scenario]["domains"]["combined"]
        ci = result["descriptive_bootstrap_ci90"]
        status = (
            "supported"
            if result["combined_practical_stability"]
            else "inconclusive"
        )
        lines.append(
            f"- {scenario}: 90% interval [{ci[0]:+.3f}, {ci[1]:+.3f}] "
            f"against the frozen ±0.050 bound; practical stability {status}."
        )
    lines.extend(["", "## Parsing audit", ""])
    for scenario in ("greenfield", "riverton"):
        audit = summary["scenarios"][scenario]["parse_failures"]
        lines.append(
            f"- {scenario}: {audit['response_json_parse_failures']}/"
            f"{audit['n_responses']} response JSON failures; "
            f"{audit['missing_item_answers']}/"
            f"{audit['n_expected_item_answers']} missing item answers."
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--greenfield",
        type=Path,
        default=EVIDENCE / "objective_specificity_greenfield.jsonl",
    )
    parser.add_argument(
        "--riverton",
        type=Path,
        default=EVIDENCE / "objective_specificity_riverton.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=EVIDENCE / "objective_specificity_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=EVIDENCE / "objective_specificity_summary.md",
    )
    args = parser.parse_args()

    summary = {
        "protocol": "objective_specificity_frozen_plan.md",
        "independent_unit": "run",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "scenarios": {},
    }
    manifest_hashes = set()
    for scenario, path in (
        ("greenfield", args.greenfield),
        ("riverton", args.riverton),
    ):
        rows = load_complete_rows(path)
        manifest_hashes.update(row["manifest_sha256"] for row in rows)
        summary["scenarios"][scenario] = {
            "n_c2_runs": 8,
            "n_c4_runs": 5,
            "agents_per_run": 8,
            "domains": {
                domain: summarize_domain(rows, domain, index)
                for index, domain in enumerate(DOMAINS)
            },
            "parse_failures": parse_failure_summary(rows),
        }
    if len(manifest_hashes) != 1:
        raise RuntimeError("Greenfield and Riverton used different manifests")
    summary["manifest_sha256"] = next(iter(manifest_hashes))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

