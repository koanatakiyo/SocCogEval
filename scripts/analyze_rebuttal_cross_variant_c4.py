#!/usr/bin/env python3
"""Audit and summarize the frozen cross-variant C2-C4 rebuttal panel.

The independent unit is one completed run.  The script records the exact
artifact selected for every model/scenario/replica cell, reports raw run
values, and keeps model-revision limitations visible in the output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "rebuttal" / "2026-07-25"
DATA_ROOTS = (
    ROOT / "data",
    DEFAULT_OUT / "gpus_result",
)
ALIASES = ("dpsk", "gem", "gpt", "l4api", "qvl")
SCENARIOS = ("greenfield", "riverton")
CONDITIONS = ("segreg_asym_seed", "seed_only_no_interact")
FACTORS = (
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
)
DISPLAY_NAMES = {
    "dpsk": "DeepSeek",
    "gem": "Gemini-2.5-Flash",
    "gpt": "GPT-4.1-mini",
    "l4api": "Llama-4-Scout",
    "qvl": "Qwen3-VL-32B",
}
RUN_RE = re.compile(
    r"^rep(?P<rep>0[0-2])_(?P<alias>dpsk|gem|gpt|l4api|qvl)_"
    r"(?P<condition>segreg_asym_seed|seed_only_no_interact)_"
    r"(?P<scenario>greenfield|riverton)_(?P<timestamp>\d{8}_\d{6})$"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_complete_runs() -> dict[tuple[str, str, str, str], list[Path]]:
    found: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    required = (
        "config.json",
        "pre_measurements.json",
        "post_measurements.json",
        "summary.json",
    )
    for root in DATA_ROOTS:
        if not root.exists():
            continue
        for path in root.iterdir():
            match = RUN_RE.match(path.name)
            if not match or not path.is_dir():
                continue
            if not all((path / filename).exists() for filename in required):
                continue
            fields = match.groupdict()
            key = (
                fields["alias"],
                fields["scenario"],
                fields["rep"],
                fields["condition"],
            )
            found[key].append(path)
    return found


def select_runs(
    found: dict[tuple[str, str, str, str], list[Path]]
) -> tuple[dict[tuple[str, str, str, str], Path], list[dict]]:
    selected = {}
    rows = []
    for alias in ALIASES:
        for scenario in SCENARIOS:
            for replica in ("00", "01", "02"):
                for condition in CONDITIONS:
                    key = (alias, scenario, replica, condition)
                    candidates = sorted(found.get(key, []), key=lambda path: path.name)
                    if not candidates:
                        raise RuntimeError(f"missing complete artifact: {key}")
                    # Retries are selected by the latest timestamp encoded in the
                    # directory name; every candidate remains listed for audit.
                    path = candidates[-1]
                    selected[key] = path
                    config = read_json(path / "config.json")
                    rows.append(
                        {
                            "alias": alias,
                            "scenario": scenario,
                            "replica": replica,
                            "condition": "C2" if condition == CONDITIONS[0] else "C4",
                            "selected_run": str(path.relative_to(ROOT)),
                            "configured_model": config.get("model", ""),
                            "candidate_count": len(candidates),
                            "all_complete_candidates": ";".join(
                                str(candidate.relative_to(ROOT))
                                for candidate in candidates
                            ),
                        }
                    )
    return selected, rows


def run_l1_mean(path: Path) -> float:
    pre_rows = read_json(path / "pre_measurements.json").get("bfi10", [])
    post_rows = read_json(path / "post_measurements.json").get("bfi10", [])
    pre = {row["agent_name"]: row["scores"] for row in pre_rows}
    post = {row["agent_name"]: row["scores"] for row in post_rows}
    if len(pre) != 8 or set(pre) != set(post):
        raise RuntimeError(f"incomplete BFI payload: {path}")
    agent_values = []
    for agent in sorted(pre):
        agent_values.append(
            sum(abs(float(post[agent][factor]) - float(pre[agent][factor]))
                for factor in FACTORS)
        )
    return float(np.mean(agent_values))


def bootstrap_difference(
    c2: list[float], c4: list[float], rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    c2_array = np.asarray(c2)
    c4_array = np.asarray(c4)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        draws[index] = (
            np.mean(rng.choice(c2_array, len(c2_array), replace=True))
            - np.mean(rng.choice(c4_array, len(c4_array), replace=True))
        )
    lo, hi = np.percentile(draws, (2.5, 97.5))
    return float(lo), float(hi)


def model_match_label(c2_models: set[str], c4_models: set[str]) -> str:
    if c2_models == c4_models:
        return "configured_model_identifier_matches"
    return "configured_model_identifier_mismatch"


def analyze(
    selected: dict[tuple[str, str, str, str], Path], n_boot: int
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(20260725)
    cell_rows = []
    aggregate_rows = []
    for alias in ALIASES:
        scenario_deltas = []
        match_labels = []
        for scenario in SCENARIOS:
            values = {}
            models = {}
            for condition in CONDITIONS:
                paths = [
                    selected[(alias, scenario, replica, condition)]
                    for replica in ("00", "01", "02")
                ]
                values[condition] = [run_l1_mean(path) for path in paths]
                models[condition] = {
                    read_json(path / "config.json").get("model", "") for path in paths
                }
            c2 = values[CONDITIONS[0]]
            c4 = values[CONDITIONS[1]]
            delta = float(np.mean(c2) - np.mean(c4))
            lo, hi = bootstrap_difference(c2, c4, rng, n_boot)
            match = model_match_label(models[CONDITIONS[0]], models[CONDITIONS[1]])
            match_labels.append(match)
            scenario_deltas.append(delta)
            cell_rows.append(
                {
                    "variant": DISPLAY_NAMES[alias],
                    "alias": alias,
                    "scenario": scenario,
                    "n_c2_runs": len(c2),
                    "n_c4_runs": len(c4),
                    "c2_run_l1_means": ";".join(f"{value:.4f}" for value in c2),
                    "c4_run_l1_means": ";".join(f"{value:.4f}" for value in c4),
                    "c2_mean": float(np.mean(c2)),
                    "c4_mean": float(np.mean(c4)),
                    "delta_c2_minus_c4": delta,
                    "descriptive_boot_ci_lo": lo,
                    "descriptive_boot_ci_hi": hi,
                    "c2_configured_models": ";".join(sorted(models[CONDITIONS[0]])),
                    "c4_configured_models": ";".join(sorted(models[CONDITIONS[1]])),
                    "configured_model_match": match,
                    "revision_note": "exact historical provider/model revision not recorded",
                }
            )
        aggregate_rows.append(
            {
                "variant": DISPLAY_NAMES[alias],
                "alias": alias,
                "greenfield_delta": scenario_deltas[0],
                "riverton_delta": scenario_deltas[1],
                "scenario_averaged_delta": float(np.mean(scenario_deltas)),
                "positive_direction": "yes" if np.mean(scenario_deltas) > 0 else "no",
                "configured_model_match_both_scenarios": (
                    "yes"
                    if all(label == "configured_model_identifier_matches"
                           for label in match_labels)
                    else "no"
                ),
            }
        )
    return cell_rows, aggregate_rows


def markdown_summary(cell_rows: list[dict], aggregate_rows: list[dict]) -> str:
    positive = sum(row["positive_direction"] == "yes" for row in aggregate_rows)
    lines = [
        "# Cross-variant C2-C4 run-level sensitivity analysis",
        "",
        "Frozen outcome: mean per-agent BFI-10 L1 post-pre change, summarized "
        "first within each run. Each cell is 3 C2 versus 3 C4 runs.",
        "",
        "|Variant|Greenfield Δ|Riverton Δ|Scenario-averaged Δ|Configured model ID matched?|",
        "|---|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"|{row['variant']}|{row['greenfield_delta']:+.3f}|"
            f"{row['riverton_delta']:+.3f}|"
            f"{row['scenario_averaged_delta']:+.3f}|"
            f"{row['configured_model_match_both_scenarios']}|"
        )
    lines.extend(
        [
            "",
            f"Direction: {positive}/{len(aggregate_rows)} prespecified variants "
            "have positive scenario-averaged C2-C4 differences. If the complete "
            "five-variant directional panel is treated as the frozen sign test, "
            f"the one-sided exact p-value is {2 ** -len(aggregate_rows):.5f}.",
            "",
            "Interpretation limit: this is a directional sensitivity panel, not "
            "five strict token/revision-matched causal replications. Exact "
            "historical provider revisions were not recorded. In addition, the "
            "historical DeepSeek C2 artifacts use the configured identifier "
            "`deepseek-chat`, whereas the repaired C4 artifacts use the explicit "
            "identifier `deepseek-v4-flash`. The configured strings therefore "
            "differ, but DeepSeek's official changelog maps the rolling "
            "`deepseek-chat` alias to V4-Flash at the historical C2 run dates; "
            "both sets also use DeepSeek's official API endpoint.",
            "",
            "Descriptive bootstrap intervals and all replica-level values are in "
            "`cross_variant_c4_results.csv`; exact selected artifacts are in "
            "`cross_variant_c4_selected_runs.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=50_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    found = discover_complete_runs()
    selected, selected_rows = select_runs(found)
    cell_rows, aggregate_rows = analyze(selected, args.n_boot)
    write_csv(args.out / "cross_variant_c4_selected_runs.csv", selected_rows)
    write_csv(args.out / "cross_variant_c4_results.csv", cell_rows)
    write_csv(args.out / "cross_variant_c4_variant_summary.csv", aggregate_rows)
    (args.out / "cross_variant_c4_summary.md").write_text(
        markdown_summary(cell_rows, aggregate_rows)
    )
    print(f"Wrote {len(cell_rows)} scenario cells and {len(aggregate_rows)} variants")


if __name__ == "__main__":
    main()
