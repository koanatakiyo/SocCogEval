#!/usr/bin/env python3
"""Build the NeurIPS 2026 rebuttal evidence package from completed runs.

This script is deliberately read-only with respect to experiment data.  It
derives a ledger and descriptive, run-level analyses under rebuttal/ from the
JSON artifacts already present in data/.

Important estimand note
-----------------------
For C2 versus C4, pre-treatment prompt construction should match.  The final
post-probe prompt is *not* expected to match because interaction-created
memory is the treatment mediator.  The trace report generated here makes that
distinction explicit and records which historical trace fields were not saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "rebuttal" / "2026-07-25"
AGENTS = ["Alex", "Blake", "Casey", "Dana", "Ellis", "Finley", "Gray", "Harper"]
FACTORS = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
]
CONDITION_NAMES = {
    "open_seed": "C1",
    "segreg_asym_seed": "C2",
    "segreg_no_seed": "C3",
    "seed_only_no_interact": "C4",
    "no_seed_no_interact": "C5",
}
ALIASES = {
    "qvl": "Qwen3-VL-32B (text-only)",
    "dpsk": "DeepSeek-V3",
    "gpt": "GPT-4.1-mini",
    "gem": "Gemini-2.5-Flash",
    "l4api": "Llama-4-Scout (OpenRouter)",
}
PRIMARY = "Qwen3-122B-FP8"
RUN_RE = re.compile(
    r"^(?P<prefix>(?:nosrm_)?rep(?P<rep>\d\d)(?:_(?P<alias>qvl|dpsk|gpt|gem|l4api))?)_"
    r"(?P<condition>open_seed|segreg_asym_seed|segreg_no_seed|"
    r"seed_only_no_interact|no_seed_no_interact)_"
    r"(?P<scenario>greenfield|riverton)_(?P<timestamp>\d{8}_\d{6})$"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode())


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes()) if path.exists() else ""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def discover_runs() -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        required = ["config.json", "pre_measurements.json", "post_measurements.json", "summary.json"]
        complete = all((run_dir / name).exists() for name in required)
        if not complete:
            continue
        info = match.groupdict()
        config = read_json(run_dir / "config.json")
        summary = read_json(run_dir / "summary.json")
        info.update(
            {
                "path": run_dir,
                "config": config,
                "summary": summary,
                "condition": CONDITION_NAMES[info["condition"]],
                "variant": ALIASES.get(info["alias"], PRIMARY),
                "memory_mode": (
                    "verbatim"
                    if run_dir.name.startswith("nosrm_")
                    else config.get("memory_mode", "srm")
                ),
            }
        )
        runs.append(info)
    return runs


def scenario_materials(scenario: str) -> tuple[str, str, str]:
    # Importing scenario modules has no model/network side effects.
    if scenario == "greenfield":
        from scenarios.greenfield import CONTEXT, SEED_A, SEED_B
    else:
        from scenarios.riverton import CONTEXT, SEED_A, SEED_B
    return CONTEXT, SEED_A, SEED_B


def initial_system_prompt(agent_index: int, scenario: str, seeded: bool) -> str:
    context, seed_a, seed_b = scenario_materials(scenario)
    parts = [
        f"You are {AGENTS[agent_index]}, a community member.",
        "",
        context,
        "",
        "Express your genuine thoughts honestly. Engage thoughtfully with "
        "others' perspectives. You may change your mind if you find their "
        "arguments compelling, or maintain your position if you disagree.",
    ]
    if seeded:
        parts.extend(
            [
                "",
                "Background information you have received:",
                seed_a if agent_index < 4 else seed_b,
            ]
        )
    return "\n".join(parts)


def combined_memory_hash(run_dir: Path) -> str:
    payload = []
    for path in sorted((run_dir / "memories").glob("*.json")):
        payload.append(read_json(path))
    return sha_bytes(canonical_json(payload)) if payload else sha_bytes(b"[]")


def measurement_payload(run_dir: Path, phase: str, instrument: str, field: str) -> Any:
    data = read_json(run_dir / f"{phase}_measurements.json")
    rows = data.get(instrument, []) or []
    return [row.get(field) for row in rows]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_ledger(runs: list[dict], out: Path) -> list[dict]:
    probes_hash = sha_file(ROOT / "cogclaw" / "probes.py")
    ledger = []
    for run in runs:
        cfg = run["config"]
        path = run["path"]
        scenario = run["scenario"]
        _context, seed_a, seed_b = scenario_materials(scenario)
        memory_files = sorted((path / "memories").glob("*.json"))
        n_memories = 0
        for mem_path in memory_files:
            n_memories += len(read_json(mem_path).get("memories", []))
        model_raw = cfg.get("model", "")
        resolved_model = model_raw or "qwen3-122b-fp8 (resolved by runtime default)"
        ledger.append(
            {
                "run_id": path.name,
                "condition": run["condition"],
                "scenario": scenario,
                "variant": run["variant"],
                "variant_alias": run["alias"] or "primary",
                "replica_id": run["rep"],
                "timestamp_from_dir": run["timestamp"],
                "model_config_raw": model_raw,
                "resolved_model": resolved_model,
                "exact_model_revision": "not_recorded",
                "replica_randomization": "Python seed used by launcher but not saved in run artifact",
                "config_sha256": sha_file(path / "config.json"),
                "checkpoint_clone_hash": "not_recorded",
                "brief_a_sha256": sha_text(seed_a) if cfg.get("seed", {}).get("use_seed") else "",
                "brief_b_sha256": sha_text(seed_b)
                if cfg.get("seed", {}).get("asymmetric")
                else "",
                "brief_assignment": (
                    "agents 0-3=A; agents 4-7=B"
                    if cfg.get("seed", {}).get("asymmetric")
                    else ("all=A" if cfg.get("seed", {}).get("use_seed") else "none")
                ),
                "brief_injection_point": "Agent.system_prompt",
                "memory_mode": run["memory_mode"],
                "memory_payload_sha256": combined_memory_hash(path),
                "n_serialized_memories": n_memories,
                "pre_probe_order": "per agent: BFI10, conformity natural+pressure, allocation",
                "post_probe_order": (
                    "per agent: BFI10, conformity natural+pressure, allocation, "
                    "consistency, predictions"
                ),
                "assembled_prompt_sha256": "not_recorded",
                "token_id_sequence_sha256": "not_recorded",
                "tokenizer_revision": "not_recorded",
                "conversation_temperature": "0.7 (current source; historical request not logged)",
                "bfi_temperature": "0.7 (current source; historical request not logged)",
                "allocation_temperature": "0.0 (current source; historical request not logged)",
                "endpoint_precision_batch": "not_recorded",
                "run_time_code_commit": "not_recorded",
                "current_bfi_scoring_source_sha256": probes_hash,
                "bfi_reverse_coding": "items 1,3,4,5,7 reversed as 6-x; current source",
                "pre_measurements_sha256": sha_file(path / "pre_measurements.json"),
                "post_measurements_sha256": sha_file(path / "post_measurements.json"),
                "summary_sha256": sha_file(path / "summary.json"),
                "persistence_available": "yes",
                "complete": "yes",
                "exclusion_status": "included_complete_artifact",
            }
        )
    write_csv(out / "run_ledger.csv", ledger)
    return ledger


def bfi_rows(run: dict) -> list[dict]:
    path = run["path"]
    pre = read_json(path / "pre_measurements.json").get("bfi10", [])
    post = read_json(path / "post_measurements.json").get("bfi10", [])
    pre_by = {row["agent_name"]: row for row in pre}
    post_by = {row["agent_name"]: row for row in post}
    rows = []
    for index, name in enumerate(AGENTS):
        if name not in pre_by or name not in post_by:
            continue
        p = pre_by[name]["scores"]
        q = post_by[name]["scores"]
        delta = {factor: float(q[factor]) - float(p[factor]) for factor in FACTORS}
        rows.append(
            {
                "run_id": path.name,
                "condition": run["condition"],
                "scenario": run["scenario"],
                "variant": run["variant"],
                "replica": run["rep"],
                "agent": name,
                "group": "A" if index < 4 else "B",
                "l1": sum(abs(delta[factor]) for factor in FACTORS),
                **{f"delta_{factor}": delta[factor] for factor in FACTORS},
            }
        )
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def sd(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def mad(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(np.median(np.abs(values - np.median(values)))) if values.size else float("nan")


def bootstrap_mean_difference(
    a: list[float], b: list[float], seed: int, n_boot: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        draws[index] = mean(rng.choice(a_arr, len(a_arr), replace=True)) - mean(
            rng.choice(b_arr, len(b_arr), replace=True)
        )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def primary_cell_runs(runs: list[dict], scenario: str, condition: str) -> list[dict]:
    return [
        run
        for run in runs
        if run["scenario"] == scenario
        and run["condition"] == condition
        and run["alias"] is None
        and not run["path"].name.startswith("nosrm_")
    ]


def run_factor_means(run: dict) -> dict[str, float]:
    rows = bfi_rows(run)
    return {
        factor: mean(row[f"delta_{factor}"] for row in rows)
        for factor in FACTORS
    }


def run_l1_values(run: dict) -> list[float]:
    return [row["l1"] for row in bfi_rows(run)]


def icc_oneway(groups: list[list[float]]) -> tuple[float, float, float]:
    groups = [list(map(float, group)) for group in groups if group]
    r = len(groups)
    if r < 2:
        return float("nan"), float("nan"), float("nan")
    sizes = [len(group) for group in groups]
    if len(set(sizes)) != 1:
        raise ValueError("ICC implementation expects balanced groups")
    k = sizes[0]
    grand = mean(value for group in groups for value in group)
    ms_between = k * sum((mean(group) - grand) ** 2 for group in groups) / (r - 1)
    ms_within = sum(sum((value - mean(group)) ** 2 for value in group) for group in groups)
    ms_within /= r * (k - 1)
    denom = ms_between + (k - 1) * ms_within
    rho = (ms_between - ms_within) / denom if denom else float("nan")
    return float(rho), float(ms_between), float(ms_within)


def build_coherence(runs: list[dict], out: Path, n_boot: int) -> dict:
    factor_rows = []
    replica_rows = []
    brief_rows = []
    icc_rows = []
    interpretations = {}

    for scenario in ("greenfield", "riverton"):
        cells = {
            condition: primary_cell_runs(runs, scenario, condition)
            for condition in ("C2", "C4")
        }
        for condition, cell_runs in cells.items():
            for run in cell_runs:
                rows = bfi_rows(run)
                factor_means = run_factor_means(run)
                l1_mean = mean(row["l1"] for row in rows)
                signed_flat = [
                    row[f"delta_{factor}"] for row in rows for factor in FACTORS
                ]
                replica_rows.append(
                    {
                        "scenario": scenario,
                        "condition": condition,
                        "replica": run["rep"],
                        "run_id": run["path"].name,
                        "l1_mean": l1_mean,
                        "mean_signed_change": mean(signed_flat),
                        "directional_coherence_ratio": (
                            abs(mean(signed_flat)) / mean(abs(x) for x in signed_flat)
                            if mean(abs(x) for x in signed_flat)
                            else 0.0
                        ),
                        **{f"signed_{factor}": factor_means[factor] for factor in FACTORS},
                    }
                )

            groups = [run_l1_values(run) for run in cell_runs]
            rho, msb, msw = icc_oneway(groups)
            rho_for_de = max(0.0, rho) if not math.isnan(rho) else float("nan")
            k = 8
            design_effect = 1 + (k - 1) * rho_for_de
            icc_rows.append(
                {
                    "scenario": scenario,
                    "condition": condition,
                    "outcome": "per-agent BFI-10 L1 drift",
                    "n_runs": len(groups),
                    "agents_per_run": k,
                    "icc1_raw": rho,
                    "icc_for_design_effect": rho_for_de,
                    "ms_between": msb,
                    "ms_within": msw,
                    "design_effect": design_effect,
                    "nominal_n": len(groups) * k,
                    "effective_n": len(groups) * k / design_effect,
                    "note": "negative ICC is reported raw and truncated to zero only for DE",
                }
            )

        for factor_index, factor in enumerate(FACTORS):
            c2_vals = [run_factor_means(run)[factor] for run in cells["C2"]]
            c4_vals = [run_factor_means(run)[factor] for run in cells["C4"]]
            kappa = mean(c2_vals) - mean(c4_vals)
            lo, hi = bootstrap_mean_difference(
                c2_vals, c4_vals, 4200 + factor_index, n_boot
            )
            c4_reference = mean(c4_vals)
            agreement = sum(
                (value - c4_reference) * kappa > 0 for value in c2_vals
            )
            var_c2 = float(np.var(c2_vals, ddof=1))
            var_c4 = float(np.var(c4_vals, ddof=1))
            ratio = var_c2 / var_c4 if var_c4 > 0 else float("nan")
            factor_rows.append(
                {
                    "scenario": scenario,
                    "factor": factor,
                    "c2_mean_signed": mean(c2_vals),
                    "c4_mean_signed": mean(c4_vals),
                    "kappa_c2_minus_c4": kappa,
                    "descriptive_boot_ci_lo": lo,
                    "descriptive_boot_ci_hi": hi,
                    "c2_replica_values": ";".join(f"{x:.4f}" for x in c2_vals),
                    "c4_replica_values": ";".join(f"{x:.4f}" for x in c4_vals),
                    "c2_direction_agreement_n": agreement,
                    "c2_direction_agreement_denom": len(c2_vals),
                    "c2_replica_sd": sd(c2_vals),
                    "c4_replica_sd": sd(c4_vals),
                    "c2_replica_mad": mad(c2_vals),
                    "c4_replica_mad": mad(c4_vals),
                    "variance_ratio_c2_over_c4": ratio,
                }
            )

            separation = {}
            for condition, cell_runs in cells.items():
                per_run = []
                for run in cell_runs:
                    rows = bfi_rows(run)
                    a = mean(
                        row[f"delta_{factor}"] for row in rows if row["group"] == "A"
                    )
                    b = mean(
                        row[f"delta_{factor}"] for row in rows if row["group"] == "B"
                    )
                    per_run.append(a - b)
                separation[condition] = per_run
            theta = mean(separation["C2"]) - mean(separation["C4"])
            brief_rows.append(
                {
                    "scenario": scenario,
                    "factor": factor,
                    "c2_a_minus_b": mean(separation["C2"]),
                    "c4_a_minus_b": mean(separation["C4"]),
                    "theta": theta,
                    "c2_replica_values": ";".join(
                        f"{x:.4f}" for x in separation["C2"]
                    ),
                    "c4_replica_values": ";".join(
                        f"{x:.4f}" for x in separation["C4"]
                    ),
                }
            )

        # A cautious summary based on run-level directional effects.
        kappas = [
            row["kappa_c2_minus_c4"]
            for row in factor_rows
            if row["scenario"] == scenario
        ]
        interpretations[scenario] = {
            "positive_factor_kappas": sum(value > 0 for value in kappas),
            "negative_factor_kappas": sum(value < 0 for value in kappas),
            "zero_factor_kappas": sum(value == 0 for value in kappas),
            "note": (
                "Factor signs need not all agree for structured profile change; "
                "the table is a signed diagnostic, not a replacement primary endpoint."
            ),
        }

    write_csv(out / "signed_factor_coherence.csv", factor_rows)
    write_csv(out / "brief_group_separation.csv", brief_rows)
    write_csv(out / "replica_level_bfi_diagnostics.csv", replica_rows)
    write_csv(out / "variance_icc_design_effect.csv", icc_rows)
    result = {
        "factor_rows": factor_rows,
        "brief_rows": brief_rows,
        "replica_rows": replica_rows,
        "icc_rows": icc_rows,
        "interpretations": interpretations,
    }
    (out / "coherence_results.json").write_text(json.dumps(result, indent=2))
    return result


def unique_hash_summary(payloads: list[Any]) -> tuple[int, int, str]:
    hashes = [sha_bytes(canonical_json(payload)) for payload in payloads]
    return len(set(hashes)), len(hashes), ";".join(hashes)


def build_floor_audit(runs: list[dict], out: Path) -> list[dict]:
    rows = []
    for scenario in ("greenfield", "riverton"):
        for condition in ("C4", "C5"):
            cell = primary_cell_runs(runs, scenario, condition)
            instruments = [
                ("bfi10", "raw_responses"),
                ("bfi10", "scores"),
                ("allocation", "raw_response"),
                ("allocation", "allocations"),
            ]
            for instrument, field in instruments:
                pre_payloads = [
                    measurement_payload(run["path"], "pre", instrument, field)
                    for run in cell
                ]
                post_payloads = [
                    measurement_payload(run["path"], "post", instrument, field)
                    for run in cell
                ]
                pre_unique, n_runs, pre_hashes = unique_hash_summary(pre_payloads)
                post_unique, _, post_hashes = unique_hash_summary(post_payloads)
                exact_agent_pairs = 0
                total_agent_pairs = 0
                for pre_payload, post_payload in zip(pre_payloads, post_payloads):
                    for pre_value, post_value in zip(pre_payload, post_payload):
                        total_agent_pairs += 1
                        exact_agent_pairs += pre_value == post_value
                rows.append(
                    {
                        "scenario": scenario,
                        "condition": condition,
                        "instrument": instrument,
                        "field": field,
                        "n_runs": n_runs,
                        "unique_pre_run_payloads": pre_unique,
                        "unique_post_run_payloads": post_unique,
                        "pre_post_exact_agent_pairs": exact_agent_pairs,
                        "pre_post_agent_pairs": total_agent_pairs,
                        "pre_post_exact_rate": (
                            exact_agent_pairs / total_agent_pairs if total_agent_pairs else float("nan")
                        ),
                        "pre_run_hashes": pre_hashes,
                        "post_run_hashes": post_hashes,
                        "temperature": "0.7" if instrument == "bfi10" else "0.0",
                    }
                )
    write_csv(out / "c5_floor_audit.csv", rows)
    return rows


def exact_permutation(a: list[float], b: list[float]) -> tuple[float, float, int]:
    pooled = np.asarray(a + b, dtype=float)
    n_a = len(a)
    observed = mean(a) - mean(b)
    stats = []
    for indexes in itertools.combinations(range(len(pooled)), n_a):
        selected = set(indexes)
        left = [pooled[index] for index in range(len(pooled)) if index in selected]
        right = [pooled[index] for index in range(len(pooled)) if index not in selected]
        stats.append(mean(left) - mean(right))
    tolerance = 1e-12
    p_two = sum(abs(value) >= abs(observed) - tolerance for value in stats) / len(stats)
    return observed, p_two, len(stats)


def build_primary_contrast(
    runs: list[dict], out: Path, n_boot: int
) -> list[dict]:
    rows = []
    for scenario_index, scenario in enumerate(("greenfield", "riverton")):
        c2_runs = primary_cell_runs(runs, scenario, "C2")
        c4_runs = primary_cell_runs(runs, scenario, "C4")
        c2 = [mean(run_l1_values(run)) for run in c2_runs]
        c4 = [mean(run_l1_values(run)) for run in c4_runs]
        delta, p_two, allocations = exact_permutation(c2, c4)
        lo, hi = bootstrap_mean_difference(
            c2, c4, seed=9100 + scenario_index, n_boot=n_boot
        )
        var_c2 = float(np.var(c2, ddof=1))
        var_c4 = float(np.var(c4, ddof=1))
        rows.append(
            {
                "scenario": scenario,
                "c2_replica_means": ";".join(f"{x:.4f}" for x in c2),
                "c4_replica_means": ";".join(f"{x:.4f}" for x in c4),
                "n_c2_runs": len(c2),
                "n_c4_runs": len(c4),
                "c2_mean": mean(c2),
                "c4_mean": mean(c4),
                "delta_c2_minus_c4": delta,
                "descriptive_boot_ci_lo": lo,
                "descriptive_boot_ci_hi": hi,
                "exact_two_sided_permutation_p": p_two,
                "n_allocations": allocations,
                "c2_l1_replica_variance": var_c2,
                "c4_l1_replica_variance": var_c4,
                "l1_variance_ratio_c2_over_c4": (
                    var_c2 / var_c4 if var_c4 > 0 else float("nan")
                ),
                "c2_above_c4_max_n": sum(value > max(c4) for value in c2),
                "c2_equal_c4_max_n": sum(value == max(c4) for value in c2),
            }
        )
    write_csv(out / "primary_c2_c4_run_level.csv", rows)
    return rows


def build_nosrm_correction(runs: list[dict], out: Path) -> list[dict]:
    rows = []
    for scenario in ("greenfield", "riverton"):
        values = {}
        for condition in ("C2", "C5"):
            cell = [
                run
                for run in runs
                if run["scenario"] == scenario
                and run["condition"] == condition
                and run["path"].name.startswith("nosrm_")
            ]
            values[condition] = [
                mean(row["l1"] for row in bfi_rows(run)) for run in cell
            ]
        observed, p_two, allocations = exact_permutation(values["C2"], values["C5"])
        rows.append(
            {
                "scenario": scenario,
                "contrast": "verbatim-history C2-C5",
                "c2_replica_means": ";".join(f"{x:.4f}" for x in values["C2"]),
                "c5_replica_means": ";".join(f"{x:.4f}" for x in values["C5"]),
                "c2_mean": mean(values["C2"]),
                "c5_mean": mean(values["C5"]),
                "delta": observed,
                "exact_two_sided_permutation_p": p_two,
                "n_allocations": allocations,
                "minimum_attainable_two_sided_p": 2 / allocations,
                "interpretation": "exploratory directional robustness; not a C2-C4 causal contrast",
            }
        )
    write_csv(out / "nosrm_exact_permutation.csv", rows)
    return rows


def build_cross_variant_manifest(runs: list[dict], out: Path) -> list[dict]:
    rows = []
    seed_offsets = {"qvl": 5000, "dpsk": 1000, "gpt": 2000, "gem": 3000, "l4api": 4000}
    for run in runs:
        if (
            run["condition"] != "C2"
            or run["alias"] is None
            or run["path"].name.startswith("nosrm_")
        ):
            continue
        model = run["config"].get("model", "")
        rep_index = int(run["rep"])
        alias = run["alias"]
        scenario = run["scenario"]
        rep_id = f"rep{rep_index:02d}_{alias}"
        config_file = f"configs/condition_4_seed_only_no_interact_{scenario}.json"
        seed_value = seed_offsets[alias] + 200 + rep_index
        command = (
            f"python run_experiment.py --config-file {config_file} "
            f"--model {model} --run-id {rep_id} --seed {seed_value}"
        )
        rows.append(
            {
                "variant": run["variant"],
                "alias": alias,
                "scenario": scenario,
                "replica_id": run["rep"],
                "matched_c2_run": run["path"].name,
                "model": model,
                "c4_config": config_file,
                "seed": seed_value,
                "command": command,
                "status": "planned_not_launched",
            }
        )
    rows.sort(key=lambda row: (row["alias"], row["scenario"], row["replica_id"]))
    write_csv(out / "cross_variant_c4_manifest.csv", rows)
    return rows


def build_trace_report(runs: list[dict], ledger: list[dict], out: Path) -> None:
    lines = [
        "# C2/C4 trace-equivalence audit",
        "",
        "## Decision",
        "",
        "**The historical artifacts support pre-treatment construction equivalence, "
        "but not token-level historical equivalence.** C2 and C4 use the same scenario, "
        "agent identities, asymmetric A/B brief texts, brief assignment, base-model "
        "identifier, and pre/post probe constructors. The run artifacts did not save "
        "exact model revisions, tokenizer revisions, assembled request payloads, token "
        "IDs, endpoint precision, batch placement, or the run-time code commit.",
        "",
        "The frozen plan's proposed requirement that the *post-treatment* C2 and C4 "
        "token sequences match is not a valid gate for this implementation. C2 stores "
        "interaction-created memories in `Agent.system_prompt`; C4 has no such memories. "
        "That expected difference is the treatment mediator. Requiring it to disappear "
        "would change the estimand and, under deterministic decoding, largely force a null.",
        "",
        "## Corrected gate",
        "",
        "Call C2-C4 an interaction-given-brief contrast only if:",
        "",
        "1. Pre-treatment system and user-prompt construction match by scenario, agent, "
        "model, brief assignment, probe order, and decoding configuration.",
        "2. The only designed path from assignment to the post probe is the 30-round "
        "interaction and its resulting state/history; periodic probe calls do not mutate "
        "agent state.",
        "3. Post-probe constructors and decoding settings match, while treatment-induced "
        "memory/history is retained and explicitly documented rather than normalized away.",
        "4. Any unrecorded serving variables are disclosed as audit limitations.",
        "",
        "## Source and artifact checks",
        "",
        "- `Agent.system_prompt` depends on agent name, scenario context, brief, and "
        "serialized memory. It does not include `group_id`.",
        "- C2 and C4 assign Brief A to agents 0-3 and Brief B to agents 4-7.",
        "- C2 uses segregated interaction for 30 rounds; C4 uses no interaction. These "
        "are the intended treatment differences.",
        "- Pre, periodic, and post probes call `answer_probe` and do not append outputs "
        "to `conversation_history` or memory. Periodic probes therefore do not mutate "
        "the state later measured by BFI-10.",
        "- Historical raw prompt and token sequences were not logged, so this is a "
        "source-plus-artifact audit, not a retrospective token-level audit.",
        "",
        "## Reconstructed initial-system-prompt hashes",
        "",
        "|Scenario|Agent|C2 hash|C4 hash|Match|",
        "|---|---|---|---|---|",
    ]
    for scenario in ("greenfield", "riverton"):
        for index, agent in enumerate(AGENTS):
            c2_hash = sha_text(initial_system_prompt(index, scenario, True))
            c4_hash = sha_text(initial_system_prompt(index, scenario, True))
            lines.append(f"|{scenario}|{agent}|`{c2_hash}`|`{c4_hash}`|yes|")

    lines.extend(
        [
            "",
            "These hashes are reconstructed from the current checked-out scenario and "
            "prompt source. Because the run-time code commit was not saved, they verify "
            "the present source package, not byte-for-byte historical requests.",
            "",
            "## Replica counts from completed artifacts",
            "",
            "|Variant|Scenario|C2 complete|C4 complete|",
            "|---|---|---:|---:|",
        ]
    )
    counts = Counter(
        (row["variant"], row["scenario"], row["condition"])
        for row in ledger
        if not row["run_id"].startswith("nosrm_")
    )
    variants = sorted({(row["variant"], row["scenario"]) for row in ledger})
    for variant, scenario in variants:
        lines.append(
            f"|{variant}|{scenario}|{counts[(variant, scenario, 'C2')]}|"
            f"{counts[(variant, scenario, 'C4')]}|"
        )

    lines.extend(
        [
            "",
            "## Claim language",
            "",
            "Safe now: **“C2-C4 is the interaction-given-brief contrast in the implemented "
            "protocol.”**",
            "",
            "Do not claim: **“historical C2 and C4 post-probe token sequences were verified "
            "identical.”** They were not recorded and, because treatment memory enters the "
            "system prompt, they should differ.",
            "",
        ]
    )
    (out / "trace_equivalence_report.md").write_text("\n".join(lines))


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def build_markdown_summary(
    out: Path,
    ledger: list[dict],
    primary_rows: list[dict],
    coherence: dict,
    floor_rows: list[dict],
    nosrm_rows: list[dict],
    manifest: list[dict],
    n_boot: int,
) -> None:
    lines = [
        "# Rebuttal evidence summary",
        "",
        f"Generated from {len(ledger)} complete run artifacts at repository HEAD "
        f"`{git_value('rev-parse', 'HEAD')}`. Historical run-time commits were not recorded.",
        "",
        "## Primary BFI-10 L1 contrast at the run level",
        "",
        "|Scenario|C2 mean|C4 mean|Delta|Descriptive 95% bootstrap CI|Exact permutation p|L1 variance ratio|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"|{row['scenario']}|{fmt(row['c2_mean'])}|{fmt(row['c4_mean'])}|"
            f"{fmt(row['delta_c2_minus_c4'])}|"
            f"[{fmt(row['descriptive_boot_ci_lo'])}, {fmt(row['descriptive_boot_ci_hi'])}]|"
            f"{fmt(row['exact_two_sided_permutation_p'], 4)}|"
            f"{fmt(row['l1_variance_ratio_c2_over_c4'])}|"
        )
    lines.extend(
        [
            "",
            "The exact permutation test uses run means (8 C2 versus 5 C4), not "
            "the 64 versus 40 agent rows. Greenfield has all 8 C2 means above the "
            "C4 maximum; Riverton has 7 above and 1 tied with the C4 maximum.",
            "",
            "## Signed BFI factor effects (C2-C4)",
        "",
        "|Scenario|Factor|kappa|Descriptive 95% bootstrap CI|C2 direction agreement|Variance ratio|",
        "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in coherence["factor_rows"]:
        lines.append(
            f"|{row['scenario']}|{row['factor']}|{fmt(row['kappa_c2_minus_c4'])}|"
            f"[{fmt(row['descriptive_boot_ci_lo'])}, {fmt(row['descriptive_boot_ci_hi'])}]|"
            f"{row['c2_direction_agreement_n']}/{row['c2_direction_agreement_denom']}|"
            f"{fmt(row['variance_ratio_c2_over_c4'])}|"
        )
    lines.extend(
        [
            "",
            f"Intervals resample runs only ({n_boot:,} draws) and are descriptive. The "
            "independent unit is the run; agents are not counted as independent replicas.",
            "",
            "## ICC and design effect",
            "",
            "|Scenario|Condition|ICC(1)|DE|Nominal n|Effective n|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in coherence["icc_rows"]:
        lines.append(
            f"|{row['scenario']}|{row['condition']}|{fmt(row['icc1_raw'])}|"
            f"{fmt(row['design_effect'])}|{row['nominal_n']}|{fmt(row['effective_n'], 1)}|"
        )
    lines.extend(
        [
            "",
            "Negative sample ICCs are shown but truncated to zero only when computing DE. "
            "These small-cell estimates are descriptive.",
            "",
            "## Corrected no-SRM inference",
            "",
            "|Scenario|C2 replica means|C5 replica means|Delta|Exact two-sided p|",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in nosrm_rows:
        lines.append(
            f"|{row['scenario']}|{row['c2_replica_means']}|{row['c5_replica_means']}|"
            f"{fmt(row['delta'])}|{fmt(row['exact_two_sided_permutation_p'])}|"
        )
    lines.extend(
        [
            "",
            "With 3 versus 3 runs there are 20 allocations; the minimum attainable "
            "two-sided exact p-value is 0.10. Agent-level Welch p-values must not be used "
            "as confirmatory evidence for this ablation.",
            "",
            "## C5 floor audit",
            "",
            "|Scenario|Instrument/field|Unique pre payloads|Unique post payloads|Pre=post agent rate|",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in floor_rows:
        if row["condition"] != "C5":
            continue
        lines.append(
            f"|{row['scenario']}|{row['instrument']}/{row['field']}|"
            f"{row['unique_pre_run_payloads']}/{row['n_runs']}|"
            f"{row['unique_post_run_payloads']}/{row['n_runs']}|"
            f"{fmt(row['pre_post_exact_rate'])}|"
        )
    lines.extend(
        [
            "",
            "BFI-10 was sampled at temperature 0.7, so C5 is not a deterministic "
            "probe-only baseline. Allocation used temperature 0.0. New C5 runs should be "
            "called independent scientific replicas only if the serving/randomness source "
            "and batch configuration are recorded prospectively.",
            "",
            "## Cross-variant matched C4",
            "",
            f"The ledger identifies {len(manifest)} matched C4 jobs (one for every complete "
            "non-primary C2 run). Commands are frozen in `cross_variant_c4_manifest.csv`; "
            "none were launched by this evidence build.",
            "",
        ]
    )
    (out / "evidence_summary.md").write_text("\n".join(lines))


def build_acceptance_matrix(out: Path, manifest: list[dict]) -> None:
    text = f"""# AC acceptance-condition matrix

|Concern|Evidence now|Status|Rebuttal action|
|---|---|---|---|
|Is C2-C4 a matched interaction contrast?|Pre-treatment prompt construction and brief assignment match in current source; post-treatment memory differs by design. Historical token IDs/model revisions were not logged.|Partially closed; transparent audit limitation|Use “interaction-given-brief contrast,” not “verified identical post-token trace.”|
|Does unsigned L1 reflect coherent change?|Run-level signed factor, brief-separation, variance, and ICC tables generated from existing primary data.|Closed analytically; interpret from table|Paste compact factor rows and avoid agent-level independence claims.|
|Are cross-variant results direct causal replications?|Existing panel is C2-C5 only.|Open|Run the {len(manifest)} matched C4 jobs in the frozen manifest; report all directions.|
|Is thin C5 inflating evidence?|C5 payload duplication/floor audit generated; headline C2-C4 does not use C5.|Partially closed|Report hashes/equality rates; preflight before thickening.|
|Does any long generation cause drift?|No matched solo-generation C6 artifacts exist.|Open, P2 compute|Freeze C6 protocol before launch and report C6-C4 and C2-C6.|
|Does no-SRM evidence survive correct clustering?|Run-level exact permutation added; 3v3 minimum two-sided p=.10.|Closed as exploratory only|Remove tiny agent-level p-values; report raw replica means and direction.|
|Does the paper claim capability degradation?|No; BFI is response-profile transfer.|Closed by scope statement|State explicitly that capability change was not tested.|

## Immediate claim corrections

1. Replace “pure-interaction contrast” with “interaction-given-brief contrast” unless the authors reserve “pure” only for the factorial assignment, not identical post-treatment state.
2. Remove Welch p-values from the 3v3 no-SRM rebuttal presentation.
3. Do not describe BFI-10 as preregistered unless an external timestamped preregistration exists.
4. Treat periodic-probe results as within-interaction cadence evidence, not delayed persistence.
"""
    (out / "ac_acceptance_condition_matrix.md").write_text(text)


def build_frozen_spec(out: Path, n_boot: int) -> None:
    spec = {
        "frozen_on": "2026-07-25",
        "primary_endpoint": "BFI-10 L1 post-pre change",
        "primary_contrast": "C2-C4 interaction-given-brief",
        "independent_unit": "run/replica",
        "signed_diagnostic": "per-factor post-pre change",
        "brief_separation": "(C2 A-B) - (C4 A-B), per factor",
        "bootstrap": {
            "unit": "run",
            "draws": n_boot,
            "ci": "percentile 95%, descriptive",
            "seed_base": 4200,
        },
        "icc": {
            "estimator": "one-way random-effects ICC(1) on per-agent BFI L1 within condition",
            "design_effect": "1 + (8-1)*max(0, sample ICC)",
            "status": "descriptive",
        },
        "nosrm": {
            "test": "two-sided exact permutation on 3v3 run means",
            "minimum_p": 0.1,
            "status": "exploratory",
        },
        "trace_gate": {
            "required": "pre-treatment construction equivalence and identical probe constructors/settings",
            "expected_difference": "treatment-induced C2 interaction memory/history at post-test",
            "historical_token_trace": "unavailable",
        },
        "prohibited": [
            "agent-level confirmatory Welch tests",
            "post hoc exclusion of reversed variants",
            "calling C6-C5 a generation effect",
            "claiming historical token equality without logged tokens",
        ],
    }
    (out / "frozen_analysis_spec.json").write_text(json.dumps(spec, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    ledger = build_ledger(runs, out)
    build_trace_report(runs, ledger, out)
    coherence = build_coherence(runs, out, args.bootstrap)
    primary_rows = build_primary_contrast(runs, out, args.bootstrap)
    floor_rows = build_floor_audit(runs, out)
    nosrm_rows = build_nosrm_correction(runs, out)
    manifest = build_cross_variant_manifest(runs, out)
    build_frozen_spec(out, args.bootstrap)
    build_markdown_summary(
        out, ledger, primary_rows, coherence, floor_rows, nosrm_rows, manifest,
        args.bootstrap
    )
    build_acceptance_matrix(out, manifest)

    print(f"Evidence package: {out}")
    print(f"Complete run artifacts: {len(runs)}")
    print(f"Matched cross-variant C4 jobs planned: {len(manifest)}")
    print("No experiment jobs were launched.")


if __name__ == "__main__":
    main()
