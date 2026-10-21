"""Analyze the verbatim-history no-SRM ablation.

This is intentionally separate from analysis_p3_conformity_bfi.py so the
pilot robustness runs do not enter the main factorial analyses. It computes
the BFI-10 L1 drift contrast used in the SRM-confound check:

    no-SRM C2 - no-SRM C5

with the same unit of analysis as the paper: per-agent BFI-10 L1 drift, and a
replica-clustered bootstrap CI that resamples replicas and then agents within
replicas.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

DATA = (Path(__file__).resolve().parents[1] / "data")
OUT = DATA / "analysis_no_srm_ablation.json"
N_BOOT = 10_000
SEED = 42
BFI_FACTORS = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
]


def run_key(path: Path) -> str:
    m = re.match(r"nosrm_rep(\d\d)_", path.name)
    return m.group(1) if m else path.name


def condition_key(path: Path) -> str | None:
    name = path.name
    if "_segreg_asym_seed_" in name:
        return "C2"
    if "_no_seed_no_interact_" in name:
        return "C5"
    return None


def scenario_key(path: Path) -> str | None:
    name = path.name
    if "_greenfield_" in name:
        return "greenfield"
    if "_riverton_" in name:
        return "riverton"
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def bfi_rows(run_dir: Path) -> list[dict]:
    pre = load_json(run_dir / "pre_measurements.json")
    post = load_json(run_dir / "post_measurements.json")
    pre_bfi = {r["agent_name"]: r for r in pre.get("bfi10", [])}
    post_bfi = {r["agent_name"]: r for r in post.get("bfi10", [])}
    rows = []
    for agent in sorted(set(pre_bfi) & set(post_bfi)):
        pre_scores = pre_bfi[agent].get("scores") or {}
        post_scores = post_bfi[agent].get("scores") or {}
        if not pre_scores or not post_scores:
            continue
        deltas = {
            f: float(post_scores.get(f, float("nan")))
            - float(pre_scores.get(f, float("nan")))
            for f in BFI_FACTORS
        }
        if any(math.isnan(v) for v in deltas.values()):
            continue
        rows.append(
            {
                "dir": run_dir.name,
                "rep": run_key(run_dir),
                "agent": agent,
                "l1_drift": float(sum(abs(deltas[f]) for f in BFI_FACTORS)),
                **{f"delta_{f}": float(deltas[f]) for f in BFI_FACTORS},
            }
        )
    return rows


def hedges_g(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled < 0.05:
        return None
    correction = 1 - 3 / (4 * (na + nb - 2) - 1)
    return float(((np.mean(a) - np.mean(b)) / pooled) * correction)


def cluster_bootstrap_ci(
    a_by_rep: dict[str, list[float]],
    b_by_rep: dict[str, list[float]],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float]:
    a_reps = sorted(a_by_rep)
    b_reps = sorted(b_by_rep)
    diffs = []
    for _ in range(n_boot):
        a_sample = []
        for rep in rng.choice(a_reps, size=len(a_reps), replace=True):
            vals = np.asarray(a_by_rep[rep], dtype=float)
            a_sample.extend(rng.choice(vals, size=len(vals), replace=True))
        b_sample = []
        for rep in rng.choice(b_reps, size=len(b_reps), replace=True):
            vals = np.asarray(b_by_rep[rep], dtype=float)
            b_sample.extend(rng.choice(vals, size=len(vals), replace=True))
        diffs.append(float(np.mean(a_sample) - np.mean(b_sample)))
    return tuple(float(x) for x in np.percentile(diffs, [2.5, 97.5]))


def summarize_scenario(scenario: str) -> dict:
    rows_by_cond = {"C2": [], "C5": []}
    for run_dir in sorted(DATA.glob(f"nosrm_rep*_{scenario}_*")):
        cond = condition_key(run_dir)
        if cond not in rows_by_cond:
            continue
        rows_by_cond[cond].extend(bfi_rows(run_dir))

    by_rep = {"C2": {}, "C5": {}}
    for cond, rows in rows_by_cond.items():
        for row in rows:
            by_rep[cond].setdefault(row["rep"], []).append(row["l1_drift"])

    a = np.asarray([r["l1_drift"] for r in rows_by_cond["C2"]], dtype=float)
    b = np.asarray([r["l1_drift"] for r in rows_by_cond["C5"]], dtype=float)
    rng = np.random.default_rng(SEED)
    lo, hi = cluster_bootstrap_ci(by_rep["C2"], by_rep["C5"], rng)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    g = hedges_g(a, b)

    return {
        "scenario": scenario,
        "contrast": "no-SRM C2-C5",
        "metric": "BFI-10 L1 drift",
        "condition_means": {
            "C2": float(np.mean(a)),
            "C5": float(np.mean(b)),
        },
        "delta": float(np.mean(a) - np.mean(b)),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "hedges_g": g,
        "welch_t": float(t),
        "welch_p": float(p),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_agents": {"C2": int(len(a)), "C5": int(len(b))},
        "n_reps": {"C2": len(by_rep["C2"]), "C5": len(by_rep["C5"])},
        "per_rep_means": {
            cond: {rep: float(np.mean(vals)) for rep, vals in reps.items()}
            for cond, reps in by_rep.items()
        },
        "per_agent": rows_by_cond,
    }


def main() -> None:
    result = {
        "analysis": "verbatim-history no-SRM ablation",
        "note": (
            "C2 uses segregated asymmetric briefs with SRM extraction disabled; "
            "C5 uses no brief and no interaction with the same no-SRM flag. "
            "This is a C2-C5 pilot robustness check, not a replacement for the "
            "main C2-C4 pure-interaction contrast."
        ),
        "bootstrap": {
            "type": "two-level replica-clustered bootstrap",
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "scenarios": {
            scenario: summarize_scenario(scenario)
            for scenario in ["greenfield", "riverton"]
        },
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"Saved: {OUT}")
    for scenario, row in result["scenarios"].items():
        g = row["hedges_g"]
        g_s = "undef" if g is None else f"{g:+.2f}"
        print(
            f"{scenario:10s} {row['contrast']}: "
            f"C2={row['condition_means']['C2']:.2f}, "
            f"C5={row['condition_means']['C5']:.2f}, "
            f"Delta={row['delta']:+.2f}, "
            f"CI=[{row['ci95_lo']:+.2f},{row['ci95_hi']:+.2f}], "
            f"g={g_s}, p={row['welch_p']:.4g}, "
            f"n={row['n_agents']['C2']} vs {row['n_agents']['C5']}"
        )


if __name__ == "__main__":
    main()
