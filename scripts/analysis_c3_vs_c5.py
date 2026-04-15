"""analysis_c3_vs_c5.py — C3 (Segreg+NoSeed) vs C5 (NoSeed+NoInteract) Part-1 contrast.

Tests the core question (auto-memory: "C3 critical ablation"):
    "Does interaction alone — without any seed asymmetry — create
     allocation divergence above the no-interaction baseline?"

Spec: Chapter 4 §4.3.1
    Per-dimension allocation divergence between groups A and B, with
    Hedges' g (small-sample-corrected Cohen's d) and 95% bootstrap CI
    (10,000 resamples, stratified by rep), AFTER subtracting the matching
    quantity from Condition 5.

Because C5 has no real group structure, its baseline is computed as the
expected |mean_A - mean_B| under random 4-vs-4 splits of its 8 agents
(permutation null, 1000 splits per rep, then averaged within rep).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

DATA = Path("/home/yandan/cogsociety/data")
OUTPUT = DATA / "analysis_c3_vs_c5_greenfield.json"
SCENARIO = "greenfield"
AREAS = ["Education", "Healthcare", "Green Spaces", "Infrastructure", "Economic Development"]
N_BOOT = 10_000
N_PERM_NULL = 1_000
ALPHA = 0.05
SEED = 42


def load_run(run_dir: Path):
    """Return dict with names, groups, X (n_agents × n_areas) — or None."""
    try:
        summary = json.load(open(run_dir / "summary.json"))
        post = json.load(open(run_dir / "post_measurements.json"))
    except FileNotFoundError:
        return None
    name_to_group = {a["name"]: a.get("group_id") for a in summary["agents"]}
    rows, groups, names = [], [], []
    for entry in post.get("allocation", []) or []:
        n = entry.get("agent_name")
        a = entry.get("allocations") or {}
        if not (isinstance(a, dict) and n):
            continue
        rows.append([float(a.get(area, 0.0)) for area in AREAS])
        groups.append(name_to_group.get(n))
        names.append(n)
    if not rows:
        return None
    return {"names": names, "groups": np.array(groups, dtype=object), "X": np.asarray(rows, dtype=float)}


def divergence_per_dim(X: np.ndarray, group_ids, a_id=0, b_id=1) -> np.ndarray:
    a_mask = np.array([g == a_id for g in group_ids])
    b_mask = np.array([g == b_id for g in group_ids])
    if a_mask.sum() == 0 or b_mask.sum() == 0:
        return np.full(X.shape[1], np.nan)
    return np.abs(X[a_mask].mean(0) - X[b_mask].mean(0))


def null_divergence_mean(X: np.ndarray, rng: np.random.Generator, n_perm=N_PERM_NULL) -> np.ndarray:
    """Expected |mean_A - mean_B| under random 4-vs-4 splits of an 8-agent C5 rep."""
    n = X.shape[0]
    half = n // 2
    accum = np.zeros(X.shape[1])
    idx = np.arange(n)
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        a, b = perm[:half], perm[half : 2 * half]
        accum += np.abs(X[a].mean(0) - X[b].mean(0))
    return accum / n_perm


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    s_pool = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pool == 0:
        return 0.0
    d = (mx - my) / s_pool
    df = nx + ny - 2
    j = 1 - 3 / (4 * df - 1)  # Hedges' small-sample correction
    return d * j


def bootstrap_contrast(c3_rep: np.ndarray, c5_rep: np.ndarray, rng: np.random.Generator):
    """Stratified bootstrap on (C3 mean − C5 mean) per dimension.

    c3_rep:  (n_c3_reps, n_areas)  observed |mean_A − mean_B| per C3 rep
    c5_rep:  (n_c5_reps, n_areas)  expected |mean_A − mean_B| under random splits per C5 rep
    Returns array shape (n_areas, 5): [delta, ci_lo, ci_hi, hedges_g, dim_idx]
    """
    n_areas = c3_rep.shape[1]
    n3, n5 = c3_rep.shape[0], c5_rep.shape[0]
    out = np.empty((n_areas, 5))
    for j in range(n_areas):
        a = c3_rep[:, j]
        b = c5_rep[:, j]
        diffs = np.empty(N_BOOT)
        for k in range(N_BOOT):
            ai = rng.integers(0, n3, n3)
            bi = rng.integers(0, n5, n5)
            diffs[k] = a[ai].mean() - b[bi].mean()
        lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out[j] = [a.mean() - b.mean(), lo, hi, hedges_g(a, b), j]
    return out


def main():
    rng = np.random.default_rng(SEED)

    c3_dirs = sorted(DATA.glob(f"rep*_segreg_no_seed_{SCENARIO}_*"))
    c5_dirs = sorted(DATA.glob(f"rep*_no_seed_no_interact_{SCENARIO}_*"))
    print(f"C3 (Segreg+NoSeed) reps found: {len(c3_dirs)}")
    print(f"C5 (NoSeed+NoInteract) reps found: {len(c5_dirs)}")
    if not c3_dirs or not c5_dirs:
        print("ERROR: need at least one rep of each condition")
        sys.exit(1)

    print("\n--- C3 per-rep divergence (observed group A vs B) ---")
    c3_per_rep = []
    for d in c3_dirs:
        run = load_run(d)
        if run is None:
            print(f"  SKIP {d.name}: no parseable allocation"); continue
        div = divergence_per_dim(run["X"], run["groups"], 0, 1)
        if np.isnan(div).any():
            print(f"  SKIP {d.name}: missing group_id"); continue
        c3_per_rep.append(div)
        print(f"  {d.name}: " + "  ".join(f"{a[:4]}={v:5.2f}" for a, v in zip(AREAS, div)))
    c3_per_rep = np.vstack(c3_per_rep)

    print("\n--- C5 per-rep null divergence (mean over 1000 random 4-vs-4 splits) ---")
    c5_per_rep = []
    for d in c5_dirs:
        run = load_run(d)
        if run is None:
            print(f"  SKIP {d.name}: no parseable allocation"); continue
        null_mean = null_divergence_mean(run["X"], rng)
        c5_per_rep.append(null_mean)
        print(f"  {d.name}: " + "  ".join(f"{a[:4]}={v:5.2f}" for a, v in zip(AREAS, null_mean)))
    c5_per_rep = np.vstack(c5_per_rep)

    print(f"\nShapes: C3 {c3_per_rep.shape}, C5 {c5_per_rep.shape}")

    res = bootstrap_contrast(c3_per_rep, c5_per_rep, rng)

    print("\n=== Part-1 contrast: C3 (interaction-only) − C5 (baseline null), Greenfield ===")
    header = f"{'Dimension':<22}{'C3 mean':>10}{'C5 mean':>10}{'Δ':>10}{'95% CI':>20}{'Hedges g':>11}{'  signal'}"
    print(header)
    print("-" * len(header))
    n_sig = 0
    for i, area in enumerate(AREAS):
        delta, lo, hi, g, _ = res[i]
        sig = "✓" if (lo > 0 or hi < 0) else " "
        if sig == "✓":
            n_sig += 1
        c3m = float(c3_per_rep[:, i].mean())
        c5m = float(c5_per_rep[:, i].mean())
        print(
            f"{area:<22}{c3m:>10.2f}{c5m:>10.2f}{delta:>+10.2f}"
            f"   [{lo:+5.2f}, {hi:+5.2f}]   {g:>+8.2f}    {sig}"
        )

    print(f"\nDimensions where 95% CI on (C3 − C5) excludes zero: {n_sig}/{len(AREAS)}")
    if n_sig >= 2:
        verdict = "SIGNAL"
        narrative = (
            "Interaction alone produces divergence above baseline on multiple dimensions.\n"
            "  Paper narrative: C3 critical ablation is positive; C2 amplifies further."
        )
    elif n_sig == 1:
        verdict = "WEAK"
        narrative = (
            "Only one dimension significant — could be noise. Defer call until C2 data lands."
        )
    else:
        verdict = "NULL"
        narrative = (
            "Interaction without seed asymmetry does NOT exceed baseline.\n"
            "  Paper narrative shift: C2 (seed × interaction) becomes the ONLY load-bearing condition.\n"
            "  Novelty hook narrows to 'seed asymmetry that produces ~zero individual\n"
            "   effect generates divergence ONLY when combined with interaction.'"
        )
    print(f"\n>>> VERDICT: {verdict}\n  {narrative}")

    out = {
        "scenario": SCENARIO,
        "n_c3_reps": int(c3_per_rep.shape[0]),
        "n_c5_reps": int(c5_per_rep.shape[0]),
        "n_bootstrap": N_BOOT,
        "n_perm_null": N_PERM_NULL,
        "alpha": ALPHA,
        "areas": AREAS,
        "c3_per_rep": c3_per_rep.tolist(),
        "c5_per_rep_null_mean": c5_per_rep.tolist(),
        "results": [
            {
                "area": AREAS[i],
                "c3_mean": float(c3_per_rep[:, i].mean()),
                "c5_mean": float(c5_per_rep[:, i].mean()),
                "delta": float(res[i, 0]),
                "ci95_lo": float(res[i, 1]),
                "ci95_hi": float(res[i, 2]),
                "hedges_g": float(res[i, 3]),
                "ci_excludes_zero": bool(res[i, 1] > 0 or res[i, 2] < 0),
            }
            for i in range(len(AREAS))
        ],
        "n_significant_dims": int(n_sig),
        "verdict": verdict,
    }
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
