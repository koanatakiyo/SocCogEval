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
    # Mirror analysis_part1_contrasts.HEDGES_G_SD_FLOOR. Returns NaN when the
    # pooled SD is too small for d to be interpretable (typically the C5
    # constant-baseline case).
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    s_pool = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pool < 0.05:
        return float("nan")
    d = (mx - my) / s_pool
    df = nx + ny - 2
    j = 1 - 3 / (4 * df - 1)  # Hedges' small-sample correction
    return d * j


def _prep_c3_rep(run: dict):
    """Split a C3 rep into (X_a, X_b) — agent allocations for group 0 and 1."""
    a_mask = np.array([g == 0 for g in run["groups"]])
    b_mask = np.array([g == 1 for g in run["groups"]])
    return run["X"][a_mask], run["X"][b_mask]


def bootstrap_contrast(
    c3_runs: list[dict], c5_runs: list[dict], rng: np.random.Generator
):
    """Agent-stratified cluster bootstrap on (C3 div − C5 null) per dimension.

    Implements §4.3.4: "95% bootstrap CI, stratified by agent, ≥ 10,000 resamples."
    Two levels of resampling per iteration:
      (cluster)   resample reps with replacement
      (within)    within each sampled rep, resample agents with replacement
                  separately in group A and group B (stratified) — this is
                  what makes agent-level variance propagate into the CI.

    For C5, which has no real groups, the "within-rep" step instead performs
    a random 4-vs-4 partition of the 8 agents AND resamples agents with
    replacement within each of those halves (to match the C3 variance
    structure under the null).

    Returns (res, c3_divs, c5_nulls) where:
      res        — (n_areas, 5): [delta, ci_lo, ci_hi, hedges_g, dim_idx]
      c3_divs    — (n_c3_reps, n_areas) observed rep-level divergences
      c5_nulls   — (n_c5_reps, n_areas) expected null divergences per C5 rep
    """
    n_areas = c3_runs[0]["X"].shape[1]
    n3, n5 = len(c3_runs), len(c5_runs)

    # Pre-split C3 reps into group-A / group-B agent matrices
    c3_split = [_prep_c3_rep(r) for r in c3_runs]

    # Point estimates for Hedges' g and the display row
    c3_divs = np.vstack([
        np.abs(Xa.mean(0) - Xb.mean(0)) for (Xa, Xb) in c3_split
    ])
    c5_nulls = np.vstack([null_divergence_mean(r["X"], rng) for r in c5_runs])

    diffs = np.empty((N_BOOT, n_areas))
    for k in range(N_BOOT):
        # ── C3: resample reps, then resample agents within each group of each sampled rep
        rep_sample_c3 = rng.integers(0, n3, n3)
        c3_vals = np.empty((n3, n_areas))
        for out_i, rep_i in enumerate(rep_sample_c3):
            Xa, Xb = c3_split[rep_i]
            na, nb = len(Xa), len(Xb)
            a_boot = rng.integers(0, na, na)
            b_boot = rng.integers(0, nb, nb)
            c3_vals[out_i] = np.abs(Xa[a_boot].mean(0) - Xb[b_boot].mean(0))
        c3_mean_k = c3_vals.mean(0)

        # ── C5: resample reps, within each draw random 4-vs-4 split + agent resample
        rep_sample_c5 = rng.integers(0, n5, n5)
        c5_vals = np.empty((n5, n_areas))
        for out_i, rep_i in enumerate(rep_sample_c5):
            X = c5_runs[rep_i]["X"]
            n = X.shape[0]
            half = n // 2
            perm = rng.permutation(n)
            a_idx, b_idx = perm[:half], perm[half : 2 * half]
            # resample within each half
            a_boot = rng.integers(0, half, half)
            b_boot = rng.integers(0, half, half)
            c5_vals[out_i] = np.abs(X[a_idx][a_boot].mean(0) - X[b_idx][b_boot].mean(0))
        c5_mean_k = c5_vals.mean(0)

        diffs[k] = c3_mean_k - c5_mean_k

    out = np.empty((n_areas, 5))
    for j in range(n_areas):
        lo, hi = np.percentile(diffs[:, j], [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out[j] = [
            c3_divs[:, j].mean() - c5_nulls[:, j].mean(),
            lo, hi,
            hedges_g(c3_divs[:, j], c5_nulls[:, j]),
            j,
        ]
    return out, c3_divs, c5_nulls


def main():
    rng = np.random.default_rng(SEED)

    c3_dirs = sorted(DATA.glob(f"rep[0-9][0-9]_segreg_no_seed_{SCENARIO}_*"))
    c5_dirs = sorted(DATA.glob(f"rep[0-9][0-9]_no_seed_no_interact_{SCENARIO}_*"))
    print(f"C3 (Segreg+NoSeed) reps found: {len(c3_dirs)}")
    print(f"C5 (NoSeed+NoInteract) reps found: {len(c5_dirs)}")
    if not c3_dirs or not c5_dirs:
        print("ERROR: need at least one rep of each condition")
        sys.exit(1)

    print("\n--- C3 per-rep divergence (observed group A vs B) ---")
    c3_runs = []
    for d in c3_dirs:
        run = load_run(d)
        if run is None:
            print(f"  SKIP {d.name}: no parseable allocation"); continue
        div = divergence_per_dim(run["X"], run["groups"], 0, 1)
        if np.isnan(div).any():
            print(f"  SKIP {d.name}: missing group_id"); continue
        c3_runs.append(run)
        print(f"  {d.name}: " + "  ".join(f"{a[:4]}={v:5.2f}" for a, v in zip(AREAS, div)))

    print("\n--- C5 per-rep null divergence (mean over 1000 random 4-vs-4 splits) ---")
    c5_runs = []
    for d in c5_dirs:
        run = load_run(d)
        if run is None:
            print(f"  SKIP {d.name}: no parseable allocation"); continue
        null_mean = null_divergence_mean(run["X"], rng)
        c5_runs.append(run)
        print(f"  {d.name}: " + "  ".join(f"{a[:4]}={v:5.2f}" for a, v in zip(AREAS, null_mean)))

    print(f"\nReps loaded: C3={len(c3_runs)}, C5={len(c5_runs)}")
    print("Bootstrap: cluster-resampling reps, recomputing divergence from agent-level data per iteration")
    print(f"  ({N_BOOT} iterations, {N_PERM_NULL} perm-null per C5 rep point estimate, 100 per boot iter)")

    res, c3_per_rep, c5_per_rep = bootstrap_contrast(c3_runs, c5_runs, rng)

    print(f"\n=== Part-1 contrast: C3 (interaction-only) − C5 (baseline null), {SCENARIO.title()} ===")
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
        "n_c3_reps": len(c3_runs),
        "n_c5_reps": len(c5_runs),
        "bootstrap_method": "agent-stratified cluster bootstrap: resample reps with replacement, then within each sampled rep resample agents with replacement stratified by group (§4.3.4)",
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
                "hedges_g": None if math.isnan(res[i, 3]) else float(res[i, 3]),
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
