"""analysis_part1_contrasts.py — Part 1 allocation-divergence contrasts (§4.3.1 / §4.3.3).

Extends analysis_c3_vs_c5.py from a single contrast (C3 vs C5) to the set of
Part-1 contrasts that are computable from current data:

    (2) − (5)   C2 (seed × interaction)  vs  C5 (baseline null)    → total effect
    (3) − (5)   C3 (interaction alone)   vs  C5 (baseline null)    → interaction-alone (C3 critical ablation)
    (2) − (3)   C2                       vs  C3                    → incremental seed × interaction effect

Contrasts that require conditions not yet run:
    (4) − (5)   pure seed          → needs C4 (launching in background)
    (2) − (4)   interaction | seed → needs C4
    (2) − (1)   segregation effect → needs C1 (launching in background)

Statistical protocol (§4.3.4):
    Point estimate: per-dim |mean(group A) − mean(group B)| for segregated reps;
                    null-permutation mean for C5 (no real groups).
    Inference:      agent-stratified cluster bootstrap, 10,000 iterations.
                    Per iter: resample reps → within each sampled rep resample
                    agents with replacement separately in group A and group B.
    Effect size:    Hedges' g on per-rep distributions.
    Output:         per-dimension 95% CI on the contrast (C_i - C_j) per area,
                    plus overall mean across dimensions.

Usage:
    python scripts/analysis_part1_contrasts.py                  # Greenfield
    python scripts/analysis_part1_contrasts.py --scenario riverton
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DATA = Path("/home/yandan/cogsociety/data")
N_BOOT = 10_000
N_PERM_NULL = 1_000
ALPHA = 0.05
SEED = 42

AREAS = {
    "greenfield": ["Education", "Healthcare", "Green Spaces", "Infrastructure", "Economic Development"],
    "riverton":   ["Automation", "Workforce Training", "Employee Support", "Innovation Lab", "Data Infrastructure"],
}

# Condition-name → glob pattern for finding rep directories
COND_GLOB = {
    "C1": "rep[0-9][0-9]_open_seed_{scenario}_*",
    "C2": "rep[0-9][0-9]_segreg_asym_seed_{scenario}_*",
    "C3": "rep[0-9][0-9]_segreg_no_seed_{scenario}_*",
    "C4": "rep[0-9][0-9]_seed_only_no_interact_{scenario}_*",
    "C5": "rep[0-9][0-9]_no_seed_no_interact_{scenario}_*",
}


def load_run(run_dir: Path, areas: list[str]):
    """Return dict with names, groups, X (n_agents × n_areas), has_real_groups — or None.

    Group assignment priority:
      1. If network.topology == 'segregated': use group_id from summary (network groups).
      2. Else if seed.asymmetric == True: virtual groups by agent index
         (agents 0..n/2-1 got SEED_A, n/2..n-1 got SEED_B — see run_experiment.create_agents).
      3. Else: no groups (single-group condition, will use random-split null).

    `has_real_groups=True` iff path 1 or 2 applied.
    """
    try:
        summary = json.load(open(run_dir / "summary.json"))
        post = json.load(open(run_dir / "post_measurements.json"))
        cfg = json.load(open(run_dir / "config.json"))
    except FileNotFoundError:
        return None

    topology = cfg.get("network", {}).get("topology")
    seed_asym = cfg.get("seed", {}).get("asymmetric", False)

    agents_in_order = summary["agents"]
    n = len(agents_in_order)
    half = n // 2

    if topology == "segregated":
        name_to_group = {a["name"]: a.get("group_id") for a in agents_in_order}
        has_real_groups = True
    elif seed_asym:
        # Virtual group assignment by agent index (matches run_experiment.create_agents:
        # `seed_info = SEED_A if i < n_agents // 2 else SEED_B`)
        name_to_group = {
            a["name"]: (0 if i < half else 1)
            for i, a in enumerate(agents_in_order)
        }
        has_real_groups = True
    else:
        name_to_group = {a["name"]: None for a in agents_in_order}
        has_real_groups = False

    rows, groups, names = [], [], []
    for entry in post.get("allocation", []) or []:
        n_name = entry.get("agent_name")
        a = entry.get("allocations") or {}
        if not (isinstance(a, dict) and n_name):
            continue
        rows.append([float(a.get(area, 0.0)) for area in areas])
        groups.append(name_to_group.get(n_name))
        names.append(n_name)
    if not rows:
        return None
    return {
        "names": names,
        "groups": np.array(groups, dtype=object),
        "X": np.asarray(rows, dtype=float),
        "has_real_groups": has_real_groups,
        "topology": topology,
        "seed_asymmetric": seed_asym,
    }


def divergence_segregated(run: dict, a_id=0, b_id=1) -> np.ndarray:
    """|mean(group A) − mean(group B)| per dimension. For segregated reps."""
    g = run["groups"]
    X = run["X"]
    a_mask = np.array([x == a_id for x in g])
    b_mask = np.array([x == b_id for x in g])
    if a_mask.sum() == 0 or b_mask.sum() == 0:
        return np.full(X.shape[1], np.nan)
    return np.abs(X[a_mask].mean(0) - X[b_mask].mean(0))


def null_divergence_mean(X: np.ndarray, rng: np.random.Generator, n_perm=N_PERM_NULL) -> np.ndarray:
    """Expected |mean_A − mean_B| under random half-vs-half splits (for single-group conditions)."""
    n = X.shape[0]
    half = n // 2
    accum = np.zeros(X.shape[1])
    for _ in range(n_perm):
        perm = rng.permutation(n)
        accum += np.abs(X[perm[:half]].mean(0) - X[perm[half : 2 * half]].mean(0))
    return accum / n_perm


def rep_divergence(run: dict, has_groups: bool, rng: np.random.Generator) -> np.ndarray:
    """Point estimate of per-dim divergence for one rep.
    Segregated condition → group A vs B.  Single-group condition → permutation null."""
    if has_groups:
        return divergence_segregated(run)
    return null_divergence_mean(run["X"], rng)


# Pooled-SD floor below which Hedges' g is treated as undefined.
# Rationale: when one or both groups are near-constant (e.g., C5 outputs the
# same allocation across all reps because greedy decoding is deterministic),
# pooled SD collapses to 0 or near-0, and g = Δ / SD blows up to a meaningless
# value (we observed g = -76.9 from a delta of -1.6 with SD ≈ 0.02). Use NaN
# in those cases so the aggregator can skip them rather than averaging an
# artifact alongside legitimate effect sizes.
HEDGES_G_SD_FLOOR = 0.05


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    s_pool = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pool < HEDGES_G_SD_FLOOR:
        return float("nan")
    d = (mx - my) / s_pool
    df = nx + ny - 2
    return d * (1 - 3 / (4 * df - 1))


def sample_rep_divergence(run: dict, has_groups: bool, rng: np.random.Generator) -> np.ndarray:
    """Single-iteration bootstrap draw of per-dim divergence for one rep.
    Segregated: resample agents within each group separately (stratified).
    Single-group: random half-vs-half split + resample within each half."""
    X = run["X"]
    if has_groups:
        a_mask = np.array([g == 0 for g in run["groups"]])
        b_mask = np.array([g == 1 for g in run["groups"]])
        Xa, Xb = X[a_mask], X[b_mask]
        na, nb = len(Xa), len(Xb)
        return np.abs(
            Xa[rng.integers(0, na, na)].mean(0)
            - Xb[rng.integers(0, nb, nb)].mean(0)
        )
    # Single-group: one random split per iter + resample within
    n = X.shape[0]
    half = n // 2
    perm = rng.permutation(n)
    a_idx, b_idx = perm[:half], perm[half : 2 * half]
    return np.abs(
        X[a_idx][rng.integers(0, half, half)].mean(0)
        - X[b_idx][rng.integers(0, half, half)].mean(0)
    )


def bootstrap_contrast(
    cond_a_runs: list[dict], cond_b_runs: list[dict],
    a_has_groups: bool, b_has_groups: bool,
    rng: np.random.Generator,
):
    """Agent-stratified cluster bootstrap on (cond_a − cond_b) per dimension.
    Returns: (n_areas × 5) array [delta, ci_lo, ci_hi, hedges_g, dim_idx]
             plus per-rep point estimates for both conditions (n_reps × n_areas)."""
    n_areas = cond_a_runs[0]["X"].shape[1]
    a_divs = np.vstack([rep_divergence(r, a_has_groups, rng) for r in cond_a_runs])
    b_divs = np.vstack([rep_divergence(r, b_has_groups, rng) for r in cond_b_runs])

    na, nb = len(cond_a_runs), len(cond_b_runs)
    diffs = np.empty((N_BOOT, n_areas))
    for k in range(N_BOOT):
        a_sample = rng.integers(0, na, na)
        a_vals = np.vstack([
            sample_rep_divergence(cond_a_runs[i], a_has_groups, rng)
            for i in a_sample
        ])
        b_sample = rng.integers(0, nb, nb)
        b_vals = np.vstack([
            sample_rep_divergence(cond_b_runs[i], b_has_groups, rng)
            for i in b_sample
        ])
        diffs[k] = a_vals.mean(0) - b_vals.mean(0)

    out = np.empty((n_areas, 5))
    for j in range(n_areas):
        lo, hi = np.percentile(diffs[:, j], [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out[j] = [
            a_divs[:, j].mean() - b_divs[:, j].mean(),
            lo, hi,
            hedges_g(a_divs[:, j], b_divs[:, j]),
            j,
        ]
    return out, a_divs, b_divs


def load_condition(cond_key: str, scenario: str, areas: list[str]) -> list[dict]:
    pattern = COND_GLOB[cond_key].format(scenario=scenario)
    dirs = sorted(DATA.glob(pattern))
    runs = []
    for d in dirs:
        r = load_run(d, areas)
        if r is None:
            print(f"  SKIP {d.name}: parse failed")
            continue
        runs.append(r)
    return runs


def run_contrast(label: str, runs_a: list[dict], runs_b: list[dict],
                 a_groups: bool, b_groups: bool, areas: list[str],
                 rng: np.random.Generator) -> dict:
    print(f"\n{'─' * 88}")
    print(f" Contrast: {label}   (n_A={len(runs_a)}, n_B={len(runs_b)})")
    print("─" * 88)
    if not runs_a or not runs_b:
        print("  SKIP: insufficient data")
        return None
    res, a_divs, b_divs = bootstrap_contrast(runs_a, runs_b, a_groups, b_groups, rng)
    header = f"{'Area':<22}{'A mean':>8}{'B mean':>8}{'Δ':>8}{'95% CI':>22}{'Hedges g':>10}  sig"
    print(header)
    print("-" * len(header))
    n_sig = 0
    rows = []
    for i, area in enumerate(areas):
        delta, lo, hi, g, _ = res[i]
        excl = lo > 0 or hi < 0
        if excl: n_sig += 1
        # NaN g (degenerate pooled SD) → store as None and print as "undef" so
        # downstream JSON parsers don't choke on NaN literals.
        g_undef = math.isnan(float(g))
        g_str = "    undef" if g_undef else f"{g:>+8.2f}"
        print(f"{area:<22}{a_divs[:,i].mean():>8.2f}{b_divs[:,i].mean():>8.2f}{delta:>+8.2f}"
              f"   [{lo:+5.2f}, {hi:+5.2f}]   {g_str}  {'✓' if excl else ' '}")
        rows.append({"area": area, "a_mean": float(a_divs[:, i].mean()),
                     "b_mean": float(b_divs[:, i].mean()), "delta": float(delta),
                     "ci95_lo": float(lo), "ci95_hi": float(hi),
                     "hedges_g": None if g_undef else float(g),
                     "ci_excludes_zero": bool(excl)})
    print(f"  {n_sig}/{len(areas)} dimensions with CI excluding zero")
    return {
        "label": label,
        "n_a_reps": len(runs_a),
        "n_b_reps": len(runs_b),
        "n_significant_dims": n_sig,
        "results": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["greenfield", "riverton"], default="greenfield")
    args = ap.parse_args()
    scenario = args.scenario
    areas = AREAS[scenario]
    rng = np.random.default_rng(SEED)

    print("=" * 88)
    print(f" Part-1 allocation-divergence contrasts — scenario: {scenario.upper()}")
    print(f" Bootstrap: agent-stratified cluster, {N_BOOT} iters, α={ALPHA}")
    print("=" * 88)

    # Load all available conditions. has_groups is determined per-run from the
    # loaded config (not hardcoded by condition key) so C4 (asymmetric seed, no
    # topology) correctly gets seed-based virtual groups rather than a
    # random-split null.
    cond_runs = {}
    cond_has_groups = {}
    for cond in ["C1", "C2", "C3", "C4", "C5"]:
        runs = load_condition(cond, scenario, areas)
        cond_runs[cond] = runs
        cond_has_groups[cond] = runs[0]["has_real_groups"] if runs else False
        tag = "seed-groups" if cond_has_groups[cond] and cond in ("C1", "C4") else (
              "topology-groups" if cond_has_groups[cond] else "single-group (null)")
        print(f"  {cond}: {len(runs)} reps loaded  [{tag}]")

    all_contrasts = {}

    # Always-computable contrasts
    for label, a, b in [
        ("C2 − C5  (total effect: seed × interaction)", "C2", "C5"),
        ("C3 − C5  (interaction alone — C3 critical ablation)", "C3", "C5"),
        ("C2 − C3  (incremental effect of seed given interaction)", "C2", "C3"),
    ]:
        if cond_runs[a] and cond_runs[b]:
            c = run_contrast(label,
                             cond_runs[a], cond_runs[b],
                             cond_has_groups[a], cond_has_groups[b],
                             areas, rng)
            if c: all_contrasts[label.split("  ")[0]] = c
        else:
            print(f"\n[SKIP] {label} — need {a}={len(cond_runs[a])} and {b}={len(cond_runs[b])} reps")

    # Optional contrasts (require C4 and/or C1)
    for label, a, b in [
        ("C4 − C5  (pure seed effect)", "C4", "C5"),
        ("C2 − C4  (interaction effect given seed)", "C2", "C4"),
        ("C2 − C1  (segregation effect)", "C2", "C1"),
    ]:
        if cond_runs[a] and cond_runs[b]:
            c = run_contrast(label,
                             cond_runs[a], cond_runs[b],
                             cond_has_groups[a], cond_has_groups[b],
                             areas, rng)
            if c: all_contrasts[label.split("  ")[0]] = c
        else:
            print(f"\n[SKIP] {label} — waiting for {a}={len(cond_runs[a])}, {b}={len(cond_runs[b])} reps")

    # Write JSON
    out_path = DATA / f"analysis_part1_contrasts_{scenario}.json"
    out = {
        "scenario": scenario,
        "areas": areas,
        "n_bootstrap": N_BOOT,
        "alpha": ALPHA,
        "bootstrap_method": "agent-stratified cluster bootstrap (§4.3.4): resample reps, within each rep resample agents stratified by group",
        "n_reps_per_condition": {k: len(v) for k, v in cond_runs.items()},
        "contrasts": all_contrasts,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{'=' * 88}\nSaved: {out_path}")


if __name__ == "__main__":
    main()
