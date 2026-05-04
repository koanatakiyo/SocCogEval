"""Cross-family replication summary (§4.3.4 / §6.X).

For each model family — primary Qwen3-122B-FP8 plus 4 non-primary API
families (DeepSeek v3, GPT-4.1-mini, Gemini-2.5-flash, Llama 4 Scout via
OpenRouter) — compute the (2)−(5) BFI, conformity, and consistency
effects on the Greenfield scenario.

Why (2)−(5) and not (2)−(4)?
    Cross-model API runs only cover C2 (segreg+asym_seed) and C5
    (no_seed+no_interact). C4 (seed_only+no_interact) was not run on
    cross-model because of cost/quota; the §4.3.4 minimum requirement
    is "effect survives on at least one non-primary family", which the
    total-effect (2)−(5) contrast suffices for. Primary Qwen has the
    full factorial and is the load-bearing source for the (2)−(4)
    pure-interaction contrast in §4.3.3.

How to read the output
    For every model family, we report:
      Δ              = mean(C2 metric) − mean(C5 metric)  (per-agent pooled)
      Hedges' g       = standardized effect size (NaN when pooled SD < 0.05)
      95% CI          = agent-stratified cluster bootstrap
      Welch p         = unequal-variance two-sample t-test (parametric reference)
      ci_excludes_zero = the headline replication criterion

A model family REPLICATES the headline if its (2)−(5) BFI effect goes
in the same direction as primary Qwen and CI excludes zero. Conformity
and consistency are reported as supplementary.

Output:
    data/analysis_cross_model_<scenario>.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N_BOOT = 10_000
ALPHA = 0.05
HEDGES_G_SD_FLOOR = 0.05  # mirror analysis_part1_contrasts

BFI_FACTORS = ["extraversion", "agreeableness", "conscientiousness",
               "neuroticism", "openness"]

# (alias_used_in_rep_dirs, display_name)
# alias="" denotes primary Qwen runs (no model-suffix in rep dir name).
# qvl is Qwen-VL-32B-Instruct run in TEXT-ONLY mode — same Qwen lineage as
# the primary, different architecture (VL) and parameter count (32B vs 122B).
# Reported in §6.X as a within-family (cross-architecture) robustness check
# rather than a separate family.
MODELS = [
    ("",      "Qwen3-122B-FP8 (primary)"),
    ("qvl",   "Qwen3-VL-32B (text-only)"),
    ("dpsk",  "DeepSeek v3"),
    ("gpt",   "GPT-4.1 mini"),
    ("gem",   "Gemini 2.5 Flash"),
    ("l4api", "Llama 4 Scout (OpenRouter)"),
]

CONDITION_BASE = {
    "C2": "segreg_asym_seed",
    "C5": "no_seed_no_interact",
}


# ── stats helpers ───────────────────────────────────────────────────────────

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g, NaN when pooled SD is degenerate (matches analysis_part1_contrasts)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    s_pool = math.sqrt(((nx - 1) * np.var(x, ddof=1)
                        + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if s_pool < HEDGES_G_SD_FLOOR:
        return float("nan")
    d = (float(np.mean(x)) - float(np.mean(y))) / s_pool
    return d * (1 - 3 / (4 * (nx + ny - 2) - 1))


def cluster_bootstrap_ci(rep_groups_a: list[np.ndarray],
                         rep_groups_b: list[np.ndarray],
                         rng: np.random.Generator) -> tuple[float, float]:
    """Two-level bootstrap: resample reps, then resample agents within each
    sampled rep. Mirrors the agent-stratified strategy in
    analysis_part1_contrasts and analysis_p3_conformity_bfi."""
    if not rep_groups_a or not rep_groups_b:
        return (float("nan"), float("nan"))
    n_a, n_b = len(rep_groups_a), len(rep_groups_b)
    deltas = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sa = []
        for _ in range(n_a):
            i = rng.integers(0, n_a)
            agents = rep_groups_a[i]
            if len(agents) == 0: continue
            j_idx = rng.integers(0, len(agents), size=len(agents))
            sa.extend(agents[j_idx])
        sb = []
        for _ in range(n_b):
            i = rng.integers(0, n_b)
            agents = rep_groups_b[i]
            if len(agents) == 0: continue
            j_idx = rng.integers(0, len(agents), size=len(agents))
            sb.extend(agents[j_idx])
        if not sa or not sb:
            deltas[k] = float("nan")
        else:
            deltas[k] = float(np.mean(sa) - np.mean(sb))
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(deltas, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


# ── data extraction ─────────────────────────────────────────────────────────

def find_rep_dirs(alias: str, condition_base: str, scenario: str) -> list[Path]:
    """Find all rep dirs (with summary.json) for a given (alias, condition).

    Primary Qwen reps:    rep[0-9][0-9]_<base>_<scenario>_*
    Cross-model reps:     rep[0-9][0-9]_<alias>_<base>_<scenario>_*
    """
    if alias:
        pattern = f"rep[0-9][0-9]_{alias}_{condition_base}_{scenario}_*"
    else:
        pattern = f"rep[0-9][0-9]_{condition_base}_{scenario}_*"
    dirs = sorted(DATA.glob(pattern))
    return [d for d in dirs if (d / "summary.json").exists()]


def per_agent_bfi_l1(rep_dir: Path) -> np.ndarray:
    """Per-agent L1 drift = Σ_factor |post − pre| across 5 BFI-10 factors."""
    pre = json.load(open(rep_dir / "pre_measurements.json"))
    post = json.load(open(rep_dir / "post_measurements.json"))
    pre_by = {e["agent_name"]: e.get("scores", {}) for e in pre.get("bfi10", [])}
    post_by = {e["agent_name"]: e.get("scores", {}) for e in post.get("bfi10", [])}
    out = []
    for name in sorted(set(pre_by) & set(post_by)):
        p, q = pre_by[name], post_by[name]
        if not p or not q:
            continue
        out.append(sum(abs(float(q.get(f, 0)) - float(p.get(f, 0)))
                       for f in BFI_FACTORS))
    return np.asarray(out, dtype=float)


def per_agent_conformity_rate(rep_dir: Path) -> np.ndarray:
    """Post-only conformity rate per agent (between [0,1])."""
    post = json.load(open(rep_dir / "post_measurements.json"))
    return np.asarray(
        [float(c.get("conformity_rate", 0.0)) for c in post.get("conformity", [])],
        dtype=float,
    )


def per_agent_consistency_score(rep_dir: Path) -> np.ndarray:
    """Per-agent consistency_score across the 3 phrasings.

    The post_measurements.json schema stores `consistency_score` directly
    on each consistency entry. Higher = more consistent (less variance
    across rephrasings); the (2)−(5) direction we expect is positive
    (interaction makes the agent's position MORE internally stable).
    """
    post = json.load(open(rep_dir / "post_measurements.json"))
    out = []
    for c in post.get("consistency", []):
        v = c.get("consistency_score")
        if v is not None:
            out.append(float(v))
    return np.asarray(out, dtype=float)


INSTRUMENTS = [
    ("bfi_l1",        "BFI-10 L1 drift",       per_agent_bfi_l1),
    ("conformity",    "Conformity rate",       per_agent_conformity_rate),
    ("consistency",   "Consistency variance",  per_agent_consistency_score),
]


# ── per-model contrast ──────────────────────────────────────────────────────

def analyze_one_model(alias: str, name: str, scenario: str,
                      rng_master_seed: int) -> dict:
    print(f"\n──── {name} (alias={alias!r}) ────")
    out = {"alias": alias, "name": name, "instruments": {}}

    c2_dirs = find_rep_dirs(alias, CONDITION_BASE["C2"], scenario)
    c5_dirs = find_rep_dirs(alias, CONDITION_BASE["C5"], scenario)
    out["n_c2_reps"] = len(c2_dirs)
    out["n_c5_reps"] = len(c5_dirs)
    print(f"  C2 reps: {len(c2_dirs)}   C5 reps: {len(c5_dirs)}")

    if not c2_dirs or not c5_dirs:
        print("  insufficient data — skipping all instruments")
        for key, _label, _fn in INSTRUMENTS:
            out["instruments"][key] = None
        return out

    for key, label, extract in INSTRUMENTS:
        c2_groups = [extract(d) for d in c2_dirs]
        c5_groups = [extract(d) for d in c5_dirs]
        c2_groups = [g for g in c2_groups if g.size > 0]
        c5_groups = [g for g in c5_groups if g.size > 0]
        if not c2_groups or not c5_groups:
            print(f"  {label:24s}  ✗ empty extraction")
            out["instruments"][key] = None
            continue
        c2_pool = np.concatenate(c2_groups)
        c5_pool = np.concatenate(c5_groups)
        delta = float(c2_pool.mean() - c5_pool.mean())
        g = hedges_g(c2_pool, c5_pool)
        t, p = stats.ttest_ind(c2_pool, c5_pool, equal_var=False)
        rng = np.random.default_rng(rng_master_seed)
        lo, hi = cluster_bootstrap_ci(c2_groups, c5_groups, rng)
        excl = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        g_ser = None if math.isnan(g) else float(g)
        g_str = f"{g:+.2f}" if not math.isnan(g) else "undef"
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]" if not math.isnan(lo) else "[NaN]"
        print(f"  {label:24s}  Δ={delta:+7.3f}  g={g_str:>6}  CI={ci_str:>22}  "
              f"p={float(p):.4f}  n={c2_pool.size}/{c5_pool.size}  {'✓' if excl else ' '}")
        out["instruments"][key] = {
            "delta": delta,
            "hedges_g": g_ser,
            "ci95_lo": float(lo) if not math.isnan(lo) else None,
            "ci95_hi": float(hi) if not math.isnan(hi) else None,
            "welch_t": float(t) if not math.isnan(t) else None,
            "welch_p": float(p) if not math.isnan(p) else None,
            "n_c2_agents": int(c2_pool.size),
            "n_c5_agents": int(c5_pool.size),
            "ci_excludes_zero": bool(excl),
        }
    return out


# ── headline replication table ──────────────────────────────────────────────

def print_headline_table(results: list[dict], primary_alias: str = "") -> None:
    print()
    print("=" * 110)
    print("  HEADLINE: BFI-10 (2)−(5) cross-family replication")
    print("=" * 110)
    print(f"  {'Model':<32}  {'C2 reps':>7}  {'C5 reps':>7}  "
          f"{'Δ̄':>8}  {'g':>8}  {'95% CI':>22}  {'p':>8}  {'sig':>5}  {'matches primary':>16}")
    print("  " + "─" * 108)

    primary_dir = None
    for r in results:
        if r["alias"] == primary_alias:
            bfi = r["instruments"].get("bfi_l1")
            if bfi:
                primary_dir = +1 if bfi["delta"] > 0 else -1
            break

    for r in results:
        bfi = r["instruments"].get("bfi_l1")
        if bfi is None:
            print(f"  {r['name']:<32}  {r['n_c2_reps']:>7d}  {r['n_c5_reps']:>7d}  "
                  f"{'(insufficient data)':>67}")
            continue
        g = bfi["hedges_g"]
        g_str = f"{g:+.2f}" if g is not None else "undef"
        lo, hi = bfi["ci95_lo"], bfi["ci95_hi"]
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None and hi is not None else "[NaN]"
        sig = "✓" if bfi["ci_excludes_zero"] else " "
        match = ""
        if r["alias"] != primary_alias and primary_dir is not None:
            this_dir = +1 if bfi["delta"] > 0 else -1
            match = "✓ same dir" if this_dir == primary_dir else "✗ opposite"
        elif r["alias"] == primary_alias:
            match = "(reference)"
        print(f"  {r['name']:<32}  {r['n_c2_reps']:>7d}  {r['n_c5_reps']:>7d}  "
              f"{bfi['delta']:>+8.3f}  {g_str:>8}  {ci_str:>22}  "
              f"{bfi['welch_p']:>8.4f}  {sig:>5}  {match:>16}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="greenfield",
                    choices=["greenfield", "riverton"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 110)
    print(f"  Cross-family replication summary  ({args.scenario.upper()})  "
          f"— C2 vs C5  (n_bootstrap={N_BOOT})")
    print("=" * 110)

    results = [analyze_one_model(alias, name, args.scenario, SEED)
               for alias, name in MODELS]

    print_headline_table(results, primary_alias="")

    out_path = Path(args.out) if args.out else (
        DATA / f"analysis_cross_model_{args.scenario}.json")
    payload = {
        "scenario": args.scenario,
        "n_bootstrap": N_BOOT,
        "alpha": ALPHA,
        "hedges_g_sd_floor": HEDGES_G_SD_FLOOR,
        "rng_seed": SEED,
        "instruments": {k: label for k, label, _ in INSTRUMENTS},
        "models": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
