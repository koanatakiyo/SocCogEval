"""analysis_p3_conformity_bfi.py — Part 2 cross-task transfer metrics (§4.3.2).

Implements the statistics required by Chapter 4 §4.3.2:

  CONFORMITY (§4.3.2 line 120):
    "Within-agent, within-item paired design maps onto McNemar's test per
     agent, aggregated via mixed-effects logistic regression with agent
     as random intercept."
    - Per-agent McNemar's test on the 15 per-item paired outcomes:
          (pre-switched binary)  vs  (post-switched binary)
    - Discordant-pair exact binomial test (aka exact McNemar).
    - Aggregated across agents via Fisher's combined probability test
      (approximates mixed-effects logistic without statsmodels).
    - If statsmodels is installed, also runs the proper mixed-effects
      logistic regression: switch ~ phase + (1|agent).
    - Cross-condition contrast on per-agent conformity-rate delta, with
      Welch t-test + 95% bootstrap CI + Hedges' g (retained for P4 integration).

  BFI-10 DRIFT (§4.3.2 line 122):
    "For each of the 5 factors, paired pre/post delta per agent."
    - Reads the `scores` dict (extraversion, agreeableness, etc.) from
      pre/post measurements — already factor-scored by the probe pipeline.
    - Per-factor paired Wilcoxon within each condition.
    - Per-factor Welch t + 95% bootstrap CI between conditions.
    - Reports aggregate L1 drift as an additional descriptive column.

  CROSS-CONDITION CONTRAST KEYS:
    Stored as "C{i}-C{j}" (ASCII hyphen, no spaces) to match
    analysis_p4_causal_decomposition.py's lookup format.

Usage:
    python scripts/analysis_p3_conformity_bfi.py --scenario greenfield
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import pandas as pd
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

DATA = Path("/home/yandan/cogsociety/data")
COND_GLOB = {
    "C1": "rep[0-9][0-9]_open_seed_{scenario}_*",
    "C2": "rep[0-9][0-9]_segreg_asym_seed_{scenario}_*",
    "C3": "rep[0-9][0-9]_segreg_no_seed_{scenario}_*",
    "C4": "rep[0-9][0-9]_seed_only_no_interact_{scenario}_*",
    "C5": "rep[0-9][0-9]_no_seed_no_interact_{scenario}_*",
}
N_BOOT = 10_000
ALPHA = 0.05
SEED = 42

BFI_FACTORS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_run_measurements(run_dir: Path):
    try:
        pre = json.load(open(run_dir / "pre_measurements.json"))
        post = json.load(open(run_dir / "post_measurements.json"))
    except FileNotFoundError:
        return None
    return {
        "pre_conf":  {e["agent_name"]: e for e in pre.get("conformity", [])},
        "post_conf": {e["agent_name"]: e for e in post.get("conformity", [])},
        "pre_bfi":   {e["agent_name"]: e for e in pre.get("bfi10", [])},
        "post_bfi":  {e["agent_name"]: e for e in post.get("bfi10", [])},
        # Consistency is post-only (§4.3.2 line 124). We use the probe-computed
        # consistency_score (higher = more consistent). Missing on pre-battery.
        "post_cons": {e["agent_name"]: e for e in post.get("consistency", [])},
    }


def load_condition(cond_key: str, scenario: str):
    pattern = COND_GLOB[cond_key].format(scenario=scenario)
    dirs = sorted(DATA.glob(pattern))
    runs = []
    for d in dirs:
        m = load_run_measurements(d)
        if m is None: continue
        runs.append({"dir": d.name, **m})
    return runs


# ═══════════════════════════════════════════════════════════════════════════
# Conformity — McNemar per agent + Fisher-combined + mixed-effects (optional)
# ═══════════════════════════════════════════════════════════════════════════

def per_item_switched(conf_record: dict) -> list[int]:
    """From one pre or post conformity record, return per-item `switched` as 0/1 list."""
    return [int(bool(r.get("switched"))) for r in conf_record.get("responses", [])]


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Exact McNemar. `b` and `c` are the discordant-pair counts.
    Returns (statistic, two-sided p-value)."""
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    # Exact binomial two-sided: sum of P(X=k) for k s.t. P(X=k) <= P(X=min(b,c))
    p = stats.binomtest(min(b, c), n, p=0.5).pvalue
    return (float(min(b, c)), float(p))


def fishers_combined(pvalues: list[float]) -> tuple[float, float]:
    """Fisher's combined probability test for aggregating k independent p-values."""
    pvalues = [p for p in pvalues if p is not None and 0 < p <= 1]
    if not pvalues:
        return (0.0, 1.0)
    chi2 = -2 * sum(math.log(p) for p in pvalues)
    df = 2 * len(pvalues)
    return (float(chi2), float(stats.chi2.sf(chi2, df)))


def analyze_conformity(runs: list[dict], label: str) -> dict:
    """Return dict with per-agent McNemar p-values, Fisher-combined p,
    per-agent conformity-rate deltas, and (optionally) mixed-effects result."""
    per_agent_rows = []
    mcnemar_ps = []
    rate_deltas = []
    # For mixed-effects: long-format (agent, phase, switched, item, rep)
    long_rows = []
    for r in runs:
        names = set(r["pre_conf"].keys()) & set(r["post_conf"].keys())
        for name in sorted(names):
            pre_items = per_item_switched(r["pre_conf"][name])
            post_items = per_item_switched(r["post_conf"][name])
            if len(pre_items) != len(post_items) or not pre_items:
                continue
            # McNemar: count discordant pairs
            b = sum(1 for x, y in zip(pre_items, post_items) if x == 0 and y == 1)  # new switches
            c = sum(1 for x, y in zip(pre_items, post_items) if x == 1 and y == 0)  # lost switches
            stat, p = mcnemar_exact(b, c)
            pre_rate = r["pre_conf"][name].get("conformity_rate", float("nan"))
            post_rate = r["post_conf"][name].get("conformity_rate", float("nan"))
            rate_deltas.append(post_rate - pre_rate)
            per_agent_rows.append({
                "rep": r["dir"], "agent": name,
                "mcnemar_b_new_switch": b, "mcnemar_c_lost_switch": c,
                "mcnemar_stat": stat, "mcnemar_p": p,
                "pre_rate": float(pre_rate), "post_rate": float(post_rate),
                "rate_delta": float(post_rate - pre_rate),
            })
            mcnemar_ps.append(p)
            # Long format for mixed-effects
            for i, s in enumerate(pre_items):
                long_rows.append({"agent": f"{r['dir']}:{name}", "phase": 0,
                                   "item": i, "switched": s, "rep": r["dir"]})
            for i, s in enumerate(post_items):
                long_rows.append({"agent": f"{r['dir']}:{name}", "phase": 1,
                                   "item": i, "switched": s, "rep": r["dir"]})
    # Aggregate McNemar via Fisher's combined test
    chi2, fisher_p = fishers_combined(mcnemar_ps)
    # Optional: mixed-effects logistic regression
    mixed_result = None
    if _HAS_STATSMODELS and long_rows:
        try:
            df = pd.DataFrame(long_rows)
            # switched ~ phase  with agent as random intercept
            md = smf.mixedlm("switched ~ phase", df, groups=df["agent"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = md.fit(method=["lbfgs"], reml=False)
            mixed_result = {
                "phase_beta": float(res.params.get("phase", float("nan"))),
                "phase_se": float(res.bse.get("phase", float("nan"))),
                "phase_p": float(res.pvalues.get("phase", float("nan"))),
                "n_obs": int(len(df)),
                "n_agents": int(df["agent"].nunique()),
            }
        except Exception as e:
            mixed_result = {"error": str(e)}
    rate_deltas = np.asarray(rate_deltas, dtype=float)
    return {
        "label": label,
        "n_agents": len(per_agent_rows),
        "per_agent": per_agent_rows,
        "fishers_combined": {"chi2": chi2, "p": fisher_p, "df": 2 * len(mcnemar_ps)},
        "mixed_effects_logistic": mixed_result,
        "rate_deltas": rate_deltas.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BFI-10 — per-factor paired delta per agent (§4.3.2 line 122)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_bfi(runs: list[dict], label: str) -> dict:
    """Per-factor paired delta. Returns per-agent rows and per-factor summaries."""
    per_agent_rows = []
    per_factor_deltas = {f: [] for f in BFI_FACTORS}
    l1_total = []
    for r in runs:
        names = set(r["pre_bfi"].keys()) & set(r["post_bfi"].keys())
        for name in sorted(names):
            pre = r["pre_bfi"][name].get("scores") or {}
            post = r["post_bfi"][name].get("scores") or {}
            if not pre or not post:
                continue
            deltas = {f: float(post.get(f, float("nan"))) - float(pre.get(f, float("nan")))
                      for f in BFI_FACTORS}
            for f in BFI_FACTORS:
                per_factor_deltas[f].append(deltas[f])
            l1 = sum(abs(deltas[f]) for f in BFI_FACTORS)
            l1_total.append(l1)
            per_agent_rows.append({
                "rep": r["dir"], "agent": name,
                **{f"delta_{f}": deltas[f] for f in BFI_FACTORS},
                "l1_drift_across_factors": l1,
            })

    factor_summary = {}
    for f in BFI_FACTORS:
        arr = np.asarray(per_factor_deltas[f], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size >= 2:
            try:
                _, p = stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p = 1.0
            factor_summary[f] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)),
                "wilcoxon_p": float(p),
            }
        else:
            factor_summary[f] = {"n": int(arr.size)}
    return {
        "label": label,
        "n_agents": len(per_agent_rows),
        "per_agent": per_agent_rows,
        "per_factor": factor_summary,
        "per_factor_deltas": {f: per_factor_deltas[f] for f in BFI_FACTORS},
        "l1_drift": l1_total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Contrasts: between-condition Welch + bootstrap CI + Hedges' g
# ═══════════════════════════════════════════════════════════════════════════

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    # NaN when pooled SD is degenerate — see analysis_part1_contrasts for rationale.
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return float("nan")
    s_pool = math.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if s_pool < 0.05: return float("nan")
    return ((float(np.mean(x)) - float(np.mean(y))) / s_pool) * (1 - 3 / (4 * (nx + ny - 2) - 1))


def bootstrap_ci_diff_of_means(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot=N_BOOT):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (float("nan"), float("nan"))
    diffs = np.array([a[rng.integers(0, na, na)].mean() - b[rng.integers(0, nb, nb)].mean()
                       for _ in range(n_boot)])
    return tuple(np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))


def contrast(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict | None:
    if len(a) < 2 or len(b) < 2:
        return None
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    delta = float(a.mean() - b.mean())
    lo, hi = bootstrap_ci_diff_of_means(a, b, rng)
    g = hedges_g(a, b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    excl = lo > 0 or hi < 0
    return {
        "delta": delta,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "hedges_g": None if math.isnan(g) else float(g),
        "welch_t": float(t),
        "welch_p": float(p),
        "ci_excludes_zero": bool(excl),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["greenfield", "riverton"], default="greenfield")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    print("=" * 88)
    print(f" Part-2 transfer metrics — scenario: {args.scenario.upper()}")
    print(f" statsmodels available: {_HAS_STATSMODELS}")
    print("=" * 88)

    conds = {k: load_condition(k, args.scenario) for k in ["C1", "C2", "C3", "C4", "C5"]}
    for k, runs in conds.items():
        print(f"  {k}: {len(runs)} reps")

    # ── Conformity (§4.3.2 McNemar + mixed-effects) ─────────────────────────
    print("\n" + "━" * 88)
    print(" A.  CONFORMITY  —  per-agent McNemar + Fisher-combined + optional mixed-effects")
    print("━" * 88)
    conf_by_cond = {}
    for cond_key, runs in conds.items():
        if not runs: continue
        conf_by_cond[cond_key] = analyze_conformity(runs, cond_key)
        a = conf_by_cond[cond_key]
        fc = a["fishers_combined"]
        print(f"\n  {cond_key}:  n_agents={a['n_agents']}  Fisher χ²={fc['chi2']:.1f} "
              f"df={fc['df']}  p={fc['p']:.4f}")
        if a["mixed_effects_logistic"]:
            me = a["mixed_effects_logistic"]
            if "error" in me:
                print(f"    mixed-effects:  error — {me['error']}")
            else:
                print(f"    mixed-effects β_phase={me['phase_beta']:+.3f} "
                      f"SE={me['phase_se']:.3f}  p={me['phase_p']:.4f}  (n_obs={me['n_obs']})")

    # ── BFI-10 per-factor (§4.3.2 line 122) ─────────────────────────────────
    print("\n" + "━" * 88)
    print(" B.  BFI-10 PER-FACTOR PAIRED DELTAS")
    print("━" * 88)
    bfi_by_cond = {}
    for cond_key, runs in conds.items():
        if not runs: continue
        bfi_by_cond[cond_key] = analyze_bfi(runs, cond_key)
        b = bfi_by_cond[cond_key]
        print(f"\n  {cond_key}:  n_agents={b['n_agents']}")
        for f in BFI_FACTORS:
            s = b["per_factor"].get(f, {})
            if "mean" in s:
                print(f"    {f:18s}  mean Δ={s['mean']:+.3f}  sd={s['sd']:.3f}  "
                      f"Wilcoxon p={s['wilcoxon_p']:.4f}  (n={s['n']})")

    # ── Between-condition contrasts (ASCII key format for P4 integration) ───
    print("\n" + "━" * 88)
    print(" C.  BETWEEN-CONDITION CONTRASTS  (key format: 'Ca-Cb' for P4 integration)")
    print("━" * 88)
    contrast_pairs = [("C2", "C5"), ("C3", "C5"), ("C2", "C3"),
                      ("C2", "C4"), ("C4", "C5"), ("C2", "C1")]

    between_conf = {}
    print("\n  Conformity rate delta (post − pre):")
    for a_k, b_k in contrast_pairs:
        if a_k not in conf_by_cond or b_k not in conf_by_cond: continue
        a = np.asarray(conf_by_cond[a_k]["rate_deltas"])
        b = np.asarray(conf_by_cond[b_k]["rate_deltas"])
        res = contrast(a, b, rng)
        if res:
            between_conf[f"{a_k}-{b_k}"] = res
            mark = "✓" if res["ci_excludes_zero"] else " "
            print(f"    {a_k}-{b_k}:  Δ={res['delta']:+.3f}  CI=[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]  "
                  f"g={res['hedges_g']:+.2f}  Welch p={res['welch_p']:.4f}  {mark}")

    # BFI — contrast on L1 drift (for P4 integration + table summary)
    between_bfi = {}
    print("\n  BFI-10 L1 drift (aggregate across factors):")
    for a_k, b_k in contrast_pairs:
        if a_k not in bfi_by_cond or b_k not in bfi_by_cond: continue
        a = np.asarray(bfi_by_cond[a_k]["l1_drift"])
        b = np.asarray(bfi_by_cond[b_k]["l1_drift"])
        res = contrast(a, b, rng)
        if res:
            between_bfi[f"{a_k}-{b_k}"] = res
            mark = "✓" if res["ci_excludes_zero"] else " "
            print(f"    {a_k}-{b_k}:  Δ={res['delta']:+.3f}  CI=[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]  "
                  f"g={res['hedges_g']:+.2f}  Welch p={res['welch_p']:.4f}  {mark}")

    # Per-factor BFI contrasts (detail for paper appendix)
    between_bfi_per_factor = {}
    print("\n  BFI-10 per-factor contrasts:")
    for a_k, b_k in contrast_pairs:
        if a_k not in bfi_by_cond or b_k not in bfi_by_cond: continue
        line = f"    {a_k}-{b_k}:  "
        factor_rows = {}
        for f in BFI_FACTORS:
            a = np.asarray(bfi_by_cond[a_k]["per_factor_deltas"][f])
            b = np.asarray(bfi_by_cond[b_k]["per_factor_deltas"][f])
            res = contrast(a, b, rng)
            if res:
                factor_rows[f] = res
                mark = "*" if res["ci_excludes_zero"] else "·"
                line += f"{f[:4]}{mark}={res['delta']:+.2f}  "
        between_bfi_per_factor[f"{a_k}-{b_k}"] = factor_rows
        print(line)

    # ── Consistency (§4.3.2 line 124) ────────────────────────────────────────
    print("\n" + "━" * 88)
    print(" D.  CROSS-PROMPT CONSISTENCY  (probe-computed consistency_score per agent, post-only)")
    print("━" * 88)
    cons_by_cond = {}
    for cond_key, runs in conds.items():
        if not runs: continue
        scores = []
        per_agent = []
        for r in runs:
            for name, rec in r.get("post_cons", {}).items():
                s = rec.get("consistency_score")
                if s is None: continue
                scores.append(float(s))
                per_agent.append({"rep": r["dir"], "agent": name, "score": float(s)})
        cons_by_cond[cond_key] = {
            "n_agents": len(scores),
            "mean": float(np.mean(scores)) if scores else float("nan"),
            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
            "scores": scores,
            "per_agent": per_agent,
        }
        c = cons_by_cond[cond_key]
        print(f"  {cond_key}: n={c['n_agents']}  mean consistency={c['mean']:.3f} ± {c['sd']:.3f}")

    between_cons = {}
    print("\n  Cross-condition contrasts (consistency delta — higher Δ means condition-A more consistent):")
    for a_k, b_k in contrast_pairs:
        if a_k not in cons_by_cond or b_k not in cons_by_cond: continue
        a = np.asarray(cons_by_cond[a_k]["scores"])
        b = np.asarray(cons_by_cond[b_k]["scores"])
        res = contrast(a, b, rng)
        if res:
            between_cons[f"{a_k}-{b_k}"] = res
            mark = "✓" if res["ci_excludes_zero"] else " "
            print(f"    {a_k}-{b_k}:  Δ={res['delta']:+.3f}  CI=[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]  "
                  f"g={res['hedges_g']:+.2f}  Welch p={res['welch_p']:.4f}  {mark}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = {
        "scenario": args.scenario,
        "statsmodels_available": _HAS_STATSMODELS,
        "n_reps_per_condition": {k: len(v) for k, v in conds.items()},
        "conformity": {
            "per_condition": conf_by_cond,
            "between_conditions": between_conf,
        },
        "bfi10": {
            "per_condition": bfi_by_cond,
            "between_conditions": between_bfi,
            "between_conditions_per_factor": between_bfi_per_factor,
        },
        "consistency": {
            "per_condition": cons_by_cond,
            "between_conditions": between_cons,
        },
    }
    out_path = DATA / f"analysis_p3_conformity_bfi_{args.scenario}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    if not _HAS_STATSMODELS:
        print("\n[INFO] statsmodels not installed — mixed-effects logistic regression was skipped.")
        print("       Conformity aggregation used Fisher's combined probability test on")
        print("       per-agent McNemar p-values. To get the full §4.3.2 analysis:")
        print("         conda activate cogclaw && pip install statsmodels")


if __name__ == "__main__":
    main()
