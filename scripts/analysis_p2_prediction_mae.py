"""analysis_p2_prediction_mae.py — SCT-Prediction (the judgment moment, §3.2).

Core question: do agents form group-specific models of others after 30 rounds
of segregated interaction? If yes, in-group prediction accuracy > out-group.

Design subtlety for segregated conditions (C2, C3):
    Agents in segregated topology NEVER interact with out-group agents.
    So "predict out-group" is prediction under zero interaction history.
    Under the prediction task's "Based on your interactions..." prompt, the
    model likely falls back to one of three defaults:
        (a) Project self (predict what I'd allocate)
        (b) Project in-group stereotype (predict what my group would allocate)
        (c) Some scenario-level default (e.g., uniform 20% per area)

A naive "MAE in-group < MAE out-group" Wilcoxon cannot distinguish (a) real
group-specific other-models from (b)/(c). So this script also reports:
    - self-MAE:  predict vs predictor's own post-allocation
    - ingroup-centroid-MAE: predict vs predictor's own in-group mean
    - outgroup-centroid-MAE: predict vs out-group mean (target's true group)
    - target-MAE: predict vs target's actual post-allocation (the standard metric)

Interpretation matrix:
    MAE(predict, target) < MAE(predict, ingroup-centroid): using target's identity
    MAE(predict, target) ≈ MAE(predict, self):             self-projection
    MAE(predict, target) ≈ MAE(predict, ingroup-centroid): in-group projection
    In-group target-MAE < Out-group target-MAE AND                               ← (a) the good result
    In-group predict/self-MAE ≈ Out-group predict/ingroup-MAE                    ←     (predictor differentiates via target-specific info)

Output: per-rep, per-(predictor, target) quadruple of MAEs, aggregated by
(predictor-group, target-group) pair. Stats: paired Wilcoxon signed-rank on
per-agent (ingroup-target-MAE − outgroup-target-MAE), one-sided.

Usage:
    python scripts/analysis_p2_prediction_mae.py --scenario greenfield --condition C2
    python scripts/analysis_p2_prediction_mae.py --scenario greenfield --condition C3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DATA = (Path(__file__).resolve().parents[1] / "data")
AREAS = {
    "greenfield": ["Education", "Healthcare", "Green Spaces", "Infrastructure", "Economic Development"],
    "riverton":   ["Automation", "Workforce Training", "Employee Support", "Innovation Lab", "Data Infrastructure"],
}
COND_GLOB = {
    "C1": "rep[0-9][0-9]_open_seed_{scenario}_*",
    "C2": "rep[0-9][0-9]_segreg_asym_seed_{scenario}_*",
    "C3": "rep[0-9][0-9]_segreg_no_seed_{scenario}_*",
}  # Only conditions with actual interaction; C4/C5 predictions are meaningless
SEED = 42


def vec(d: dict, areas: list[str]) -> np.ndarray:
    return np.array([float(d.get(a, 0.0)) for a in areas])


def load_run_predictions(run_dir: Path, areas: list[str]):
    """Return dict:
       {agents: {name: {group, actual_alloc}},
        predictions: [(pred_name, target_name, pred_alloc), ...]}"""
    try:
        summ = json.load(open(run_dir / "summary.json"))
        post = json.load(open(run_dir / "post_measurements.json"))
    except FileNotFoundError:
        return None
    agents = {}
    for a in summ["agents"]:
        agents[a["name"]] = {"group": a.get("group_id"), "actual_alloc": None}
    # Fill actual allocations
    for entry in post.get("allocation", []):
        name = entry.get("agent_name")
        allocs = entry.get("allocations") or {}
        if name in agents and isinstance(allocs, dict):
            agents[name]["actual_alloc"] = vec(allocs, areas)
    # Fill predictions
    preds = []
    for entry in post.get("prediction", []):
        pred = entry.get("agent_name")
        target = entry.get("target_name")
        allocs = entry.get("predicted_allocations") or entry.get("allocations") or {}
        if pred and target and isinstance(allocs, dict) and allocs:
            preds.append((pred, target, vec(allocs, areas)))
    # Sanity: skip if missing any agent's actual
    if any(a["actual_alloc"] is None for a in agents.values()):
        return None
    return {"agents": agents, "predictions": preds}


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def analyse_run(run_data: dict) -> list[dict]:
    """Per prediction: target-MAE, self-MAE, ingroup-centroid-MAE, outgroup-centroid-MAE.
    Return per-prediction records."""
    agents = run_data["agents"]
    # Group centroids (based on actual post-allocations)
    g_to_allocs: dict[int, list[np.ndarray]] = {}
    for name, info in agents.items():
        g_to_allocs.setdefault(info["group"], []).append(info["actual_alloc"])
    g_centroid = {g: np.mean(np.vstack(v), axis=0) for g, v in g_to_allocs.items()}

    recs = []
    for pred_name, tgt_name, pred_alloc in run_data["predictions"]:
        p = agents.get(pred_name)
        t = agents.get(tgt_name)
        if p is None or t is None or p["actual_alloc"] is None or t["actual_alloc"] is None:
            continue
        pg, tg = p["group"], t["group"]
        relation = "in" if (pg is not None and pg == tg) else "out"
        recs.append({
            "predictor": pred_name,
            "target": tgt_name,
            "predictor_group": pg,
            "target_group": tg,
            "relation": relation,
            "target_mae":            mae(pred_alloc, t["actual_alloc"]),
            "self_mae":              mae(pred_alloc, p["actual_alloc"]),
            "ingroup_centroid_mae":  mae(pred_alloc, g_centroid.get(pg, np.zeros_like(pred_alloc))),
            "outgroup_centroid_mae": mae(pred_alloc, g_centroid.get(tg, np.zeros_like(pred_alloc))),
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["greenfield", "riverton"], default="greenfield")
    ap.add_argument("--condition", choices=list(COND_GLOB.keys()), default="C2")
    args = ap.parse_args()
    areas = AREAS[args.scenario]

    pattern = COND_GLOB[args.condition].format(scenario=args.scenario)
    run_dirs = sorted(DATA.glob(pattern))
    print(f"P2 Prediction MAE — condition {args.condition}, scenario {args.scenario}")
    print(f"Reps found: {len(run_dirs)}")
    if not run_dirs:
        sys.exit(f"ERROR: no reps match {pattern}")

    all_recs: list[dict] = []
    per_rep_by_relation = []
    for d in run_dirs:
        run = load_run_predictions(d, areas)
        if run is None:
            print(f"  SKIP {d.name}: missing data")
            continue
        recs = analyse_run(run)
        for r in recs:
            r["rep"] = d.name
        all_recs.extend(recs)
        # per-rep aggregate
        ing = [r for r in recs if r["relation"] == "in"]
        out = [r for r in recs if r["relation"] == "out"]
        per_rep_by_relation.append({
            "rep": d.name,
            "n_in": len(ing),
            "n_out": len(out),
            "in_target_mae":        float(np.mean([r["target_mae"] for r in ing]))            if ing else np.nan,
            "out_target_mae":       float(np.mean([r["target_mae"] for r in out]))           if out else np.nan,
            "in_self_mae":          float(np.mean([r["self_mae"] for r in ing]))              if ing else np.nan,
            "out_self_mae":         float(np.mean([r["self_mae"] for r in out]))              if out else np.nan,
            "in_ingroup_cent_mae":  float(np.mean([r["ingroup_centroid_mae"] for r in ing]))  if ing else np.nan,
            "out_ingroup_cent_mae": float(np.mean([r["ingroup_centroid_mae"] for r in out])) if out else np.nan,
        })
        print(f"  {d.name[:50]}...  in={len(ing):2d}  out={len(out):2d}")

    # ─────────────────────────────────────────────────────────────────────────
    # Headline: per-agent aggregate (pool all predictions by predictor across reps)
    # ─────────────────────────────────────────────────────────────────────────
    per_agent_key = {}  # (rep, predictor) → dict of {in_tgt_mae, out_tgt_mae, ...}
    for r in all_recs:
        key = (r["rep"], r["predictor"])
        slot = per_agent_key.setdefault(key, {
            "in_tgt": [], "out_tgt": [],
            "in_self": [], "out_self": [],
            "in_ingcent": [], "out_ingcent": [],
        })
        tag = r["relation"]
        slot[f"{tag}_tgt"].append(r["target_mae"])
        slot[f"{tag}_self"].append(r["self_mae"])
        slot[f"{tag}_ingcent"].append(r["ingroup_centroid_mae"])

    # build per-agent paired vectors
    agents_summary = []
    for (rep, predictor), slot in per_agent_key.items():
        if not slot["in_tgt"] or not slot["out_tgt"]:
            continue  # skip agents in non-segregated conditions without out-group
        agents_summary.append({
            "rep": rep, "predictor": predictor,
            "in_tgt_mae":  float(np.mean(slot["in_tgt"])),
            "out_tgt_mae": float(np.mean(slot["out_tgt"])),
            "in_self_mae":  float(np.mean(slot["in_self"])),
            "out_self_mae": float(np.mean(slot["out_self"])),
            "in_ingcent_mae":  float(np.mean(slot["in_ingcent"])),
            "out_ingcent_mae": float(np.mean(slot["out_ingcent"])),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Main statistic: per-agent paired Wilcoxon on (in_tgt − out_tgt), one-sided
    # ─────────────────────────────────────────────────────────────────────────
    in_tgt = np.array([a["in_tgt_mae"] for a in agents_summary])
    out_tgt = np.array([a["out_tgt_mae"] for a in agents_summary])
    diffs = in_tgt - out_tgt

    if len(diffs) < 3:
        print("\nInsufficient agents for statistical test")
        sys.exit(1)

    # Standard SCT-Prediction test
    w_stat, p_two = stats.wilcoxon(in_tgt, out_tgt, zero_method="wilcox", alternative="two-sided")
    w_less, p_less = stats.wilcoxon(in_tgt, out_tgt, zero_method="wilcox", alternative="less")

    print()
    print("=" * 80)
    print(f"HEADLINE: SCT-Prediction test  (n={len(agents_summary)} predictor-reps)")
    print("=" * 80)
    print(f"  in-group target MAE  (mean ± sd): {in_tgt.mean():.3f} ± {in_tgt.std(ddof=1):.3f}")
    print(f"  out-group target MAE (mean ± sd): {out_tgt.mean():.3f} ± {out_tgt.std(ddof=1):.3f}")
    print(f"  Δ (in − out):                     {diffs.mean():+.3f} ± {diffs.std(ddof=1):.3f}")
    print(f"  Wilcoxon signed-rank (in < out):  W={w_less:.1f}, p={p_less:.4f} (one-sided)")
    print(f"  Wilcoxon two-sided:               W={w_stat:.1f}, p={p_two:.4f}")
    verdict = "POSITIVE" if p_less < 0.05 and diffs.mean() < 0 else ("NEGATIVE" if p_less > 0.05 else "?")
    print(f"  → SCT-Prediction {verdict}")

    # ─────────────────────────────────────────────────────────────────────────
    # Self-projection diagnostic: does predict match self more than target?
    # ─────────────────────────────────────────────────────────────────────────
    in_self = np.array([a["in_self_mae"] for a in agents_summary])
    out_self = np.array([a["out_self_mae"] for a in agents_summary])
    in_ingcent = np.array([a["in_ingcent_mae"] for a in agents_summary])
    out_ingcent = np.array([a["out_ingcent_mae"] for a in agents_summary])

    print()
    print("=" * 80)
    print("DIAGNOSTIC: are predictions actually group-specific, or just self/in-group projection?")
    print("=" * 80)
    print(f"{'Metric':<38}{'in':>10}{'out':>10}{'Δ(in-out)':>12}")
    print("-" * 70)
    print(f"{'target MAE (lower = more accurate)':<38}{in_tgt.mean():>10.3f}{out_tgt.mean():>10.3f}{diffs.mean():>+12.3f}")
    print(f"{'self MAE (lower = predicts like self)':<38}{in_self.mean():>10.3f}{out_self.mean():>10.3f}{(in_self - out_self).mean():>+12.3f}")
    print(f"{'ingroup-centroid MAE (lower = in-group like)':<38}{in_ingcent.mean():>10.3f}{out_ingcent.mean():>10.3f}{(in_ingcent - out_ingcent).mean():>+12.3f}")

    # Interpretation flags
    print()
    print("Interpretation:")
    if in_tgt.mean() < out_tgt.mean() - 0.5:
        print(f"  [i] in-target < out-target by {abs(diffs.mean()):.2f}   →  in-group predicted more accurately")
    else:
        print(f"  [i] in-target ≈ out-target   →  no in-group prediction advantage")

    if abs(in_tgt.mean() - in_self.mean()) < 0.5:
        print(f"  [ii] in-target MAE ≈ in-self MAE ({in_tgt.mean():.2f} vs {in_self.mean():.2f})")
        print(f"       → predictor may be projecting own allocation onto in-group targets")
    if abs(out_tgt.mean() - out_ingcent.mean()) < 0.5:
        print(f"  [iii] out-target MAE ≈ predict-vs-predictor's-own-group MAE ({out_tgt.mean():.2f} vs {out_ingcent.mean():.2f})")
        print(f"        → predictor may be projecting own GROUP onto out-group targets (in-group stereotype for out-group)")

    # ─────────────────────────────────────────────────────────────────────────
    # Save JSON
    # ─────────────────────────────────────────────────────────────────────────
    out_path = DATA / f"analysis_p2_prediction_{args.condition.lower()}_{args.scenario}.json"
    # Effective rep count = distinct rep dirs that produced usable predictor
    # data (both in-group and out-group target MAEs). This excludes glob
    # duplicates from failed/restarted runs and any rep that didn't yield a
    # well-formed prediction matrix. Use this — not n_reps_found — for any
    # sample-size citation in the paper.
    n_reps_used = len({a["rep"] for a in agents_summary})
    out = {
        "scenario": args.scenario,
        "condition": args.condition,
        "n_reps_found": len(run_dirs),
        "n_reps_used": n_reps_used,
        "n_predictor_observations": len(agents_summary),
        "in_target_mae":        {"mean": float(in_tgt.mean()), "sd": float(in_tgt.std(ddof=1))},
        "out_target_mae":       {"mean": float(out_tgt.mean()), "sd": float(out_tgt.std(ddof=1))},
        "delta_in_minus_out":   {"mean": float(diffs.mean()), "sd": float(diffs.std(ddof=1))},
        "wilcoxon_one_sided":   {"W": float(w_less), "p": float(p_less)},
        "wilcoxon_two_sided":   {"W": float(w_stat), "p": float(p_two)},
        "in_self_mae_mean":     float(in_self.mean()),
        "out_self_mae_mean":    float(out_self.mean()),
        "in_ingcent_mae_mean":  float(in_ingcent.mean()),
        "out_ingcent_mae_mean": float(out_ingcent.mean()),
        "agents_summary":       agents_summary,
        "per_rep_by_relation":  per_rep_by_relation,
        "verdict":              verdict,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
