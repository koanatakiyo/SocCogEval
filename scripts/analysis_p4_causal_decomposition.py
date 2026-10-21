"""analysis_p4_causal_decomposition.py — Part 3 causal contrast schema (§4.3.3).

Stitches together Part 1 (allocation) + Part 2 (conformity, BFI-10) into a
single summary table following the five-row contrast schema:

    (4) − (5)   Pure seed effect                 needs C4
    (2) − (4)   Interaction given seed           needs C4
    (2) − (3)   Seed given interaction           ready
    (2) − (1)   Network segregation              needs C1
    all − (5)   Baseline noise subtraction       (informational)

This script reads the three upstream JSON files:
    data/analysis_part1_contrasts_<scenario>.json
    data/analysis_p3_conformity_bfi_<scenario>.json

and produces a unified "SCT-positive" verdict per instrument per contrast,
following the §4.3.4 reporting standard:
  A finding is SCT-positive only if positive at Part 1 + positive at Part 2
  + survives the Part 3 interaction contrast with CI excluding zero.

Missing contrasts are reported as "(pending C1)" / "(pending C4)" rather
than dropped — so the table is drop-in ready for Chapter 6 and updates
automatically when C1/C4 finish.

Usage:
    python scripts/analysis_p4_causal_decomposition.py --scenario greenfield
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = (Path(__file__).resolve().parents[1] / "data")

# Keys match the ASCII "C{i}-C{j}" format used by analysis_p3_conformity_bfi.py
# and analysis_part1_contrasts.py when they write per-contrast rows to JSON.
# `display_label` is the pretty-printed version used only in table output.
CONTRAST_SPEC = [
    ("C4-C5", "(4) − (5)",  "Pure seed effect"),
    ("C2-C4", "(2) − (4)",  "Interaction effect, conditional on seed"),
    ("C2-C3", "(2) − (3)",  "Seed effect, conditional on interaction"),
    ("C2-C1", "(2) − (1)",  "Joint effect of segregation + seed asymmetry (C1 uses symmetric seed)"),
    ("C2-C5", "(2) − (5)",  "Total effect (seed × interaction)"),
    ("C3-C5", "(3) − (5)",  "Interaction alone  (C3 critical ablation)"),
]


def load_or_none(path: Path):
    if path.exists():
        return json.load(open(path))
    return None


def fmt_contrast(row: dict | None) -> str:
    if row is None:
        return "                (pending)                   "
    lo = row.get("ci95_lo"); hi = row.get("ci95_hi")
    if lo is None or hi is None:
        return "                (pending)                   "
    delta = row.get("delta", 0.0)
    g = row.get("hedges_g")
    if g is None:
        g_str = "    -"
    else:
        g_str = f"{g:+.2f}"
    excl = row.get("ci_excludes_zero", False)
    mark = "✓" if excl else " "
    return f"  Δ={delta:+5.2f}  CI=[{lo:+5.2f}, {hi:+5.2f}]  g={g_str}  {mark}"


def part1_row_for_contrast(p1: dict, contrast_label: str) -> dict | None:
    """Part 1 contrasts are stored per-dim. We return an aggregate row
    (mean Δ across dims, union of per-dim CIs is not meaningful — so we report
    the 'number of dims with CI excluding zero' instead)."""
    if p1 is None:
        return None
    contrasts = p1.get("contrasts", {})
    # Older P1 JSONs use unicode-minus + spaces ("C2 − C5"); P4 / P3 use
    # ASCII hyphen ("C2-C5"). Try both so we don't silently report "pending".
    c = contrasts.get(contrast_label) or contrasts.get(contrast_label.replace("-", " − "))
    if c is None:
        return None
    results = c["results"]
    n_sig = c.get("n_significant_dims", sum(1 for r in results if r["ci_excludes_zero"]))
    # Hedges' g may be None on dims where pooled SD collapsed (typically
    # contrasts against the constant C5 baseline). Average only over dims
    # where g is defined; if no dim has a defined g, omit g entirely.
    valid_g = [r["hedges_g"] for r in results if r.get("hedges_g") is not None]
    g_mean = (sum(valid_g) / len(valid_g)) if valid_g else None
    return {
        "delta": float(sum(r["delta"] for r in results) / len(results)),
        "ci95_lo": None, "ci95_hi": None,  # per-dim; see full table below
        "hedges_g": float(g_mean) if g_mean is not None else None,
        "n_dims_with_valid_g": len(valid_g),
        "ci_excludes_zero": n_sig >= len(results) // 2 + 1,  # "most dims significant"
        "n_sig_dims": n_sig, "n_total_dims": len(results),
    }


def fmt_part1_aggregate(p1: dict, contrast_label: str) -> str:
    if p1 is None:
        return "  (p1 json missing)                         "
    contrasts = p1.get("contrasts", {})
    c = contrasts.get(contrast_label) or contrasts.get(contrast_label.replace("-", " − "))
    if c is None:
        return "                (pending)                   "
    r = c["results"]
    n_sig = c.get("n_significant_dims")
    mean_delta = sum(x["delta"] for x in r) / len(r)
    valid_g = [x["hedges_g"] for x in r if x.get("hedges_g") is not None]
    if valid_g:
        mean_g = sum(valid_g) / len(valid_g)
        g_str = f"ḡ={mean_g:+.2f} ({len(valid_g)}/{len(r)})"
    else:
        g_str = "ḡ=undef (SD≈0)"
    return f"  Δ̄={mean_delta:+5.2f}  {g_str}   {n_sig}/{len(r)} dims sig"


def prediction_row_for_contrast(contrast_key: str, p2_pred: dict) -> dict | None:
    """For prediction, the natural summary per condition is Δ(out − in) target-MAE
    (positive = in-group predicted more accurately). We build condition-level
    'effects' and subtract to form the contrast row."""
    def condition_effect(cond_key: str) -> float | None:
        j = p2_pred.get(cond_key)
        if j is None: return None
        in_mae = j.get("in_target_mae", {}).get("mean")
        out_mae = j.get("out_target_mae", {}).get("mean")
        if in_mae is None or out_mae is None: return None
        return out_mae - in_mae   # larger = better in-group modelling
    pairs = {"C2-C5": ("C2", None), "C3-C5": ("C3", None), "C2-C3": ("C2", "C3"),
             "C4-C5": (None, None), "C2-C4": (None, None), "C2-C1": (None, None)}
    a_key, b_key = pairs.get(contrast_key, (None, None))
    a_eff = condition_effect(a_key) if a_key else None
    b_eff = condition_effect(b_key) if b_key else 0.0  # C5 / no-interaction predictions are meaningless
    if a_eff is None:
        return None
    delta = a_eff - (b_eff if b_eff is not None else 0.0)
    # Wilcoxon p from the A condition's JSON (one-sided p)
    p_val = (p2_pred.get(a_key) or {}).get("wilcoxon_one_sided", {}).get("p")
    return {
        "delta": float(delta),
        "wilcoxon_one_sided_p": float(p_val) if p_val is not None else None,
        "ci_excludes_zero": (p_val is not None and p_val < 0.05 and delta > 0),
        "a_effect_out_minus_in": float(a_eff),
        "b_condition": b_key,
    }


def fmt_prediction_for_contrast(contrast_key: str, p2_pred: dict) -> str:
    row = prediction_row_for_contrast(contrast_key, p2_pred)
    if row is None:
        return "    (n/a or pending)      "
    mark = "✓" if row["ci_excludes_zero"] else " "
    p_str = f"p={row['wilcoxon_one_sided_p']:.3f}" if row.get("wilcoxon_one_sided_p") is not None else "p=?"
    return f"  Δ(out−in)={row['delta']:+5.2f}  {p_str} {mark}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["greenfield", "riverton"], default="greenfield")
    args = ap.parse_args()

    p1_path = DATA / f"analysis_part1_contrasts_{args.scenario}.json"
    p3_path = DATA / f"analysis_p3_conformity_bfi_{args.scenario}.json"
    # Prediction is Part 1 (per §4.3.1 line 114). Loaded per-condition and
    # summarized as Δ(out − in) target-MAE since a *larger* gap means better
    # group-specific modelling.
    p2_pred_paths = {
        "C2": DATA / f"analysis_p2_prediction_c2_{args.scenario}.json",
        "C3": DATA / f"analysis_p2_prediction_c3_{args.scenario}.json",
    }
    p1 = load_or_none(p1_path)
    p3 = load_or_none(p3_path)
    p2_pred = {k: load_or_none(v) for k, v in p2_pred_paths.items()}

    print("=" * 100)
    print(f"  Part 3 — Causal Decomposition Summary   ({args.scenario.upper()})")
    print("=" * 100)
    print(f"  Upstream JSON files:")
    print(f"    Part 1 allocation:   {'✓' if p1 else 'MISSING — run analysis_part1_contrasts.py first'}")
    print(f"    Part 1 prediction:   C2={'✓' if p2_pred['C2'] else '✗'}  C3={'✓' if p2_pred['C3'] else '✗'}")
    print(f"    Part 2 (conformity + BFI + consistency): {'✓' if p3 else 'MISSING — run analysis_p3_conformity_bfi.py first'}")
    if p1:
        n = p1.get("n_reps_per_condition", {})
        print(f"    Part 1 reps per condition: {n}")

    # Table layout — 5 instrument columns per §4.3.1 + §4.3.2
    col_w = 42
    sep = "─" * 210
    print()
    print(sep)
    print(f"{'Contrast':<13}{'Description':<44}"
          f"{'Part 1a alloc':<{col_w}}"
          f"{'Part 1b pred':<30}"
          f"{'Part 2a conformity':<32}"
          f"{'Part 2b BFI-10':<32}"
          f"{'Part 2c consistency':<32}")
    print(sep)
    summary = {"scenario": args.scenario, "contrasts": {}}
    for key, label, descr in CONTRAST_SPEC:
        p1_s = fmt_part1_aggregate(p1, key) if p1 else "  (p1 missing)              "
        p2c = p3.get("conformity", {}).get("between_conditions", {}).get(key) if p3 else None
        p2b = p3.get("bfi10", {}).get("between_conditions", {}).get(key) if p3 else None
        p2cons = p3.get("consistency", {}).get("between_conditions", {}).get(key) if p3 else None
        # Prediction column: only direct C2 and C3 single-condition summaries
        # exist; for contrasts we show the raw Δ(out−in) target-MAE from C2
        # minus C3 where relevant.
        pred_cell = fmt_prediction_for_contrast(key, p2_pred)
        print(f"{label:<13}{descr:<44}"
              f"{p1_s:<{col_w}}"
              f"{pred_cell:<30}"
              f"{fmt_contrast(p2c):<32}"
              f"{fmt_contrast(p2b):<32}"
              f"{fmt_contrast(p2cons):<32}")
        summary["contrasts"][label] = {
            "description": descr,
            "part1_allocation": part1_row_for_contrast(p1, key) if p1 else None,
            "part1_prediction": prediction_row_for_contrast(key, p2_pred),
            "part2_conformity": p2c,
            "part2_bfi10": p2b,
            "part2_consistency": p2cons,
        }
    print(sep)

    # SCT-positive judgement per §4.3.4:
    #   "A finding is called SCT-positive only if it is positive at Part 1,
    #    positive at Part 2, and survives the Part 3 interaction contrast
    #    with CI excluding zero."
    # §4.3.3 defines the Part 3 interaction contrast as (2)−(4). So the gate
    # requires three separate positive tests:
    #   (i)   Part 1 majority-dim-significant on (2)−(5) (total effect exists)
    #   (ii)  Part 2 instrument positive on (2)−(5)
    #   (iii) (2)−(4) Part 2 CI excludes zero (interaction itself causally
    #         responsible, not just the seed)
    print("\n" + "=" * 100)
    print("  SCT-POSITIVE GATE CHECK (§4.3.4)")
    print("=" * 100)
    print("  Required for a finding to count as SCT-positive:")
    print("    (i)  Part 1 majority-of-dims significant on (2)−(5)")
    print("    (ii) Part 2 instrument significant on (2)−(5)")
    print("    (iii) Part 2 (2)−(4) interaction contrast CI excludes zero")
    print("  (iii) cannot be evaluated until Condition 4 data has been collected.")
    print()

    def contrast_row(display_label: str):
        return summary["contrasts"].get(display_label)

    total = contrast_row("(2) − (5)")
    interaction = contrast_row("(2) − (4)")
    p1_total = total.get("part1_allocation") if total else None
    p1_ok = (p1_total or {}).get("ci_excludes_zero", False)
    if p1_total:
        p1_frac = f"{p1_total.get('n_sig_dims','?')}/{p1_total.get('n_total_dims','?')}"
    else:
        p1_frac = "(pending)"

    for inst_key, inst_name in [("part2_conformity", "conformity"), ("part2_bfi10", "BFI-10")]:
        p2_total = (total or {}).get(inst_key)
        p2_inter = (interaction or {}).get(inst_key)

        p2_ok = (p2_total or {}).get("ci_excludes_zero", False)
        p2_inter_ok = (p2_inter or {}).get("ci_excludes_zero", False) if p2_inter else None

        status_parts = []
        status_parts.append(f"(i) Part 1 (2)−(5): {p1_frac}{' ✓' if p1_ok else ' ✗'}")
        if p2_total is None:
            status_parts.append("(ii) Part 2 (2)−(5): missing")
        else:
            status_parts.append(f"(ii) Part 2 (2)−(5): Δ={p2_total.get('delta', 0):+.3f} "
                                f"{'✓' if p2_ok else '✗'}")
        if p2_inter is None:
            status_parts.append("(iii) (2)−(4): pending C4 data")
            verdict = "INCOMPLETE (waiting on C4)"
        else:
            status_parts.append(f"(iii) Part 2 (2)−(4): Δ={p2_inter.get('delta', 0):+.3f} "
                                f"{'✓' if p2_inter_ok else '✗'}")
            verdict = ("SCT-POSITIVE" if (p1_ok and p2_ok and p2_inter_ok)
                       else "NOT SCT-positive")
        print(f"  {inst_name:11s}  " + "  |  ".join(status_parts) + f"  →  {verdict}")

    # Also print the interpretive rows that were there before
    print("\n  Supporting descriptive contrasts (not the SCT-positive gate):")
    for contrast_label, descr in [("(3) − (5)", "Interaction-alone (C3 critical ablation)"),
                                   ("(2) − (3)", "Seed effect given interaction")]:
        row = contrast_row(contrast_label)
        if not row:
            print(f"    {contrast_label}: (pending)")
            continue
        for inst_key, inst_name in [("part2_conformity", "conformity"), ("part2_bfi10", "BFI-10")]:
            r = row.get(inst_key)
            if r is None:
                print(f"    {contrast_label}  {inst_name:11s}: pending")
            else:
                ok = r.get("ci_excludes_zero")
                print(f"    {contrast_label}  {inst_name:11s}: Δ={r.get('delta',0):+.3f}  "
                      f"CI excludes 0: {ok}")

    # Write JSON
    out_path = DATA / f"analysis_p4_causal_decomposition_{args.scenario}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
