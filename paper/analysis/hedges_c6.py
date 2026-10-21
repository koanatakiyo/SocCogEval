#!/usr/bin/env python3
"""Agent-level Hedges' g for the C6 rows, with the submitted formula (pooled SD, small-sample J).
Check first: the formula must reproduce the submitted C2-C4 and C4-C5 g in analysis_p4_*.json.
Usage (from the repository root): python paper/analysis/hedges_c6.py . <output dir>"""
import csv, json, sys
from math import sqrt
from pathlib import Path
from statistics import mean, variance
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); REB = ROOT / "results/reference"
F = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


def rj(p): return json.loads(Path(p).read_text())


def agent_l1(path):
    pre = {r["agent_name"]: r["scores"] for r in rj(path / "pre_measurements.json")["bfi10"]}
    post = {r["agent_name"]: r["scores"] for r in rj(path / "post_measurements.json")["bfi10"]}
    return [sum(abs(float(post[a][f]) - float(pre[a][f])) for f in F) for a in pre if a in post]


def g(x, y):
    n1, n2 = len(x), len(y)
    sp = sqrt(((n1 - 1) * variance(x) + (n2 - 1) * variance(y)) / (n1 + n2 - 2))
    return (mean(x) - mean(y)) / sp * (1 - 3 / (4 * (n1 + n2) - 9))


vals = {}
for r in csv.DictReader(open(REB / "run_ledger.csv")):
    if r["variant_alias"] == "primary" and r["memory_mode"] == "srm":
        vals.setdefault((r["scenario"], r["condition"]), []).extend(agent_l1(ROOT / "data" / r["run_id"]))
for r in csv.DictReader(open(REB / "c6_run_level.csv")):
    vals.setdefault((r["scenario"], "C6"), []).extend(float(v) for v in r["agent_l1_values"].split(";"))
out = []
for sc in ["greenfield", "riverton"]:
    ref = rj(ROOT / f"results/legacy/analysis_p4_causal_decomposition_{sc}.json")["contrasts"]
    for a, b in [("C2", "C4"), ("C4", "C5"), ("C3", "C5"), ("C2", "C5")]:
        mine = g(vals[(sc, a)], vals[(sc, b)]); sub = ref[f"({a[1]}) − ({b[1]})"]["part2_bfi10"]["hedges_g"]
        assert abs(mine - sub) < 1e-9, (sc, a, b, mine, sub)
    for a, b in [("C6", "C4"), ("C2", "C6")]:
        out.append(dict(scenario=sc, contrast=f"{a}-{b}", n_a=len(vals[(sc, a)]), n_b=len(vals[(sc, b)]), hedges_g=g(vals[(sc, a)], vals[(sc, b)])))
print("check passed: formula reproduces submitted g")
with open(OUT / "hedges_c6.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out: print(r)
