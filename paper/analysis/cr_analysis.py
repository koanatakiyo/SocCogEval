#!/usr/bin/env python3
"""Camera-ready additions computed from the stored runs (new evidence only).

1. Run-level BFI-10 L1 means for every primary condition, using the run selection in
   results/reference/run_ledger.csv (C1-C5) and c6_run_level.csv (C6), then two-sided
   exact permutation p and run-bootstrap intervals for every FSIP contrast.
   Check: C2/C4/C6 run means must equal results/reference/c6_decomposition.csv.
2. Belief-probe time course: allocation percentages parsed from the free-text belief probe
   (rounds 5..30), plus the pre/post allocation probes, turned into A/B group divergence.

Usage (from the repository root): python paper/analysis/cr_analysis.py . <output dir>
"""
import csv, json, itertools, random, re, sys
from pathlib import Path
from statistics import mean

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"; REB = ROOT / "results/reference"
FACTORS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
AREAS = {
    "greenfield": ["Education", "Healthcare", "Green Spaces", "Infrastructure", "Economic Development"],
    "riverton": ["Automation", "Workforce Training", "Employee Support", "Innovation Lab", "Data Infrastructure"],
}
BUDGET = {"greenfield": 10.0, "riverton": 5.0}  # $M


def rj(p): return json.loads(Path(p).read_text())


def selected_runs():
    runs = {}
    for r in csv.DictReader(open(REB / "run_ledger.csv")):
        if r["variant_alias"] == "primary" and r["memory_mode"] == "srm" and r["exclusion_status"].startswith("included"):
            runs.setdefault((r["scenario"], r["condition"]), []).append(DATA / r["run_id"])
    for r in csv.DictReader(open(REB / "c6_run_level.csv")):
        key = next(k for k in r if "run" in k and "id" in k) if "run_id" not in r else "run_id"
        runs.setdefault((r["scenario"], "C6"), []).append(DATA / r[key])
    return {k: sorted(v) for k, v in runs.items()}


def agent_order(path):
    pre = rj(path / "pre_measurements.json")["bfi10"]
    return [row["agent_name"] for row in pre]


def run_l1(path):
    pre = {r["agent_name"]: r["scores"] for r in rj(path / "pre_measurements.json")["bfi10"]}
    post = {r["agent_name"]: r["scores"] for r in rj(path / "post_measurements.json")["bfi10"]}
    vals = [sum(abs(float(post[a][f]) - float(pre[a][f])) for f in FACTORS) for a in pre if a in post]
    return mean(vals)


def exact_p(x, y):
    obs = abs(mean(x) - mean(y)); pool = x + y; n = len(x); hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx); a = [pool[i] for i in idx]; b = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1; hits += abs(mean(a) - mean(b)) >= obs - 1e-12
    return hits / tot


def boot_ci(x, y, draws=50000, seed=20261020):
    rng = random.Random(seed); d = []
    for _ in range(draws):
        d.append(mean(rng.choices(x, k=len(x))) - mean(rng.choices(y, k=len(y))))
    d.sort(); return d[int(0.025 * draws)], d[int(0.975 * draws) - 1]


# ---------- 1. run-level contrasts ----------
runs = selected_runs()
means = {k: [run_l1(p) for p in v] for k, v in runs.items()}
check = {r["scenario"]: r for r in csv.DictReader(open(ROOT / "results/reference/c6_decomposition.csv"))}
for sc, r in check.items():
    for c in ["c2", "c4", "c6"]:
        ref = sorted(round(float(v), 4) for v in r[f"{c}_run_values"].split(";"))
        got = sorted(round(v, 4) for v in means[(sc, c.upper())])
        assert ref == got, (sc, c, ref, got)
print("check passed: C2/C4/C6 run means equal c6_decomposition.csv")

CONTRASTS = [("C4", "C5", "brief alone"), ("C2", "C4", "interaction given brief"),
             ("C2", "C3", "brief given interaction"), ("C3", "C5", "interaction alone"),
             ("C2", "C5", "total"), ("C6", "C4", "solo generation given brief"),
             ("C2", "C6", "social versus solo"), ("C2", "C1", "descriptive: symmetry and topology")]
rows = []
for sc in ["greenfield", "riverton"]:
    for a, b, label in CONTRASTS:
        x, y = means[(sc, a)], means[(sc, b)]
        lo, hi = boot_ci(x, y)
        rows.append(dict(scenario=sc, contrast=f"{a}-{b}", isolates=label, n_a=len(x), n_b=len(y),
                         delta=mean(x) - mean(y), run_ci_lo=lo, run_ci_hi=hi, exact_p=exact_p(x, y)))
with open(OUT / "run_level_contrasts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(OUT / "run_level_means.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scenario", "condition", "run_id", "bfi_l1_run_mean"])
    for (sc, c), paths in sorted(runs.items()):
        for p, m in zip(paths, means[(sc, c)]): w.writerow([sc, c, p.name, f"{m:.6f}"])
for r in rows: print(f"{r['scenario']:10s} {r['contrast']:6s} {r['delta']:+.3f} [{r['run_ci_lo']:+.3f},{r['run_ci_hi']:+.3f}] p={r['exact_p']:.4f}")

# ---------- 2. belief-probe time course ----------
NUM = r"(\d+(?:\.\d+)?)"


def parse_alloc(text, sc):
    areas = AREAS[sc]; out = {}
    low = text
    for area in areas:
        stem = re.escape(area.split()[0]) + r"[A-Za-z ]{0,25}?"
        pats = [rf"{NUM}\s*%\s*(?:of the budget\s*)?(?:to|for|in|on|toward|towards|into)\s+(?:the\s+)?{stem}",
                rf"{stem}\s*(?:\(|:|-|at|with|receiving|gets|to)?\s*{NUM}\s*%",
                rf"\$\s*{NUM}\s*(?:M|million)\b[^.]{{0,15}}?(?:to|for|in|on|toward|into)\s+(?:the\s+)?{stem}",
                rf"{stem}\s*(?:\(|:|-|at|with|receiving|gets)?\s*\$\s*{NUM}\s*(?:M|million)"]
        for i, pat in enumerate(pats):
            m = re.search(pat, low, flags=re.I)
            if m:
                v = float(m.group(1)); out[area] = v / BUDGET[sc] * 100 if i >= 2 else v; break
    if len(out) == 5 and 95 <= sum(out.values()) <= 105:
        return [out[a] for a in areas]
    return None


def divergence(vecs, names):
    a = [vecs[n] for n in names[:4] if vecs.get(n)]; b = [vecs[n] for n in names[4:] if vecs.get(n)]
    if len(a) < 2 or len(b) < 2: return None
    return mean(abs(mean(v[j] for v in a) - mean(v[j] for v in b)) for j in range(5))


tc_rows = []; parse_stats = {}
for (sc, cond), paths in sorted(runs.items()):
    for p in paths:
        names = agent_order(p)
        pre = {r["agent_name"]: [r["allocations"][a] for a in AREAS[sc]] for r in rj(p / "pre_measurements.json")["allocation"]}
        post = {r["agent_name"]: [r["allocations"][a] for a in AREAS[sc]] for r in rj(p / "post_measurements.json")["allocation"]}
        tc_rows.append(dict(scenario=sc, condition=cond, run_id=p.name, round="pre", divergence=divergence(pre, names), n_parsed=len(pre)))
        for rf in sorted((p / "rounds").glob("round_*.json")) if (p / "rounds").exists() else []:
            r = rj(rf); pr = r.get("probe_responses") or {}
            if not pr.get("belief"): continue
            vecs = {}
            for b in pr["belief"]:
                v = parse_alloc(b["response_text"], sc); vecs[b["agent_name"]] = v
                st = parse_stats.setdefault(cond, [0, 0]); st[0] += v is not None; st[1] += 1
            tc_rows.append(dict(scenario=sc, condition=cond, run_id=p.name, round=r["round_num"],
                                divergence=divergence(vecs, names), n_parsed=sum(v is not None for v in vecs.values())))
        tc_rows.append(dict(scenario=sc, condition=cond, run_id=p.name, round="post", divergence=divergence(post, names), n_parsed=len(post)))
with open(OUT / "belief_time_course.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tc_rows[0])); w.writeheader(); w.writerows(tc_rows)
print("belief parse rate by condition:", {k: f"{a}/{b} ({a / b:.0%})" for k, (a, b) in parse_stats.items()})
summary = {}
for r in tc_rows:
    if r["divergence"] is None: continue
    summary.setdefault((r["scenario"], r["condition"], str(r["round"])), []).append(r["divergence"])
for k in sorted(summary): print(k, f"{mean(summary[k]):.2f}", f"n={len(summary[k])}")
