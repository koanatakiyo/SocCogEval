#!/usr/bin/env python3
"""Belief-probe time course (new evidence for the camera-ready).

Two per-round measures from the free-text belief probe (rounds 5..30), on the ledger-selected
primary runs (C1, C2, C3) and the C6 runs:
  (a) allocation divergence: |mean_A - mean_B| averaged over the five areas, using vectors parsed
      from the text (only probes that state all five areas; pre/post use the structured probes);
  (b) wording separation: mean within-group minus mean between-group Jaccard word overlap.
Groups are agents 0-3 (A) and 4-7 (B), as in the BFI scripts.
Usage (from the repository root): python paper/analysis/belief_time_course.py . <output dir>
"""
import csv, json, re, sys
from pathlib import Path
from statistics import mean
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); DATA = ROOT / "data"; REB = ROOT / "results/reference"
AREAS = {"greenfield": ["Education", "Healthcare", "Green Spaces", "Infrastructure", "Economic Development"],
         "riverton": ["Automation", "Workforce Training", "Employee Support", "Innovation Lab", "Data Infrastructure"]}
STEMS = {"greenfield": [r"Education", r"Health\s*care|Health", r"Green\s*Spaces?", r"(?<!Data )Infrastructure", r"Economic(?:\s+Development)?"],
         "riverton": [r"Automation", r"(?:Workforce\s+)?Training", r"(?:Employee\s+)?Support", r"Innovation(?:\s+Lab)?", r"Data(?:\s+Infrastructure)?"]}
STOP = set("the a an and or of to in for on with is are be that this as by our we my i it its at from than not but".split())


def rj(p): return json.loads(Path(p).read_text())


def runs():
    out = {}
    for r in csv.DictReader(open(REB / "run_ledger.csv")):
        if r["variant_alias"] == "primary" and r["memory_mode"] == "srm" and r["condition"] in ("C1", "C2", "C3"):
            out.setdefault((r["scenario"], r["condition"]), []).append(DATA / r["run_id"])
    for r in csv.DictReader(open(REB / "c6_run_level.csv")):
        out.setdefault((r["scenario"], "C6"), []).append(DATA / r["run_id"])
    return out


def parse(text, sc):
    vals = []
    for stem in STEMS[sc]:
        m = (re.search(rf"(\d+(?:\.\d+)?)\s*%\s*(?:of the budget\s+)?(?:to|for|in|on|toward|into)?\s*(?:the\s+)?(?:{stem})", text, re.I)
             or re.search(rf"(?:{stem})[^.%\d]{{0,25}}?(\d+(?:\.\d+)?)\s*%", text, re.I))
        if not m: return None
        vals.append(float(m.group(1)))
    return vals if 90 <= sum(vals) <= 110 else None


def words(t): return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2}


def jac(a, b): return len(a & b) / max(1, len(a | b))


def div(vecs, names):
    A = [vecs[n] for n in names[:4] if vecs.get(n)]; B = [vecs[n] for n in names[4:] if vecs.get(n)]
    if len(A) < 2 or len(B) < 2: return None
    return mean(abs(mean(v[j] for v in A) - mean(v[j] for v in B)) for j in range(5))


rows = []; ok = tot = 0
for (sc, cond), paths in sorted(runs().items()):
    for p in sorted(paths):
        names = [r["agent_name"] for r in rj(p / "pre_measurements.json")["bfi10"]]
        for tag, fn in (("pre", "pre_measurements.json"), ("post", "post_measurements.json")):
            vec = {r["agent_name"]: [r["allocations"][a] for a in AREAS[sc]] for r in rj(p / fn)["allocation"]}
            rows.append(dict(scenario=sc, condition=cond, run_id=p.name, round=0 if tag == "pre" else 31, source=tag,
                             alloc_divergence=div(vec, names), wording_separation=None, n_parsed=len(vec)))
        for rf in sorted((p / "rounds").glob("round_*.json")):
            pr = rj(rf).get("probe_responses") or {}
            if not pr.get("belief"): continue
            txt = {b["agent_name"]: b["response_text"] for b in pr["belief"]}
            vec = {n: parse(t, sc) for n, t in txt.items()}
            ok += sum(v is not None for v in vec.values()); tot += len(vec)
            W = {n: words(t) for n, t in txt.items()}
            ga = [n for n in names[:4] if n in W]; gb = [n for n in names[4:] if n in W]
            within = [jac(W[x], W[y]) for g in (ga, gb) for i, x in enumerate(g) for y in g[i + 1:]]
            between = [jac(W[x], W[y]) for x in ga for y in gb]
            rows.append(dict(scenario=sc, condition=cond, run_id=p.name, round=pr.get("round"), source="belief",
                             alloc_divergence=div(vec, names), wording_separation=mean(within) - mean(between),
                             n_parsed=sum(v is not None for v in vec.values())))
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "belief_time_course.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"parsed full allocation vectors: {ok}/{tot} ({ok / tot:.0%})")
agg = {}
for r in rows:
    for m in ("alloc_divergence", "wording_separation"):
        if r[m] is not None: agg.setdefault((m, r["scenario"], r["condition"], r["round"]), []).append(r[m])
for k in sorted(agg, key=lambda k: (k[0], k[1], k[2], k[3])):
    print(k, f"{mean(agg[k]):+.3f}", f"n={len(agg[k])}")
