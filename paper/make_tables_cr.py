#!/usr/bin/env python3
"""Camera-ready table fragments. Reads evidence/ (rebuttal), evidence_sub/ (submitted analysis
outputs) and evidence_cr/ (camera-ready additions). No number is typed by hand."""
import csv, json, math
from pathlib import Path

P = Path(__file__).resolve().parent
EV, SUB, CR, T = P / "evidence", P / "evidence_sub", P / "evidence_cr", P / "tables"
SC = ["greenfield", "riverton"]


def rcsv(p): return list(csv.DictReader(open(p)))
def rj(p): return json.load(open(p))
def row(*x): return " & ".join(map(str, x)) + r" \\"
def save(n, lines): (T / n).write_text("\n".join(lines) + "\n")
def sg(x, d=3): return f"${float(x):+.{d}f}$"
def ci(lo, hi, d=3): return f"$[{float(lo):+.{d}f},{float(hi):+.{d}f}]$"


def p_sub(p):
    """Submitted print style for Welch/Wilcoxon p."""
    if p < 1e-3:
        return f"$<10^{{-{min(7, math.floor(-math.log10(p)))}}}$"
    return f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"


def p_ex(p): return f"${float(p):.4f}$"


# ---- Table 1: FSIP decomposition + C6, run-level (rebuttal estimator), g secondary ----
c6 = {r["scenario"]: r for r in rcsv(EV / "c6_decomposition.csv")}
rl = {(r["scenario"], r["contrast"]): r for r in rcsv(CR / "run_level_contrasts.csv")}
hg6 = {(r["scenario"], r["contrast"]): float(r["hedges_g"]) for r in rcsv(CR / "hedges_c6.csv")}
p4 = {s: rj(SUB / f"analysis_p4_causal_decomposition_{s}.json")["contrasts"] for s in SC}
ROWS = [("C4-C5", "brief alone", ""), ("C2-C4", "interaction given brief", "(iii)"), ("C2-C3", "brief given interaction", ""),
        ("C3-C5", "interaction alone", ""), ("C2-C5", "total", "(ii)"), ("C6-C4", "solo generation given brief", ""),
        ("C2-C6", "social vs.\\ solo generation", "")]
out = []
for k, iso, gate in ROWS:
    a, b = k.split("-"); cells = []
    for s in SC:
        if k in ("C2-C4", "C6-C4", "C2-C6"):
            key = k.lower().replace("-", "_minus_"); r = c6[s]
            d, lo, hi, p = r[key], r[key + "_ci_lo"], r[key + "_ci_hi"], r[key + "_exact_p"]
        else:
            r = rl[(s, k)]; d, lo, hi, p = r["delta"], r["run_ci_lo"], r["run_ci_hi"], r["exact_p"]
        g = hg6[(s, k)] if k in ("C6-C4", "C2-C6") else p4[s][f"({a[1]}) − ({b[1]})"]["part2_bfi10"]["hedges_g"]
        cells += [sg(d), f"${g:+.2f}$", ci(lo, hi), p_ex(p)]
    name = f"$({a})-({b})$".replace("C", "C_")
    line = row(name, iso, gate, *cells)
    if k == "C2-C4":
        line = r"\rowcolor{c2shade} " + line.replace(name, r"\textbf{" + name + "}").replace(iso, r"\textbf{pure interaction}")
    out.append(line)
    if k == "C2-C5": out.append(r"\midrule")
save("t1_decomp.tex", out)

# ---- Table 2: setting ablations ----
t0 = rj(SUB / "bfi_t0_summary.json"); ns = {r["scenario"]: r for r in rcsv(EV / "nosrm_exact_permutation.csv")}
rows2 = [("Primary, $T{=}0.7$", "$C_2-C_4$", [(c6[s]["c2_minus_c4"], c6[s]["c2_minus_c4_exact_p"]) for s in SC]),
         ("Re-probe, $T{=}0$", "$C_2-C_4$", [(t0[s]["contrast"]["c2_minus_c4"], t0[s]["contrast"]["exact_two_sided_permutation_p"]) for s in SC]),
         ("No SRM", "$C_2-C_5$", [(ns[s]["delta"], ns[s]["exact_two_sided_permutation_p"]) for s in SC]),
         ("Solo ($C_6$)", "$C_6-C_4$", [(c6[s]["c6_minus_c4"], c6[s]["c6_minus_c4_exact_p"]) for s in SC])]
save("t2_settings.tex", [row(n, c, *[x for d, p in v for x in (sg(d), p_ex(p))]) for n, c, v in rows2])

# ---- Table 3A: gate status, as submitted (agent-stratified bootstrap, Welch p, Hedges g) ----
part1 = {s: rj(SUB / f"analysis_part1_contrasts_{s}.json")["contrasts"]["C2 − C5"] for s in SC}
def mg(c): gs = [r["hedges_g"] for r in c["results"] if r["hedges_g"] is not None]; return sum(gs) / len(gs)
g_rows = [row(r"\rowcolor{gateshade} (i) Part 1 $(C_2){-}(C_5)$ majority-dim rule",
              *[x for s in SC for x in (f"{part1[s]['n_significant_dims']} of 5 dims clear", f"$\\bar g{{=}}{mg(part1[s]):+.2f}$ \\yes")])]
for inst, key, d in [("BFI-10", "part2_bfi10", 2), ("Conformity", "part2_conformity", 3), ("Consistency", "part2_consistency", 3)]:
    res = {}
    for gate, k in [("(ii)", "(2) − (5)"), ("(iii)", "(2) − (4)")]:
        cells = []
        for s in SC:
            r = p4[s][k][key]; ok = r["ci_excludes_zero"]; res.setdefault(s, []).append(ok)
            ps = p_sub(r["welch_p"]).strip("$")
            stat = ci(r["ci95_lo"], r["ci95_hi"], 2) if (inst == "BFI-10" and gate == "(iii)") else (f"$p{{=}}{ps}$" if ps[0].isdigit() else f"$p{ps}$")
            cells += [f"${r['delta']:+.{d}f}$ (${r['hedges_g']:+.2f}$)", stat + (r" \yes" if ok else r" \no")]
        g_rows.append(row(f"{gate} {inst} $(C_2){{-}}(C_{k[7]})$", *cells))
    verdict = []
    for s in SC:
        ok = all(res[s]) and part1[s]["n_significant_dims"] >= 3
        if ok: v = r"\textbf{SCT-positive}"
        else: v = "not positive (" + ("ii" if not res[s][0] else "iii") + " fails)"
        verdict.append(rf"\multicolumn{{2}}{{c}}{{{v}}}")
    g_rows.append(r"\rowcolor{verdictshade} " + row(rf"\textbf{{{inst} verdict}}" if inst == "BFI-10" else f"{inst} verdict", *verdict))
    if inst != "Consistency": g_rows.append(r"\midrule")
save("t3_gates.tex", g_rows)

# ---- Appendix: submitted agent-level decomposition, cross-model C2-C5, no-SRM ----
app = []
for k, iso, _ in ROWS[:5]:
    a, b = k.split("-"); cells = []
    for s in SC:
        r = p4[s][f"({a[1]}) − ({b[1]})"]["part2_bfi10"]
        cells += [f"${r['delta']:+.2f}$ (${r['hedges_g']:+.2f}$)", ci(r["ci95_lo"], r["ci95_hi"], 2), p_sub(r["welch_p"])]
    app.append(row(f"$({a})-({b})$".replace("C", "C_"), iso, *cells))
save("app_decomp_agent.tex", app)
NAMES = {"": "Qwen3-122B-FP8 (primary)", "qvl": "Qwen3-VL-32B (text-only)", "dpsk": "DeepSeek-V3", "gpt": "GPT-4.1-mini",
         "gem": "Gemini-2.5-Flash", "l4api": "Llama-4-Scout (OpenRouter)"}
cm = {s: rj(SUB / f"analysis_cross_model_{s}.json")["models"] for s in SC}
crow = []
for i, m in enumerate(cm["greenfield"]):
    cells = []
    for s in SC:
        r = cm[s][i]["instruments"]["bfi_l1"]; g = r["hedges_g"]
        dag = r"^{\dagger}" if m.get("alias") == "qvl" else ""
        cells += [f"${r['delta']:+.2f}$", f"${g:+.2f}{dag}$" if g is not None else "--", p_sub(r["welch_p"]) + (r"\,\yes" if r["welch_p"] < .05 else "")]
    if m.get("alias"):
        prim = [cm[s][0]["instruments"]["bfi_l1"]["delta"] for s in SC]
        dirs = [r"${\approx}0$" if abs(cm[s][i]["instruments"]["bfi_l1"]["delta"]) < .05 else (r"\yes" if cm[s][i]["instruments"]["bfi_l1"]["delta"] * pv > 0 else r"\no") for s, pv in zip(SC, prim)]
        cells.append(", ".join(dirs))
    else:
        cells.append("ref")
    crow.append(row(NAMES.get(m.get("alias", ""), m["name"]), *cells))
save("app_cross_c2c5.tex", crow)
nsj = rj(SUB / "analysis_no_srm_ablation.json")["scenarios"]
save("app_nosrm_agent.tex", [row(s.title(), f"${nsj[s]['condition_means']['C2']:.2f}$", f"${nsj[s]['condition_means']['C5']:.2f}$",
                                  f"${nsj[s]['delta']:+.2f}$ (${nsj[s]['hedges_g']:+.2f}$)",
                                  ci(nsj[s]["ci95_lo"], nsj[s]["ci95_hi"], 2) + ", $p{=}" + f"{nsj[s]['welch_p']:.2g}".replace("e-0", r"\times10^{-") + "}$" if nsj[s]['welch_p'] < 1e-3 else ci(nsj[s]["ci95_lo"], nsj[s]["ci95_hi"], 2) + ", $p{=}" + f"{nsj[s]['welch_p']:.4f}" + "$") for s in SC])
print("tables written")
