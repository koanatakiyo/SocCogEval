#!/usr/bin/env python3
"""Camera-ready analysis figures (Fig 4-7). Every number is read from evidence/, evidence_sub/
(submitted analysis outputs) or evidence_cr/ (camera-ready additions)."""
import csv, json
from pathlib import Path
from statistics import mean
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from style import plt, COND, MARK, SCEN, TEXTWIDTH, panel, save

ROOT = Path(__file__).resolve().parents[1]
EV, SUB, CR = ROOT / "evidence", ROOT / "evidence_sub", ROOT / "evidence_cr"
SC = ["greenfield", "riverton"]
SMARK = {"greenfield": "o", "riverton": "^"}


def rcsv(p): return list(csv.DictReader(open(p)))


def rj(p): return json.load(open(p))


def scen_legend(fig, y=-0.03, extra=()):
    h = [plt.Line2D([], [], color=SCEN[s], marker=SMARK[s], ls="", ms=5) for s in SC] + [e[0] for e in extra]
    fig.legend(h, ["Greenfield", "Riverton"] + [e[1] for e in extra], loc="lower center",
               ncol=2 + len(extra), frameon=False, bbox_to_anchor=(.5, y))


# ---------------- Fig 4: primary result ----------------
def fig4():
    means = rcsv(CR / "run_level_means.csv")
    c6 = {r["scenario"]: r for r in rcsv(EV / "c6_decomposition.csv")}
    con = {(r["scenario"], r["contrast"]): r for r in rcsv(CR / "run_level_contrasts.csv")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
    order = ["C5", "C4", "C1", "C3", "C6", "C2"]
    for si, s in enumerate(SC):
        for x, c in enumerate(order):
            v = [float(r["bfi_l1_run_mean"]) for r in means if r["scenario"] == s and r["condition"] == c]
            xp = x + (si - .5) * .32
            a.scatter(xp + np.linspace(-.08, .08, len(v)), v, s=12, color=COND[c], marker=SMARK[s],
                      edgecolor="white", linewidth=.3, zorder=3)
            a.plot([xp - .12, xp + .12], [mean(v)] * 2, color="#222222", lw=1.3, zorder=4)
    a.set_xticks(range(len(order)), order); a.set_ylabel("Run-mean BFI-10 L1 drift")
    a.set_ylim(0, 2.7); panel(a, "a", "Runs by condition")
    rows = [("C4-C5", "Brief alone"), ("C2-C4", "Interaction given brief"), ("C2-C3", "Brief given interaction"),
            ("C3-C5", "Interaction alone"), ("C2-C5", "Total"), ("C6-C4", "Solo given brief"), ("C2-C6", "Social vs. solo")]
    b.axhspan(len(rows) - 2 - .45, len(rows) - 2 + .45, color=COND["C2"], alpha=.08, lw=0)
    for i, (k, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        for si, s in enumerate(SC):
            if k in ("C2-C4", "C6-C4", "C2-C6"):  # rebuttal values as stated
                key = k.lower().replace("-", "_minus_"); r = c6[s]
                d, lo, hi = float(r[key]), float(r[key + "_ci_lo"]), float(r[key + "_ci_hi"])
            else:
                r = con[(s, k)]; d, lo, hi = float(r["delta"]), float(r["run_ci_lo"]), float(r["run_ci_hi"])
            yy = y + (.5 - si) * .3
            b.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt=SMARK[s], color=SCEN[s], ms=4, capsize=2, lw=1.2)
    b.axvline(0, color="#999999", lw=.8, ls="--")
    b.set_yticks(range(len(rows) - 1, -1, -1), [r[1] for r in rows]); b.grid(axis="y", visible=False)
    b.set_xlabel("Run-level contrast, BFI-10 L1"); panel(b, "b", "FSIP decomposition")
    b.text(1.33, len(rows) - 2, "+0.92\n+0.91", va="center", fontsize=8.5, color=COND["C2"], linespacing=.9)
    b.set_xlim(-1.05, 1.9)
    scen_legend(fig, y=-.04)
    fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, "fig4_primary.pdf")


# ---------------- Fig 5: temperature-zero re-probe ----------------
def fig5():
    t0 = rj(SUB / "bfi_t0_summary.json"); c6 = {r["scenario"]: r for r in rcsv(EV / "c6_decomposition.csv")}
    fig, ax = plt.subplots(figsize=(2.35, 1.55))
    for si, s in enumerate(SC):
        for ti, (lab, c2, c4) in enumerate([
                ("T=0.7", [float(v) for v in c6[s]["c2_run_values"].split(";")], [float(v) for v in c6[s]["c4_run_values"].split(";")]),
                ("T=0", t0[s]["c2_l1_run_values"], t0[s]["c4_l1_run_values"])]):
            x0 = si * 2.4 + ti
            for off, v, c in ((-.17, c4, "C4"), (.17, c2, "C2")):
                ax.scatter(x0 + off + np.linspace(-.05, .05, len(v)), v, s=10, color=COND[c], marker=SMARK[s], zorder=3, edgecolor="white", linewidth=.3)
                ax.plot([x0 + off - .09, x0 + off + .09], [mean(v)] * 2, color="#222222", lw=1.2, zorder=4)
    ax.set_xticks([0, 1, 2.4, 3.4], ["T=0.7", "T=0", "T=0.7", "T=0"])
    for si, s in enumerate(SC): ax.text(si * 2.4 + .5, 2.55, s.title(), ha="center", fontsize=9)
    ax.set_ylim(-.1, 2.85); ax.set_ylabel("Run-mean L1")
    h = [plt.Line2D([], [], color=COND[c], marker="o", ls="", ms=4) for c in ("C4", "C2")]
    fig.legend(h, ["C4 brief only", "C2 interaction"], frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(.55, -.02), handletextpad=.2, columnspacing=.8)
    fig.tight_layout(rect=(0, .1, 1, 1)); save(fig, "fig5_t0.pdf")


# ---------------- Fig 6: SCT route ----------------
def fig6():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.95), gridspec_kw={"width_ratios": [.8, 1.25, 1.25]})
    cons = [("C2 − C5", "C2−C5"), ("C2 − C4", "C2−C4"), ("C3 − C5", "C3−C5")]
    m = np.array([[rj(SUB / f"analysis_part1_contrasts_{s}.json")["contrasts"][k]["n_significant_dims"] for s in SC] for k, _ in cons], float)
    cmap = LinearSegmentedColormap.from_list("g", ["#F4F6F7", "#A9DFBF", "#2E86C1", "#1B4F72"])
    a.imshow(m, cmap=cmap, norm=Normalize(0, 5), aspect="auto")
    for i in range(3):
        for j in range(2):
            a.text(j, i, f"{int(m[i, j])}/5", ha="center", va="center", color="white" if m[i, j] >= 4 else "#222222", fontsize=9)
    a.set_yticks(range(3), [l for _, l in cons]); a.set_xticks(range(2), ["GF", "RV"]); a.grid(False)
    for sp in a.spines.values(): sp.set_visible(False)
    panel(a, "a", "Decision")
    a.set_xlabel("Dims clearing CI")
    for ci, cond in enumerate(["C2", "C3"]):
        for si, s in enumerate(SC):
            p = rj(SUB / f"analysis_p2_prediction_{cond.lower()}_{s}.json")
            reps = [it["out_target_mae"] - it["in_target_mae"] for it in p.get("per_rep_by_relation", []) if "out_target_mae" in it]
            y = (1 - si) + (.15 if cond == "C2" else -.15)
            b.scatter(reps, [y] * len(reps), s=9, color=COND[cond], alpha=.35, edgecolor="none")
            d = p["out_target_mae"]["mean"] - p["in_target_mae"]["mean"]
            b.scatter([d], [y], s=34, color=COND[cond], marker=MARK[cond], edgecolor="white", linewidth=.6, zorder=3,
                      label=cond if si == 0 else None)
    b.axvline(0, color="#999999", lw=.8, ls="--"); b.set_yticks([1, 0], ["GF", "RV"]); b.set_ylim(-.5, 1.5)
    b.set_xlabel("Out-group − in-group MAE"); b.grid(axis="y", visible=False); b.grid(axis="x", color="#e6e6e6", lw=.6)
    b.legend(frameon=False, loc="center right", ncol=2, handletextpad=.1, columnspacing=.6, borderaxespad=.1); panel(b, "b", "Prediction")
    sf = rcsv(EV / "signed_factor_coherence.csv"); fac = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    for si, s in enumerate(SC):
        for fi, f in enumerate(fac):
            r = next(r for r in sf if r["scenario"] == s and r["factor"] == f)
            k, lo, hi = float(r["kappa_c2_minus_c4"]), float(r["descriptive_boot_ci_lo"]), float(r["descriptive_boot_ci_hi"])
            y = 4 - fi + (.15 - .3 * si)
            c.errorbar(k, y, xerr=[[k - lo], [hi - k]], fmt=SMARK[s], color=SCEN[s], ms=4, capsize=2, lw=1.1)
    c.axvline(0, color="#999999", lw=.8, ls="--"); c.set_yticks(range(4, -1, -1), ["Extra.", "Agree.", "Consc.", "Neuro.", "Open."])
    c.set_xlabel("Signed κ, C2 − C4"); c.grid(axis="y", visible=False); panel(c, "c", "Transfer")
    scen_legend(fig, y=-.05)
    fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, "fig6_route.pdf")


# ---------------- Fig 7: breadth and dependence ----------------
NAMES = {"": "Q-122B", "qvl": "Q-VL", "dpsk": "DS-V3", "gpt": "GPT-mini", "gem": "Gem-Flash", "l4api": "L4-Scout"}


def fig7():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.1), gridspec_kw={"width_ratios": [.95, 1.1, .95]})
    recs = {s: rj(SUB / f"analysis_cross_model_{s}.json")["models"] for s in SC}
    names = [NAMES.get(r.get("alias", ""), r["name"]) for r in recs["greenfield"]]
    m = np.array([[recs[s][i]["instruments"]["bfi_l1"]["delta"] for s in SC] for i in range(len(names))])
    st = np.array([[recs[s][i]["instruments"]["bfi_l1"]["ci_excludes_zero"] for s in SC] for i in range(len(names))])
    cmap = LinearSegmentedColormap.from_list("b", ["#F1948A", "#FFFFFF", "#A9DFBF", "#2E86C1", "#1B4F72"])
    a.imshow(m, cmap=cmap, norm=TwoSlopeNorm(vmin=-.3, vcenter=0, vmax=3.4), aspect="auto")
    for i in range(len(names)):
        for j in range(2):
            a.text(j, i, f"{m[i, j]:+.2f}" + ("*" if st[i, j] else ""), ha="center", va="center", fontsize=8.5,
                   color="white" if m[i, j] > 1.8 else "#222222", fontweight="bold" if st[i, j] else "normal")
    a.set_xticks([0, 1], ["GF", "RV"]); a.set_yticks(range(len(names)), names); a.grid(False)
    for sp in a.spines.values(): sp.set_visible(False)
    panel(a, "a", "Total, C2 − C5")
    cv = {r["alias"]: r for r in rcsv(EV / "cross_variant_c4_variant_summary.csv")}
    c6 = {r["scenario"]: r for r in rcsv(EV / "c6_decomposition.csv")}
    order = ["", "qvl", "dpsk", "gpt", "gem", "l4api"]
    for i, al in enumerate(order):
        y = len(order) - 1 - i
        vals = [float(c6[s]["c2_minus_c4"]) for s in SC] if al == "" else [float(cv[al][f"{s}_delta"]) for s in SC]
        b.plot(vals, [y, y], color="#cccccc", lw=1, zorder=1)
        for s, v in zip(SC, vals): b.scatter(v, y, color=SCEN[s], marker=SMARK[s], s=18, zorder=3)
        b.scatter(mean(vals), y, color="#222222", marker="|", s=60, zorder=4)
    b.axvline(0, color="#999999", lw=.8, ls="--"); b.set_yticks(range(len(order) - 1, -1, -1), ["" for _ in order]); b.set_ylim(-.5, len(order) - .5)
    b.set_xscale("symlog", linthresh=1); b.set_xticks([0, .5, 1, 3], ["0", "0.5", "1", "3"])
    b.set_xlabel("C2 − C4 (| = average)"); b.grid(axis="y", visible=False); panel(b, "b", "Pure interaction")
    icc = rcsv(EV / "variance_icc_design_effect.csv")
    labs = [f"{r['condition']}\n{'GF' if r['scenario'] == 'greenfield' else 'RV'}" for r in icc]
    de = [float(r["design_effect"]) for r in icc]
    cols = [COND[r["condition"]] for r in icc]
    c.bar(range(len(de)), de, color=cols, width=.62, edgecolor=[SCEN[r["scenario"]] for r in icc], linewidth=1.2,
          hatch=None)
    for i, r in enumerate(icc):
        c.text(i, de[i] / 2, f"{float(r['icc1_raw']):+.2f}", ha="center", va="center", rotation=90, fontsize=8.5, color="white")
    c.axhline(1, color="#999999", lw=.8, ls="--"); c.set_xticks(range(len(de)), labs)
    c.set_ylabel("1 + 7·ICC(1)"); c.set_ylim(0, 2.1); panel(c, "c", "Design effect")
    scen_legend(fig, y=-.04)
    fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, "fig7_breadth.pdf")


if __name__ == "__main__":
    for f in (fig4, fig5, fig6, fig7): f()
