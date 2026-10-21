#!/usr/bin/env python3
"""Belief-probe wording separation over rounds (evidence_cr/belief_time_course.csv)."""
import csv, random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from style import plt, COND, MARK, LABEL, TEXTWIDTH, panel, save

ROOT = Path(__file__).resolve().parents[1]
rows = [r for r in csv.DictReader(open(ROOT / "evidence_cr/belief_time_course.csv")) if r["source"] == "belief"]
data = defaultdict(list)
for r in rows: data[(r["scenario"], r["condition"], int(r["round"]))].append(float(r["wording_separation"]))


def band(v, draws=5000, seed=7):
    rng = random.Random(seed); m = sorted(mean(rng.choices(v, k=len(v))) for _ in range(draws))
    return m[int(.025 * draws)], m[int(.975 * draws) - 1]


fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.05), sharey=True)
rounds = [5, 10, 15, 20, 25, 30]
for ax, sc, letter in zip(axes, ["greenfield", "riverton"], "ab"):
    for c in ["C1", "C3", "C6", "C2"]:
        ys = [mean(data[(sc, c, t)]) for t in rounds]; bs = [band(data[(sc, c, t)]) for t in rounds]
        ax.fill_between(rounds, [b[0] for b in bs], [b[1] for b in bs], color=COND[c], alpha=.15, lw=0)
        ax.plot(rounds, ys, color=COND[c], marker=MARK[c], ms=4, lw=2.2 if c == "C2" else 1.4, label=LABEL[c])
    ax.axhline(0, color="#999999", lw=.8, ls="--")
    ax.set_xticks(rounds); ax.set_xlabel("Interaction round")
    panel(ax, letter, sc.title())
axes[0].set_ylabel("Within − between\nwording overlap")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.02), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(rect=(0, .1, 1, 1))
save(fig, "time_course.pdf"); save(fig, "time_course.png")
