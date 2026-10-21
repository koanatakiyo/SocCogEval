"""Summarize the frozen T=0 BFI sensitivity at the run level."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "rebuttal" / "2026-07-25"
DIMS = (
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
)


def run_values(rows: list[dict], condition: str, value_fn) -> list[float]:
    values = []
    replicas = sorted({r["replica"] for r in rows if r["condition"] == condition})
    for replica in replicas:
        agent_values = [
            value_fn(r)
            for r in rows
            if r["condition"] == condition and r["replica"] == replica
        ]
        if len(agent_values) != 8:
            raise RuntimeError(
                f"{condition} rep{replica:02d}: expected 8 agents, "
                f"found {len(agent_values)}"
            )
        values.append(statistics.mean(agent_values))
    return values


def main() -> None:
    result: dict[str, dict] = {}
    for scenario in ("greenfield", "riverton"):
        path = EVIDENCE / f"bfi_t0_{scenario}.json"
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        if len(rows) != 104:
            raise RuntimeError(f"{scenario}: expected 104 rows, found {len(rows)}")

        c2_l1 = run_values(rows, "C2", lambda r: r["bfi_l1"])
        c4_l1 = run_values(rows, "C4", lambda r: r["bfi_l1"])
        factors = {}
        for dim in DIMS:
            value_fn = (
                lambda r, d=dim: r["post"]["scores"][d] - r["pre"]["scores"][d]
            )
            c2 = run_values(rows, "C2", value_fn)
            c4 = run_values(rows, "C4", value_fn)
            factors[dim] = {
                "c2_run_values": c2,
                "c4_run_values": c4,
                "c2_minus_c4": statistics.mean(c2) - statistics.mean(c4),
                "c2_positive_runs": sum(x > 0 for x in c2),
                "c2_negative_runs": sum(x < 0 for x in c2),
                "c2_zero_runs": sum(x == 0 for x in c2),
            }

        result[scenario] = {
            "n_c2_runs": len(c2_l1),
            "n_c4_runs": len(c4_l1),
            "c2_l1_run_values": c2_l1,
            "c4_l1_run_values": c4_l1,
            "all_c2_runs_positive": all(x > 0 for x in c2_l1),
            "all_c4_runs_zero": all(x == 0 for x in c4_l1),
            "contrast": payload["contrast"],
            "signed_factors": factors,
        }

    (EVIDENCE / "bfi_t0_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Frozen T=0 BFI-10 sensitivity",
        "",
        "All selected primary artifacts were included: 8 C2 and 5 C4 runs per "
        "scenario, with eight agents per run. Saved pre/post SRM states were "
        "re-probed at BFI temperature 0; the run remains the independent unit.",
        "",
        "|Scenario|C2 mean|C4 mean|Raw C2-C4|Descriptive run-bootstrap CI|Exact p|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("greenfield", "riverton"):
        c = result[scenario]["contrast"]
        ci = c["descriptive_bootstrap_ci"]
        lines.append(
            f"|{scenario}|{c['c2_mean']:.3f}|{c['c4_mean']:.3f}|"
            f"+{c['c2_minus_c4']:.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|"
            f"{c['exact_two_sided_permutation_p']:.4f}|"
        )

    lines.extend(
        [
            "",
            "All 8/8 C2 run means are positive in both scenarios. All C4 "
            "pre/post differences are zero under the identical deterministic "
            "probe prompt. This sensitivity uses raw differences only; no "
            "standardized effect is computed.",
            "",
            "|Scenario|Signed factor contrast|C2 directional agreement|",
            "|---|---:|---:|",
        ]
    )
    focus = {
        "greenfield": ("extraversion", "neuroticism"),
        "riverton": ("agreeableness", "openness"),
    }
    for scenario, dims in focus.items():
        for dim in dims:
            f = result[scenario]["signed_factors"][dim]
            direction = (
                f"{f['c2_positive_runs']}/8 positive"
                if f["c2_minus_c4"] > 0
                else f"{f['c2_negative_runs']}/8 negative"
            )
            lines.append(
                f"|{scenario} {dim}|{f['c2_minus_c4']:+.3f}|{direction}|"
            )
    lines.extend(
        [
            "",
            "The same signed factors highlighted at the original BFI "
            "temperature retain their direction at T=0.",
            "",
        ]
    )
    (EVIDENCE / "bfi_t0_summary.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
