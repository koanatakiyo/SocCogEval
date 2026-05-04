"""analysis_c5_temp_sweep.py — Soft noise-floor sanity check for C5.

Context: C5 (NoSeed+NoInteract) at the production temperature=0.0 yielded
byte-identical allocations across all 8 agents × 3 reps, because greedy
decoding on a format-constrained allocation task is robust to the agent-name
perturbation in the system prompt. That zero floor is real but specific to
deterministic decoding.

This script re-probes C5-like agents (no seed, no memory) at temperatures
0.0, 0.3, 0.7 and reports per-temperature per-rep null divergence. The
output tells us what the noise floor looks like under sampling — i.e., how
much of the C3 effect could possibly be attributed to decoding stochasticity
rather than interaction-induced state divergence.

Usage:
    python scripts/analysis_c5_temp_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cogclaw import config as cfg
from cogclaw.agent import AGENT_NAMES, Agent
from cogclaw.probes import _parse_allocations
from scenarios.greenfield import CONTEXT as GREENFIELD_CONTEXT

TEMPERATURES = [0.0, 0.3, 0.7]
N_REPS = 3
N_AGENTS = 8
N_PERM_NULL = 1_000
SEED = 42
AREAS = ["Education", "Healthcare", "Green Spaces", "Infrastructure", "Economic Development"]
OUT = Path("/home/yandan/cogsociety/data/analysis_c5_temp_sweep.json")


def build_allocation_prompt() -> str:
    """Mirror probes.allocation_task prompt (probes.py:325-333)."""
    area_list = "\n".join(f"  - {a}" for a in AREAS)
    return (
        "You have 100 percentage points to distribute across the following "
        "five areas based on how important you believe each one is.\n\n"
        f"{area_list}\n\n"
        "Allocate exactly 100 points total.  Respond in this exact format, "
        "one area per line:\n"
        + "\n".join(f"  {a}: [number]%" for a in AREAS)
        + "\n\nProvide only the allocations, no other commentary."
    )


def fresh_agents() -> list[Agent]:
    """Build 8 identically initialized agents — no seed, no memory, Greenfield."""
    return [
        Agent(
            agent_id=i,
            name=AGENT_NAMES[i],
            scenario_context=GREENFIELD_CONTEXT,
            model=cfg.PILOT_MODEL,
            seed_info="",
        )
        for i in range(N_AGENTS)
    ]


def run_one_rep(agents: list[Agent], prompt: str, temperature: float) -> tuple[np.ndarray, list[str]]:
    """Call the allocation probe on each agent at the given temperature.
    Returns (X shape 8×5, list of 8 raw strings)."""
    rows, raws = [], []
    for a in agents:
        raw = a.answer_probe(prompt, max_tokens=cfg.ALLOCATION_MAX_TOKENS, temperature=temperature)
        allocs = _parse_allocations(raw, AREAS)
        row = [float(allocs.get(area, 0.0)) for area in AREAS]
        rows.append(row)
        raws.append(raw)
    return np.asarray(rows, dtype=float), raws


def null_divergence_mean(X: np.ndarray, rng: np.random.Generator, n_perm=N_PERM_NULL) -> np.ndarray:
    n = X.shape[0]
    half = n // 2
    accum = np.zeros(X.shape[1])
    idx = np.arange(n)
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        accum += np.abs(X[perm[:half]].mean(0) - X[perm[half : 2 * half]].mean(0))
    return accum / n_perm


def main():
    rng = np.random.default_rng(SEED)
    prompt = build_allocation_prompt()
    print("Prompt template:")
    print("─" * 70)
    print(prompt)
    print("─" * 70)
    print(f"\nModel: {cfg.PILOT_MODEL}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Reps per temperature: {N_REPS}")
    print(f"Agents per rep: {N_AGENTS}")
    print(f"Total LLM calls: {len(TEMPERATURES) * N_REPS * N_AGENTS}")
    print()

    results = {"temperatures": {}}

    t0 = time.time()
    for temp in TEMPERATURES:
        print(f"\n{'=' * 70}")
        print(f"Temperature = {temp}")
        print("=" * 70)
        per_rep_null = []
        per_rep_raw_uniqueness = []
        per_rep_raws = []
        for rep in range(N_REPS):
            agents = fresh_agents()  # fresh no-memory agents each rep
            X, raws = run_one_rep(agents, prompt, temp)
            null_mean = null_divergence_mean(X, rng)
            per_rep_null.append(null_mean)
            n_distinct = len(set(raws))
            per_rep_raw_uniqueness.append(n_distinct)
            per_rep_raws.append(raws)
            # print per-agent allocations
            print(f"\n  rep {rep}: distinct raw strings = {n_distinct}/{N_AGENTS}")
            for i, (name, row) in enumerate(zip(AGENT_NAMES, X)):
                print(f"    {name:8s}: " + "  ".join(f"{a[:4]}={v:5.1f}" for a, v in zip(AREAS, row)))
            print(
                f"    → null divergence per dim: "
                + "  ".join(f"{a[:4]}={v:5.2f}" for a, v in zip(AREAS, null_mean))
            )
        per_rep_null = np.vstack(per_rep_null)
        temp_mean = per_rep_null.mean(0)
        temp_std = per_rep_null.std(0, ddof=1) if N_REPS > 1 else np.zeros_like(temp_mean)
        overall = float(temp_mean.mean())
        print(f"\n  SUMMARY @ T={temp}:")
        print(f"    per-dim mean null divergence: " + "  ".join(f"{a[:4]}={v:5.2f}" for a, v in zip(AREAS, temp_mean)))
        print(f"    overall mean (avg over 5 dims): {overall:.2f}")
        print(f"    avg raw-string uniqueness:      {np.mean(per_rep_raw_uniqueness):.1f}/{N_AGENTS}")
        results["temperatures"][str(temp)] = {
            "per_rep_null": per_rep_null.tolist(),
            "per_dim_mean": temp_mean.tolist(),
            "per_dim_std": temp_std.tolist(),
            "overall_mean": overall,
            "avg_raw_uniqueness": float(np.mean(per_rep_raw_uniqueness)),
            "per_rep_raws": per_rep_raws,
        }

    dt = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total wall-clock: {dt:.0f}s ({dt/60:.1f} min)")

    print("\n=== COMPARISON TABLE ===")
    # Load C3 effect from analysis output if available, otherwise use placeholder
    c3_json = Path("/home/yandan/cogsociety/data/analysis_c3_vs_c5_greenfield.json")
    if c3_json.exists():
        c3_data = json.load(open(c3_json))
        c3_mean_per_dim = [r["c3_mean"] for r in c3_data["results"]]
        print(f"(C3 effect loaded from {c3_json.name})")
    else:
        c3_mean_per_dim = [2.28, 3.07, 3.48, 2.98, 2.96]
        print("(WARNING: using hardcoded C3 effect — rerun analysis_c3_vs_c5.py to update)")
    c3_overall = float(np.mean(c3_mean_per_dim))
    print(f"{'Temperature':<14}{'Overall null':>14}{'C3 effect':>12}{'Ratio C3/null':>16}")
    for temp in TEMPERATURES:
        r = results["temperatures"][str(temp)]
        o = r["overall_mean"]
        ratio = f"{c3_overall/o:.1f}×" if o > 0.01 else "∞"
        print(f"T={temp:<11}{o:>14.2f}{c3_overall:>12.2f}{ratio:>16}")
    print()
    print("Interpretation:")
    print("  - T=0.0 should reproduce the production C5 ≈ 0 result.")
    print("  - T=0.3/0.7 gives the soft noise floor under sampling.")
    print("  - If C3 effect dominates the soft floor by ≥5×, the contrast is robust.")
    print("  - If the T=0.7 floor approaches C3 magnitude, interaction signal is weaker than claimed.")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
