#!/usr/bin/env python3
"""Recompute the per-model (C2)-(C4) tests (paper Appendix E.1, Table 12), offline.

Each of five API-served variants has 8 C2 and 5 C4 runs per scenario. The protocol is in
docs/protocols/per_model_tests.md. The audited table is results/reference/per_model_c2_c4.csv;
outputs are written under results/reproduced/. No model calls or credentials are needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_paper import AGENTS, FACTORS, assert_close, exact_test, score_bfi  # noqa: E402

VARIANTS = ['gpt', 'l4api', 'dsf', 'sdm', 'glm47']
SCENARIOS = ['greenfield', 'riverton']


def bootstrap(a, b, seed=20261020, draws=50_000):
    """Run-bootstrap 95% interval for mean(a) - mean(b), as in the protocol."""
    rng = random.Random(seed)
    diffs = sorted(mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b)))
                   for _ in range(draws))
    return diffs[int(0.025*draws)], diffs[int(0.975*draws)-1]


def verify_files():
    for row in json.loads((ROOT/'data/per_model_checksums.json').read_text()):
        if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f"Measurement checksum mismatch: {row['path']}")


def run_l1(run_id):
    """Mean over the eight agents of the BFI-10 L1 pre/post movement, from raw ratings."""
    scores = []
    for phase in ['pre', 'post']:
        rows = json.loads((ROOT/'data'/run_id/f'{phase}_measurements.json').read_text())['bfi10']
        by_name = {r['agent_name']: r for r in rows}
        if len(rows) != 8 or set(by_name) != set(AGENTS):
            raise ValueError(f'Incomplete or duplicate BFI agent rows: {run_id}/{phase}')
        phase_scores = {}
        for agent in AGENTS:
            computed = score_bfi(by_name[agent]['raw_responses'])
            for value, factor in zip(computed, FACTORS):
                assert_close(value, by_name[agent]['scores'][factor], f'{run_id}/{phase}/{agent}')
            phase_scores[agent] = computed
        scores.append(phase_scores)
    return mean(sum(abs(b-a) for a, b in zip(scores[0][g], scores[1][g])) for g in AGENTS)


def reproduce(out):
    verify_files()
    selection = json.loads((ROOT/'data/per_model_selection.json').read_text())
    reference = {(r['variant'], r['scenario']): r
                 for r in csv.DictReader((ROOT/'results/reference/per_model_c2_c4.csv').open())}
    rows = []
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            cells = {c: sorted(k for k, v in selection.items() if v['variant'] == variant and
                               v['scenario'] == scenario and v['condition'] == c) for c in ['C2', 'C4']}
            if [len(cells['C2']), len(cells['C4'])] != [8, 5]:
                raise ValueError(f'{variant}/{scenario} needs 8 C2 and 5 C4 runs')
            ref = reference[variant, scenario]
            # Keep the reference run order so the bootstrap draws match.
            c2 = [run_l1(r) for r in ref['c2_runs'].split(';')]
            c4 = [run_l1(r) for r in ref['c4_runs'].split(';')]
            if sorted(ref['c2_runs'].split(';')) != cells['C2'] or sorted(ref['c4_runs'].split(';')) != cells['C4']:
                raise ValueError(f'{variant}/{scenario}: selection and reference runs differ')
            delta, p, n = exact_test(c2, c4)
            lo, hi = bootstrap(c2, c4)
            label = f'{variant}/{scenario}'
            for value, key in [(mean(c2), 'c2_mean'), (mean(c4), 'c4_mean'), (delta, 'delta_c2_minus_c4'),
                               (lo, 'ci_lo'), (hi, 'ci_hi'), (p, 'exact_p')]:
                assert_close(value, ref[key], f'{label}/{key}')
            rows.append(dict(variant=variant, model=ref['model'], scenario=scenario, n_c2=len(c2), n_c4=len(c4),
                             c2_mean=mean(c2), c4_mean=mean(c4), delta=delta, ci_lo=lo, ci_hi=hi,
                             exact_p=p, allocations=n))
    out.mkdir(parents=True, exist_ok=True)
    with (out/'per_model_c2_c4.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['model']:20s} {r['scenario']:10s} {r['n_c2']}/{r['n_c4']}  C4 {r['c4_mean']:.2f}  "
              f"C2 {r['c2_mean']:.2f}  delta {r['delta']:+.3f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  p {r['exact_p']:.4f}")
    print(f'{len(selection)} runs, checksums and reference comparisons passed.')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/reproduced')
    reproduce(parser.parse_args().out)
