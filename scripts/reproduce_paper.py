#!/usr/bin/env python3
"""Recompute run-level paper estimates from the bundled measurements, offline.

Original audited tables are immutable in results/reference/. Outputs are written
under results/reproduced/. No model calls, credentials, or checkpoints are needed.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
AGENTS = ['Alex', 'Blake', 'Casey', 'Dana', 'Ellis', 'Finley', 'Gray', 'Harper']


def score_bfi(ratings):
    if len(ratings) != 10 or any(x not in (1, 2, 3, 4, 5) for x in ratings):
        raise ValueError('BFI requires ten integer ratings from 1 through 5')
    r = [6-x if i in (0, 2, 3, 4, 6) else x for i, x in enumerate(ratings)]
    return [(r[i]+r[i+5])/2 for i in range(5)]


def exact_test(a, b):
    a, b = list(a), list(b)
    if not a or not b:
        raise ValueError('Both conditions must contain runs')
    pooled = np.array(a+b, dtype=float)
    observed = float(np.mean(a)-np.mean(b))
    extreme = count = 0
    for indices in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(indices)] = True
        statistic = pooled[mask].mean()-pooled[~mask].mean()
        extreme += abs(statistic) >= abs(observed)-1e-12
        count += 1
    return observed, float(extreme/count), count


def bootstrap(a, b, seed, draws=50_000):
    # Interleave the two draws to reproduce the final C6 audit RNG sequence.
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean()-rng.choice(b, len(b)).mean()
             for _ in range(draws)]
    return [float(x) for x in np.percentile(diffs, [2.5, 97.5])]


def read_csv(name):
    return list(csv.DictReader((ROOT/'results/reference'/name).open()))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_files():
    for row in json.loads((ROOT/'data/measurement_checksums.json').read_text()):
        actual = hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise ValueError(f"Measurement checksum mismatch: {row['path']}")


def load_runs():
    selection = json.loads((ROOT/'data/run_selection.json').read_text())
    runs = []
    for name, info in selection.items():
        states = []
        for phase in ['pre', 'post']:
            payload = json.loads((ROOT/'data'/name/f'{phase}_measurements.json').read_text())
            rows = payload['bfi10']
            by_name = {r['agent_name']: r for r in rows}
            if len(rows) != 8 or set(by_name) != set(AGENTS):
                raise ValueError(f'Incomplete or duplicate BFI agent rows: {name}/{phase}')
            scores = []
            for agent in AGENTS:
                r = by_name[agent]
                computed = score_bfi(r['raw_responses'])
                stored = [r['scores'][f] for f in FACTORS]
                if not np.allclose(computed, stored):
                    raise ValueError(f'BFI reverse-scoring mismatch: {name}/{phase}/{agent}')
                scores.append(computed)
            states.append(np.array(scores))
        movement = states[1]-states[0]
        runs.append(dict(info, run_id=name, l1=float(np.abs(movement).sum(1).mean()),
                         signed=movement.mean(0), agent_l1=np.abs(movement).sum(1)))
    return runs


def assert_close(a, b, label):
    if not np.isclose(float(a), float(b), atol=1e-10, rtol=0):
        raise ValueError(f'Reference mismatch for {label}: {a} versus {b}')


def reproduce(out):
    verify_files()
    runs = load_runs()
    def cell(scenario, condition, variant='primary', memory='srm'):
        return [r for r in runs if r['scenario'] == scenario and
                r['condition'] == condition and r['variant'] == variant and
                r['memory_mode'] == memory]
    primary = []
    signed = []
    no_srm = []
    for si, scenario in enumerate(['greenfield', 'riverton']):
        cells = {c: cell(scenario, c) for c in ['C2', 'C4', 'C6']}
        if [len(cells[c]) for c in cells] != [8, 5, 5]:
            raise ValueError('Primary run counts must be 8/5/5')
        reference = next(r for r in read_csv('c6_decomposition.csv') if r['scenario'] == scenario)
        for ci, (a, b) in enumerate([('C2', 'C4'), ('C6', 'C4'), ('C2', 'C6')]):
            av, bv = [r['l1'] for r in cells[a]], [r['l1'] for r in cells[b]]
            delta, p, n = exact_test(av, bv)
            lo, hi = bootstrap(av, bv, 20260728+si*10+ci)
            prefix = f'{a.lower()}_minus_{b.lower()}'
            for key, value in [('', delta), ('_exact_p', p), ('_ci_lo', lo), ('_ci_hi', hi)]:
                assert_close(value, reference[prefix+key], f'{scenario}/{prefix}{key}')
            primary.append(dict(scenario=scenario, contrast=f'{a}-{b}', n_a=len(av), n_b=len(bv),
                                a_mean=np.mean(av), b_mean=np.mean(bv), delta=delta,
                                ci_lo=lo, ci_hi=hi, exact_p=p, allocations=n))
        for fi, factor in enumerate(FACTORS):
            av, bv = [r['signed'][fi] for r in cells['C2']], [r['signed'][fi] for r in cells['C4']]
            kappa = float(np.mean(av)-np.mean(bv))
            agreement = sum((v-np.mean(bv))*kappa > 0 for v in av)
            reference_factor = next(r for r in read_csv('signed_factor_coherence.csv')
                                    if r['scenario'] == scenario and r['factor'] == factor)
            assert_close(kappa, reference_factor['kappa_c2_minus_c4'], factor)
            assert_close(agreement, reference_factor['c2_direction_agreement_n'], factor+' agreement')
            signed.append(dict(scenario=scenario, factor=factor, kappa=kappa,
                               agreement_n=agreement, n_runs=len(av)))
        av, bv = [[r['l1'] for r in cell(scenario, c, memory='verbatim')] for c in ['C2', 'C5']]
        delta, p, n = exact_test(av, bv)
        assert_close(p, .1, 'no-SRM exact p')
        no_srm.append(dict(scenario=scenario, delta=delta, exact_p=p, allocations=n))
    cross = []
    for reference in read_csv('cross_variant_c4_results.csv'):
        alias, scenario = reference['alias'], reference['scenario']
        av, bv = [[r['l1'] for r in cell(scenario, c, variant=alias)] for c in ['C2', 'C4']]
        if len(av) != 3 or len(bv) != 3:
            raise ValueError('Every cross-variant cell requires three runs per condition')
        delta, p, n = exact_test(av, bv)
        assert_close(delta, reference['delta_c2_minus_c4'], alias+'/'+scenario)
        cross.append(dict(variant=reference['variant'], scenario=scenario,
                          delta=delta, exact_p=p, allocations=n))
    # T=0 values are a secondary probe of the same saved trajectories.
    t0 = json.loads((ROOT/'results/reference/bfi_t0_summary.json').read_text())
    temperature = []
    for scenario, values in t0.items():
        snapshot = json.loads((ROOT/f'results/reference/bfi_t0_{scenario}.json').read_text())
        cells = defaultdict(dict)
        for row in snapshot['rows']:
            pre = score_bfi(row['pre']['raw_responses'])
            post = score_bfi(row['post']['raw_responses'])
            for phase, scores in [('pre', pre), ('post', post)]:
                for fi, factor in enumerate(FACTORS):
                    assert_close(scores[fi], row[phase]['scores'][factor], 'T=0 score')
            l1 = sum(abs(a-b) for a, b in zip(post, pre))
            assert_close(l1, row['bfi_l1'], 'T=0 agent L1')
            key = row['condition'], row['replica']
            if row['agent_name'] in cells[key]:
                raise ValueError('Duplicate T=0 agent row')
            cells[key][row['agent_name']] = l1
        cell_means = {}
        for condition, expected_n in [('C2', 8), ('C4', 5)]:
            if sum(c == condition for c, _ in cells) != expected_n:
                raise ValueError('Unexpected T=0 run count')
            means = []
            for replica in range(expected_n):
                agents = cells[condition, replica]
                if set(agents) != set(AGENTS):
                    raise ValueError('Incomplete T=0 run')
                means.append(float(np.mean(list(agents.values()))))
            for actual, expected in zip(means, values[condition.lower()+'_l1_run_values']):
                assert_close(actual, expected, 'T=0 run mean')
            cell_means[condition] = means
        delta, p, n = exact_test(cell_means['C2'], cell_means['C4'])
        assert_close(delta, values['contrast']['c2_minus_c4'], 'T=0/'+scenario)
        temperature.append(dict(scenario=scenario, delta=delta, exact_p=p, allocations=n))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('primary_contrasts', primary), ('signed_factors', signed),
                       ('no_srm', no_srm), ('cross_variant', cross), ('temperature', temperature)]:
        write_csv(out/(name+'.csv'), rows)
    run_rows = [dict(run_id=r['run_id'], scenario=r['scenario'], condition=r['condition'],
                    variant=r['variant'], memory_mode=r['memory_mode'], bfi_l1=r['l1']) for r in runs]
    write_csv(out/'run_means.csv', run_rows)
    averages = {v: np.mean([r['delta'] for r in cross if r['variant'] == v])
                for v in {r['variant'] for r in cross}}
    positive = sum(int(v > 0) for v in averages.values())
    import math
    sign_p = sum(math.comb(len(averages), k) for k in range(positive, len(averages)+1)) / 2**len(averages)
    report = dict(runs=len(runs), measurement_files=4*len(runs), checksum_status='passed',
                  raw_bfi_scoring_status='passed', frozen_temperature_raw_scoring_status='passed',
                  reference_comparisons='passed',
                  bootstrap_draws=50000, bootstrap_seed_base=20260728,
                  cross_variant_positive_averages=positive, one_sided_sign_p=sign_p,
                  scope='Offline measurement reanalysis; original model trajectories are not regenerated.')
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/reproduced')
    reproduce(parser.parse_args().out)
