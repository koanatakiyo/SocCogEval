#!/usr/bin/env python3
"""Regenerate LaTeX table fragments from bundled audit and secondary data."""
from pathlib import Path
import csv,json,re
paper=Path(__file__).resolve().parent; ev=paper/'evidence'
def rows(name):return list(csv.DictReader((ev/name).open()))
def save(name,lines): (paper/'tables'/name).write_text('\n'.join(lines)+'\n')
def row(*items):return ' & '.join(map(str,items))+r' \\'
def s(x): return f'{float(x):+.3f}'
def ci(a,b):return f'$[{s(a)},{s(b)}]$'
a=[];rv=[]
for r in rows('c6_decomposition.csv'):
 for c in ['c2_minus_c4','c6_minus_c4','c2_minus_c6']:
  p=float(r[c+'_exact_p']);a.append(row(r['scenario'].title(),'$'+c.replace('_minus_','-').upper().replace('C','C_')+'$',s(r[c]),ci(r[c+'_ci_lo'],r[c+'_ci_hi']),f'{p:.4f}'))
 for c in ['c2','c4','c6']:
  rv.append(row(r['scenario'].title(),'$'+c.upper().replace('C','C_')+'$',', '.join(f'{float(v):.4f}' for v in r[c+'_run_values'].split(';'))))
save('primary_rows.tex',a);save('run_rows.tex',rv)
NAME={'DeepSeek':'DeepSeek-V3'}
save('cross_rows.tex',[row(NAME.get(r['variant'],r['variant']),s(r['greenfield_delta']),s(r['riverton_delta']),s(r['scenario_averaged_delta'])) for r in rows('cross_variant_c4_variant_summary.csv')])
save('signed_rows.tex',[row(r['scenario'].title(),r['factor'].capitalize(),s(r['kappa_c2_minus_c4']),ci(r['descriptive_boot_ci_lo'],r['descriptive_boot_ci_hi']),f"{r['c2_direction_agreement_n']}/{r['c2_direction_agreement_denom']}") for r in rows('signed_factor_coherence.csv')])
ic=rows('variance_icc_design_effect.csv')
save('icc_rows.tex',[row(r['scenario'].title(), '$'+r['condition'].replace('C','C_')+'$', f"{float(r['icc1_raw']):.3f}", f"{float(r['design_effect']):.3f}", r['nominal_n'], f"{float(r['effective_n']):.1f}") for r in ic])
secondary=[];allocation=[]
for scenario in ['greenfield','riverton']:
 d=json.load(open(ev/f'analysis_p4_causal_decomposition_{scenario}.json'))
 v=d['contrasts']['(2) − (4)']
 for k,label in [('part2_conformity','Conformity'),('part2_consistency','Wording consistency')]:
  r=v[k];secondary.append(row(scenario.title(),label,s(r['delta']),ci(r['ci95_lo'],r['ci95_hi'])))
 d=json.load(open(ev/f'analysis_part1_contrasts_{scenario}.json'))
 for r in d['contrasts']['C2 − C4']['results']:
  allocation.append(row(scenario.title(),r['area'],f"{r['delta']:+.2f}",f"$[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]$"))
save('secondary_rows.tex',secondary);save('allocation_rows.tex',allocation)
