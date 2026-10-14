#!/usr/bin/env python3
"""Generate exactness and successor-compression numerical tables for V1--V4."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Mapping
from boolean_networks import exactness_summary_row, make_cases, write_tsv
CASE_ORDER = ['V1', 'V2', 'V3', 'V4']
TARGET_KIND = {'V1': 'fixed', 'V2': 'fixed', 'V3': '2-cycle', 'V4': 'fixed'}
TABLE_FIELDNAMES = ['case', 'target_kind', 'N_B', 'N_transient', 'A', 'E_B', 'NB_A_over_EB', 'max_value_error', 'MAP_exact_states', 'gap_condition_met_states']

def _compact_factor(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return 'inf'
    if abs(f - round(f)) < 1e-10:
        return str(int(round(f)))
    if f < 2:
        return f'{f:.2f}'
    return f'{f:.1f}'

def build_rows() -> List[Dict[str, object]]:
    cases = make_cases()
    rows: List[Dict[str, object]] = []
    for name in CASE_ORDER:
        case = cases[name]
        if not case.target_set or any((case.update(t) not in case.target_set for t in case.target)):
            raise ValueError(f'{name}: target must be nonempty and invariant under the uncontrolled update')
        row = exactness_summary_row(case)
        row = dict(row)
        row['target_kind'] = TARGET_KIND[name]
        row['NB_A_over_EB'] = _compact_factor(row['NB_A_over_EB'])
        rows.append(row)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='Directory for regenerated TSV outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    root = here.parent
    rows = build_rows()
    write_tsv(args.output / 'exactness_table.tsv', rows, TABLE_FIELDNAMES)
    total_states = sum((int(row['N_transient']) for row in rows))
    total_map_exact = sum((int(str(row['MAP_exact_states']).split('/')[0]) for row in rows))
    max_err = max((float(row['max_value_error']) for row in rows))
    totals = [{'cases': len(rows), 'transient_states': total_states, 'max_value_error': f'{max_err:.3g}', 'MAP_exact_states': f'{total_map_exact}/{total_states}'}]
    write_tsv(args.output / 'exactness_totals.tsv', totals, list(totals[0].keys()))
    if max_err > 1e-09:
        raise RuntimeError(f'Compressed/raw value mismatch too large: {max_err}')
    if total_map_exact != total_states:
        raise RuntimeError(f'Unexpected MAP non-optimal states: {total_map_exact}/{total_states}')
    print(f'Exactness table generated: {len(rows)} cases, value error {max_err:.3g}, MAP exact {total_map_exact}/{total_states}', flush=True)
if __name__ == '__main__':
    main()
