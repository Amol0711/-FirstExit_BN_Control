#!/usr/bin/env python3
"""Generate the complete R1/R2 decimal-weight recovery sweep and numerical diagnostics."""
from __future__ import annotations
import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Sequence
from boolean_networks import make_cases, phenotype9_case, recovery_evaluations, storage_summary_row, write_tsv
WEIGHT_GRID = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
BASE_CASES = ('R1', 'R2')

def _num(v: float) -> str:
    if math.isinf(v):
        return 'inf'
    return f'{v:.12g}'

def compression_profile_rows() -> List[Dict[str, object]]:
    cases = make_cases()
    profile_cases = [('V1', cases['V1']), ('V2', cases['V2']), ('V3', cases['V3']), ('V4', cases['V4']), ('R1', cases['R1']), ('R2', cases['R2']), ('P9r1', phenotype9_case(flip_nodes=tuple(range(1, 10)), max_flip_size=1)), ('P9r2', phenotype9_case(flip_nodes=tuple(range(1, 10)), max_flip_size=2))]
    rows: List[Dict[str, object]] = []
    for case_idx, (label, case) in enumerate(profile_cases, start=1):
        row = storage_summary_row(case)
        rows.append({'case_idx': case_idx, 'case': label, 'N_B': row['N_B'], 'A': row['A'], 'E_B': row['E_B'], 'raw_edges': row['raw_sweep'], 'compressed_edges': row['compressed_sweep'], 'compression_factor': row['compression_factor']})
    return rows

def recovery_sweep_rows() -> List[Dict[str, object]]:
    cases = make_cases()
    rows: List[Dict[str, object]] = []
    point_id = 0
    for base_name in BASE_CASES:
        base = cases[base_name]
        for w1 in WEIGHT_GRID:
            for w2 in WEIGHT_GRID:
                weighted = base.with_weights({1: w1, 2: w2}, name=base_name)
                for d in recovery_evaluations(weighted):
                    point_id += 1
                    loss_positive = not d.map_is_deterministic_optimal
                    if d.exact_transfer_condition_met:
                        category = 'condition_met_exact'
                    elif d.map_is_deterministic_optimal:
                        category = 'condition_not_met_exact'
                    else:
                        category = 'bounded_loss'
                    rows.append({'id': point_id, 'case': base_name, 'state': d.state, 'state_delta': d.state_delta, 'w1': _num(w1), 'w2': _num(w2), 'Delta': _num(d.Delta), 'epsilon': _num(d.epsilon), 'two_epsilon': _num(d.two_epsilon), 'local_loss': _num(max(0.0, d.local_loss)), 'loss_bound_slack': _num(d.two_epsilon - d.local_loss), 'exact_transfer_condition_met': 1 if d.exact_transfer_condition_met else 0, 'map_is_deterministic_optimal': 1 if d.map_is_deterministic_optimal else 0, 'local_loss_positive': 1 if loss_positive else 0, 'category': category, 'compression_factor_state': _num(d.compression_factor_state), 'y_D_star': d.deterministic_successor, 'y_hat': d.map_successor})
    return rows

def filter_rows(rows: Sequence[Mapping[str, object]], *, category: str | None=None, loss_positive: bool | None=None) -> List[Mapping[str, object]]:
    out: List[Mapping[str, object]] = []
    for row in rows:
        if category is not None and row['category'] != category:
            continue
        if loss_positive is not None and bool(int(row['local_loss_positive'])) != loss_positive:
            continue
        out.append(row)
    return out

def make_summary(rows: Sequence[Mapping[str, object]], compression_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    total = len(rows)
    condition_met = sum((int(r['exact_transfer_condition_met']) for r in rows))
    map_exact = sum((int(r['map_is_deterministic_optimal']) for r in rows))
    uncert_exact = sum((1 for r in rows if not int(r['exact_transfer_condition_met']) and int(r['map_is_deterministic_optimal'])))
    bounded = total - map_exact
    max_violation = max((float(r['local_loss']) - float(r['two_epsilon']) for r in rows)) if rows else 0.0
    min_slack = min((float(r['loss_bound_slack']) for r in rows)) if rows else 0.0
    max_loss_ratio = max((float(r['local_loss']) / float(r['two_epsilon']) for r in rows if float(r['two_epsilon']) > 0)) if rows else 0.0
    max_compression = max((float(r['compression_factor']) for r in compression_rows)) if compression_rows else 0.0
    return {'weight_grid_size': len(WEIGHT_GRID), 'finite_gap_points': sum((math.isfinite(float(r['Delta'])) for r in rows)), 'infinite_gap_points': sum((math.isinf(float(r['Delta'])) for r in rows)), 'strict_gap_comparison_tolerance': '1e-9', 'deterministic_cost_arithmetic': 'exact_decimal_fraction', 'total_state_weight_points': total, 'condition_met_exact_points': condition_met, 'map_deterministic_optimal_points': map_exact, 'condition_not_met_but_exact_points': uncert_exact, 'bounded_loss_points': bounded, 'loss_bound_violations': 1 if max_violation > 1e-08 else 0, 'condition_met_transfer_violations': sum((1 for r in rows if int(r['exact_transfer_condition_met']) and (not int(r['map_is_deterministic_optimal'])))), 'max_local_loss_minus_2epsilon': _num(max_violation), 'min_loss_bound_slack': _num(min_slack), 'max_local_loss_over_2epsilon': _num(max_loss_ratio), 'max_case_compression_factor': _num(max_compression)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='Directory for regenerated outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    compression = compression_profile_rows()
    rows = recovery_sweep_rows()
    summary = make_summary(rows, compression)
    fieldnames = ['id', 'case', 'state', 'state_delta', 'w1', 'w2', 'Delta', 'epsilon', 'two_epsilon', 'local_loss', 'loss_bound_slack', 'exact_transfer_condition_met', 'map_is_deterministic_optimal', 'local_loss_positive', 'category', 'compression_factor_state', 'y_D_star', 'y_hat']
    files: List[Path] = []
    outputs = {'recovery_conditions.tsv': (rows, fieldnames), 'recovery_compression_profile.tsv': (compression, ['case_idx', 'case', 'N_B', 'A', 'E_B', 'raw_edges', 'compressed_edges', 'compression_factor']), 'recovery_condition_summary.tsv': ([summary], list(summary.keys())), 'recovery_gap_condition_met.tsv': (filter_rows(rows, category='condition_met_exact'), fieldnames), 'recovery_gap_condition_not_met_exact.tsv': (filter_rows(rows, category='condition_not_met_exact'), fieldnames), 'recovery_gap_bounded_loss.tsv': (filter_rows(rows, category='bounded_loss'), fieldnames), 'recovery_loss_zero.tsv': (filter_rows(rows, loss_positive=False), fieldnames), 'recovery_loss_positive.tsv': (filter_rows(rows, loss_positive=True), fieldnames), 'recovery_loss_optimal.tsv': ([r for r in rows if int(r['map_is_deterministic_optimal'])], fieldnames), 'recovery_loss_bounded.tsv': ([r for r in rows if not int(r['map_is_deterministic_optimal'])], fieldnames)}
    outputs['recovery_gap_infinite.tsv'] = ([r for r in rows if math.isinf(float(r['Delta']))], fieldnames)
    outputs['recovery_gap_condition_met_finite.tsv'] = ([r for r in filter_rows(rows, category='condition_met_exact') if math.isfinite(float(r['Delta']))], fieldnames)
    for name, (data, fields) in outputs.items():
        path = args.output / name
        write_tsv(path, data, fields)
        files.append(path)
    check_path = args.output / 'recovery_condition_checks.txt'
    with check_path.open('w') as f:
        f.write('Mode-recovery condition-sweep\n')
        f.write('Exact decimal-input deterministic gaps/optimality; numerical KL mismatch and modes.\n')
        f.write('Gap test uses Delta > 2*epsilon + 1e-9, not interval-certified arithmetic.\n')
        f.write(f'weight grid: {WEIGHT_GRID}\n')
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')
    files.append(check_path)
    if int(summary['loss_bound_violations']) or int(summary['condition_met_transfer_violations']):
        raise RuntimeError(f'Recovery-condition checks failed: {summary}')
    print(f"Recovery condition-sweep generated: {summary['total_state_weight_points']} state-weight points, {summary['condition_met_exact_points']} condition met and exact, {summary['condition_not_met_but_exact_points']} condition-not-met exact, {summary['bounded_loss_points']} bounded-suboptimality, suboptimality-bound violations {summary['loss_bound_violations']}")
if __name__ == '__main__':
    main()
