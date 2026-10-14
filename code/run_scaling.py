#!/usr/bin/env python3
"""Generate graph-storage and repeated-sweep operation counts. Synthetic cases are analytic counts, not measured large-network experiments."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
from boolean_networks import phenotype9_case, storage_summary_row, write_tsv
SYNTHETIC_CONFIGS = [(70, 8), (100, 10), (200, 12)]
MAX_SWEEPS = 30

def synthetic_storage_row(N_syn: int, m_ctrl: int, case_idx: int) -> Dict[str, object]:
    ell = N_syn - m_ctrl - 2
    N_B = ell
    A = 2 ** m_ctrl
    E_B = 2 * ell
    raw_edges = N_B * A
    compression_factor = raw_edges / E_B
    break_even = raw_edges / (raw_edges - E_B) if E_B < raw_edges else float('inf')
    return {'case_idx': case_idx, 'case': f'S{N_syn}', 'family': 'synthetic', 'N_syn': N_syn, 'm_ctrl': m_ctrl, 'N_B': N_B, 'N_transient': N_B, 'A': A, 'E_B': E_B, 'raw_edges': raw_edges, 'construction_edges': raw_edges, 'raw_storage': raw_edges, 'model_storage': N_B + E_B, 'desirability_storage': N_B, 'raw_sweep': raw_edges, 'compressed_sweep': E_B, 'compression_factor': f'{compression_factor:.12g}', 'break_even_K': f'{break_even:.12g}'}

def phenotype_storage_row(max_flip_size: int, case_idx: int) -> Dict[str, object]:
    case = phenotype9_case(flip_nodes=tuple(range(1, 10)), weights={}, max_flip_size=max_flip_size)
    row = storage_summary_row(case)
    raw_edges = int(row['raw_sweep'])
    E_B = int(row['compressed_sweep'])
    break_even = raw_edges / (raw_edges - E_B) if E_B < raw_edges else float('inf')
    return {'case_idx': case_idx, 'case': f'P9r{max_flip_size}', 'family': 'phenotype9', 'N_syn': '', 'm_ctrl': max_flip_size, 'N_B': row['N_B'], 'N_transient': row['N_transient'], 'A': row['A'], 'E_B': row['E_B'], 'raw_edges': raw_edges, 'construction_edges': raw_edges, 'raw_storage': row['raw_storage'], 'model_storage': row['model_storage'], 'desirability_storage': row['desirability_storage'], 'raw_sweep': raw_edges, 'compressed_sweep': E_B, 'compression_factor': row['compression_factor'], 'break_even_K': f'{break_even:.12g}'}

def cumulative_cost_rows(storage_rows: Sequence[Mapping[str, object]], max_sweeps: int=MAX_SWEEPS) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for srow in storage_rows:
        raw = int(srow['raw_sweep'])
        comp = int(srow['compressed_sweep'])
        ratio = float(srow['compression_factor'])
        for k in range(1, max_sweeps + 1):
            raw_total = k * raw
            compressed_total = raw + k * comp
            rows.append({'case_idx': srow['case_idx'], 'case': srow['case'], 'family': srow['family'], 'K': k, 'raw_total': raw_total, 'compressed_total': compressed_total, 'raw_norm': f'{k:.12g}', 'compressed_norm': f'{compressed_total / raw:.12g}', 'compression_factor': f'{ratio:.12g}'})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-sweeps', type=int, default=MAX_SWEEPS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    idx = 1
    for N_syn, m_ctrl in SYNTHETIC_CONFIGS:
        rows.append(synthetic_storage_row(N_syn, m_ctrl, idx))
        idx += 1
    for r in (1, 2):
        rows.append(phenotype_storage_row(r, idx))
        idx += 1
    storage_fields = ['case_idx', 'case', 'family', 'N_syn', 'm_ctrl', 'N_B', 'N_transient', 'A', 'E_B', 'raw_edges', 'construction_edges', 'raw_storage', 'model_storage', 'desirability_storage', 'raw_sweep', 'compressed_sweep', 'compression_factor', 'break_even_K']
    write_tsv(args.output / 'scaling_storage.tsv', rows, storage_fields)
    cumulative_rows = cumulative_cost_rows(rows, args.max_sweeps)
    cumulative_fields = ['case_idx', 'case', 'family', 'K', 'raw_total', 'compressed_total', 'raw_norm', 'compressed_norm', 'compression_factor']
    write_tsv(args.output / 'scaling_cumulative_cost.tsv', cumulative_rows, cumulative_fields)
    subset_names = []
    for case_name in ('P9r1', 'P9r2', 'S200'):
        fname = f'scaling_cumulative_cost_{case_name}.tsv'
        subset = [r for r in cumulative_rows if r['case'] == case_name]
        write_tsv(args.output / fname, subset, cumulative_fields)
        subset_names.append(fname)
    summary = [{'num_cases': len(rows), 'max_sweeps': args.max_sweeps, 'max_raw_storage': max((int(r['raw_storage']) for r in rows)), 'max_model_storage': max((int(r['model_storage']) for r in rows)), 'max_desirability_storage': max((int(r['desirability_storage']) for r in rows)), 'min_compression_factor': f"{min((float(r['compression_factor']) for r in rows)):.12g}", 'max_compression_factor': f"{max((float(r['compression_factor']) for r in rows)):.12g}"}]
    write_tsv(args.output / 'scaling_summary.tsv', summary)
    filenames = ['scaling_storage.tsv', *subset_names]
    print(f"Scaling/cumulative cost data generated: {len(rows)} cases, compression factors {summary[0]['min_compression_factor']}--{summary[0]['max_compression_factor']}")
if __name__ == '__main__':
    main()
