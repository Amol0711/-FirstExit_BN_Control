#!/usr/bin/env python3
"""Render standalone numerical plots from tabular data and sampling histories."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE / 'results')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == HERE or HERE in out.parents:
        parser.error('Output must be outside the package')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output must be absent or empty')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('Install requirements-plot.txt to render numerical plots') from error
    out.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    deterministic = data / 'deterministic'
    sampling = data / 'sampling'

    figure, axis = plt.subplots(figsize=(9, 6), layout='constrained')
    for path in sorted(sampling.glob('*_history.jsonl')):
        history = [json.loads(line) for line in path.read_text().splitlines()]
        final = json.loads(path.with_name(path.name.replace('_history.jsonl', '_result.json')).read_text())
        label = f"{final['model']} / {final['seed']}"
        if final['status'] == 'BUDGET_EXHAUSTED':
            label += ' (budget exhausted)'
        axis.semilogy([r['updates'] for r in history], [r['error_bound_estimate'] for r in history], label=label)
    axis.axhline(0.001, linestyle='--', linewidth=1, label='Stopping tolerance')
    axis.set(xlabel='Sampled updates', ylabel='Floating residual-based error estimate',
             title='Complete fixed-seed sampling histories')
    axis.legend(fontsize=8, ncols=2)
    figure.savefig(out / 'sampling_error_estimates.png', dpi=180)
    plt.close(figure)

    rows = read_rows(deterministic / 'recovery_conditions.tsv')
    figure, axis = plt.subplots(figsize=(8, 5.5), layout='constrained')
    for category, label in [('condition_met_exact', 'Gap test met; zero local loss'),
                            ('condition_not_met_exact', 'Gap test not met; zero local loss'),
                            ('bounded_loss', 'Positive local loss')]:
        subset = [r for r in rows if r['category'] == category]
        axis.scatter([float(r['two_epsilon']) for r in subset], [float(r['local_loss']) for r in subset],
                     s=18, alpha=0.65, label=f'{label} ({len(subset)})')
    upper = max(float(r['two_epsilon']) for r in rows)
    axis.plot([0, upper], [0, upper], linestyle='--', linewidth=1, label='Equal numerical values')
    axis.set(xlabel='Twice the computed value mismatch', ylabel='Original-cost local loss',
             title=f'Recovery sweep, all {len(rows)} state-weight records')
    axis.legend(fontsize=8)
    figure.savefig(out / 'recovery_local_loss.png', dpi=180)
    plt.close(figure)

    rows = read_rows(deterministic / 'phenotype_hitting_times.tsv')
    positions = list(range(len(rows)))
    labels = ['Not reached' if r['tau_label'] == 'inf' else r['tau_label'] for r in rows]
    figure, axis = plt.subplots(figsize=(8, 5.5), layout='constrained')
    axis.bar([i - 0.2 for i in positions], [int(r['uncontrolled']) for r in rows], width=0.4, label='Uncontrolled')
    axis.bar([i + 0.2 for i in positions], [int(r['controlled']) for r in rows], width=0.4, label='Recovered policy')
    axis.set_xticks(positions, labels)
    axis.set(xlabel='Transitions until target entry', ylabel='Number of initial states',
             title='Nine-node model, all 511 non-target initial states')
    axis.legend()
    figure.savefig(out / 'phenotype_hitting_times.png', dpi=180)
    plt.close(figure)

    rows = read_rows(deterministic / 'scaling_storage.tsv')
    positions = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(8, 5.5), layout='constrained')
    for offset, field, label in [(-0.25, 'raw_storage', 'Raw action records'),
                                (0, 'model_storage', 'Compressed model entries'),
                                (0.25, 'desirability_storage', 'Desirability entries')]:
        axis.bar([i + offset for i in positions], [int(r[field]) for r in rows], width=0.25, label=label)
    axis.set_xticks(positions, [r['case'] for r in rows])
    axis.set_yscale('log')
    axis.set(xlabel='Numerical case (S cases are analytic counts)', ylabel='Record or entry count, not bytes',
             title='Storage accounting; not peak-memory measurements')
    axis.legend(fontsize=8)
    figure.savefig(out / 'storage_counts.png', dpi=180)
    plt.close(figure)
    print('Four numerical plots written', flush=True)

if __name__ == '__main__':
    main()
