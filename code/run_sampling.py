#!/usr/bin/env python3
"""Execute the fixed-seed finite sampling study and retain every residual-scan history."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from passive_sampling import VERSION, build_kernel, dense_reference, induced_policy, sample
HERE = Path(__file__).resolve().parent

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == HERE.parent or HERE.parent in destination.parents:
        parser.error('Use an output directory outside the package')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be absent or empty; reference records are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((HERE / 'sampling_source_lock.json').read_text())
    for name, digest in freeze['sha256'].items():
        if sha(HERE / name) != digest:
            raise RuntimeError(f'Frozen source changed: {name}')
    settings = json.loads((HERE / 'sampling_settings.json').read_text())
    models = json.loads((HERE / 'sampling_models.json').read_text())
    summaries, state_rows = ([], [])
    for model_name in settings['models']:
        kernel = build_kernel(models[model_name])
        references = dense_reference(kernel)
        _, reference_modes, undefined = induced_policy(kernel, references)
        if undefined:
            raise ArithmeticError('Unresolved numerical reference policy')
        write_json(args.output / f'{model_name}_reference.json', {'z': references, 'reference_residual': kernel.residual(references), 'mode': reference_modes, 'states': kernel.states, 'rows': {x: [e.__dict__ for e in row] for x, row in kernel.rows.items()}, 'zero_boundary': sorted(kernel.zero_boundary), 'beta': kernel.beta})
        for seed in settings['seeds']:
            stem = f'{model_name}_{seed}'
            with (args.output / f'{stem}_history.jsonl').open('w') as history:

                def record(snapshot):
                    history.write(json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
                result = sample(kernel, seed=seed, max_updates=settings['max_updates'], path_cap=len(kernel.states), scan_every=settings['scan_every'], omega=settings['omega'], tolerance=settings['tolerance'], observer=record)
            result['model'] = model_name
            result['observed_desirability_error'] = max((abs(result['z'][x] - references[x]) for x in kernel.states))
            result['matching_mode_action_pairs'] = sum((result['modes'][x] == reference_modes[x] for x in kernel.states))
            result['proper_states'] = sum((p['proper'] for p in result['policy_paths'].values()))
            write_json(args.output / f'{stem}_result.json', result)
            summaries.append({'model': model_name, 'seed': seed, 'status': result['status'], **result['work'], 'minimum_state_visits': min(result['visits'].values()), 'minimum_restarts': min(result['restarts'].values()), 'residual': result['residual'], 'error_bound_estimate': result['error_bound_estimate'], 'observed_desirability_error': result['observed_desirability_error'], 'matching_mode_action_pairs': result['matching_mode_action_pairs'], 'proper_states': result['proper_states'], 'undefined_policy_rows': len(result['undefined_policy_rows'])})
            for x in kernel.states:
                mode = result['modes'].get(x, {})
                state_rows.append({'model': model_name, 'seed': seed, 'state': x, 'desirability': result['z'][x], 'reference_desirability': references[x], 'visits': result['visits'][x], 'restarts': result['restarts'][x], 'successor': mode.get('successor'), 'action': json.dumps(mode.get('action')), 'physical_successor': mode.get('physical_successor'), 'reference_mode_match': mode == reference_modes[x], 'proper': result['policy_paths'][x]['proper']})
            print(f"{stem}: {result['status']} after {result['work']['sample_updates']} updates", flush=True)
    write_csv(args.output / 'sampling_summary.csv', summaries)
    write_csv(args.output / 'state_visits.csv', state_rows)
    write_json(args.output / 'execution.json', {'implementation': VERSION, 'python': sys.version, 'platform': platform.platform(), 'source_lock_sha256': sha(HERE / 'sampling_source_lock.json'), 'code_sha256': freeze['sha256'], 'results': len(summaries), 'state_records': len(state_rows), 'status_counts': {s: sum((r['status'] == s for r in summaries)) for s in sorted({r['status'] for r in summaries})}, 'total_updates': sum((r['sample_updates'] for r in summaries)), 'total_scans': sum((r['residual_scans'] for r in summaries)), 'stored_reference_values_used_in_updates': False})
    hashes = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    write_json(args.output / 'output_hashes.json', hashes)
if __name__ == '__main__':
    main()
