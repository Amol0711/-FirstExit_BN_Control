#!/usr/bin/env python3
"""Replay comparison numerics and reaggregate fixed measurements without retiming."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import comparison_solvers as solvers
import evaluate_comparison as evaluate

SIGNATURE_FIELDS = ('config', 'method', 'outcomes', 'values', 'solver', 'mode_scores')
PHASES = ('model_validation_ns', 'graph_build_ns', 'basin_ns', 'reference_ns',
          'solve_ns', 'recovery_ns', 'properness_ns', 'total_no_diagnostic_ns')

def recorded_signature(result: dict) -> str:
    """Restore integer state keys lost in JSON before checking the saved digest."""
    signed = {key: result[key] for key in SIGNATURE_FIELDS}
    signed['mode_scores'] = {int(key): value for key, value in signed['mode_scores'].items()}
    return solvers.digest(signed)

def hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}

def csv_text(rows: list[dict]) -> str:
    if not rows:
        return ''
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()

def field_text(value: object) -> str:
    return '' if value is None else str(value)

def verify(run: Path, output: Path) -> dict:
    before = hashes(run)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise ValueError('The recorded campaign is not complete')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Verification output must be absent or empty')
    package = Path(__file__).resolve().parents[1]
    if output == package or package in output.parents or output == run or run in output.parents:
        raise ValueError('Write verification results outside the package and recorded campaign')
    output.mkdir(parents=True, exist_ok=True)
    checks = 0
    def require(ok: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)
    configurations = solvers.configurations(manifest['settings'])
    expected = {(c['id'], m) for c in configurations for m in solvers.METHODS}
    blocks, memory, trials = {}, {}, []
    for phase, store in [('timing', blocks), ('memory', memory)]:
        for path in sorted((run / phase).glob('*.json')):
            record = json.loads(path.read_text())
            key = (record['config']['id'], record['method'])
            require(key not in store, 'Duplicate comparison block')
            require(record['status'] == 'OK', f'Recorded unsuccessful block {path.name}')
            store[key] = record
            result = record['first_result']
            require(recorded_signature(result) == result['result_signature'],
                    'Saved numerical result signature')
            for trial in record['trials']:
                require(trial['result_signature'] == result['result_signature'], 'Within-block numerical equality')
                require(trial['status'] == 'OK', 'Recorded trial status')
                require(trial['phases']['total_no_diagnostic_ns'] == sum(
                    trial['phases'][k] for k in PHASES[:-1]), 'Recorded phase sum')
                if phase == 'timing':
                    trials.append({'config': key[0], 'method': key[1], 'phase': trial['phase'],
                                   'repetition': trial['index'], **{k: trial['phases'][k] for k in PHASES},
                                   'trial_wall_ns': trial['trial_wall_ns'], 'status': trial['status'],
                                   'result_signature': trial['result_signature']})
    require(set(blocks) == expected and set(memory) == expected, 'Complete method/configuration coverage')
    evaluation = run / 'evaluation'
    require(csv_text(trials).encode() == (evaluation / 'timing_trials.csv').read_bytes(),
            'Every archived warmup and measured timing row')
    summary_rows = list(csv.DictReader((evaluation / 'per_configuration_summary.csv').open(newline='')))
    summaries = {(r['config'], r['method']): r for r in summary_rows}
    references = {r['config']: r for r in csv.DictReader((evaluation / 'reference_checks.csv').open(newline=''))}
    require(set(summaries) == expected, 'Summary coverage')
    for key, block in blocks.items():
        require(block['first_result']['result_signature'] == memory[key]['first_result']['result_signature'],
                'Separate memory-worker numerical signature')
        measured = [r for r in block['trials'] if r['phase'] == 'measured']
        warmups = [r for r in block['trials'] if r['phase'] == 'warmup']
        require(len(measured) == manifest['settings']['timing']['measured_repetitions'], 'Measured repetition count')
        require(len(warmups) == manifest['settings']['timing']['warmups'], 'Warmup count')
        require(len(memory[key]['trials']) == 1, 'Separate no-warmup memory trial')
        row = summaries[key]
        for phase in PHASES:
            for label, value in evaluate.qstats([t['phases'][phase] for t in measured]).items():
                require(float(row[f'{phase}_{label}']) == value, 'Fixed timing median/quartile')
        for name, value in memory[key]['first_result']['memory'].items():
            if name != 'measurement_qualification':
                require(row[name] == field_text(value), 'Fixed memory observation')
        ref = references[key[0]]
        gap = int(ref['gap_ns']) if key[1] == 'C-KL-MODE' else 0
        independent = int(ref['independent_audit_ns'])
        require(float(row['single_case_optional_gap_ns']) == gap, 'Recorded gap duration')
        require(float(row['single_case_independent_audit_ns']) == independent, 'Recorded reference-check duration')
        require(float(row['composite_total_with_gap_ns']) == float(row['total_no_diagnostic_ns_median']) + gap,
                'Fixed composite controller-plus-gap total')
        require(float(row['composite_total_with_audit_and_gap_ns']) ==
                float(row['total_no_diagnostic_ns_median']) + gap + independent, 'Fixed composite reference-check total')
    quality, modes, diagnostics = [], [], []
    numerical_replays = 0
    # The clock is disabled: these calls verify outcomes, not execution time.
    # No fresh performance or memory measurement is produced or substituted.
    with patch.object(solvers, 'perf_counter_ns', return_value=0), \
         patch.object(evaluate, 'perf_counter_ns', return_value=0):
        for index, config in enumerate(configurations, 1):
            for method in solvers.METHODS:
                result = solvers.run_trial(config, method, memory=False)
                saved = blocks[(config['id'], method)]['first_result']
                require(result['result_signature'] == saved['result_signature'], 'Replayed solver values, policies and work')
                require(all(v == 0 for v in result['phases'].values()), 'Numerical replay did not measure time')
                numerical_replays += 1
            q, s, m, d, reference, z = evaluate.audit_one(config, blocks)
            quality.extend(q); modes.extend(m); diagnostics.extend(d)
            for item in s:
                saved = summaries[(item['config'], item['method'])]
                for field, value in item.items():
                    require(saved[field] == field_text(value), f'Replayed numerical summary {field}')
            saved_hp = json.loads((evaluation / 'high_precision' / f'{config["id"]}.json').read_text())
            require({str(key): value for key, value in z.items()} == saved_hp['z'], 'Replayed high-precision values')
            for field, value in reference.items():
                if not field.endswith('_ns'):
                    require(value == saved_hp['audit'][field], f'Replayed high-precision diagnostic {field}')
            if index % 20 == 0 or index == len(configurations):
                print(f'Numerical comparison checks {index}/{len(configurations)}', flush=True)
    for filename, rows in [('per_state_quality.csv', quality), ('mode_precision_checks.csv', modes),
                           ('gap_diagnostics.csv', diagnostics)]:
        encoded = csv_text(rows).encode()
        require(encoded == (evaluation / filename).read_bytes(), f'Regenerated {filename}')
        (output / filename).write_bytes(encoded)
    aggregate = {}
    for method in solvers.METHODS:
        rows = [r for r in quality if r['method'] == method]
        aggregate[method] = {'states': len(rows), 'proper': sum(r['proper'] for r in rows),
                            'optimal_first_action': sum(r['optimal_first_action'] for r in rows),
                            'optimal_complete_policy': sum(r['optimal_complete_policy'] for r in rows),
                            'max_cost_gap': max(r['cost_gap'] for r in rows),
                            'mean_cost_gap': statistics.mean(r['cost_gap'] for r in rows)}
    saved_summary = json.loads((evaluation / 'evaluation_summary.json').read_text())
    require(aggregate == saved_summary['quality_aggregate'], 'Complete quality aggregate including adverse cases')
    ties = sum(r['audit_status'].startswith('NUMERICAL_TIE') for r in modes)
    require(ties == saved_summary['unresolved_exact_mode_ties'], 'Unresolved exact-equality classifications')
    require(dict(Counter(r['gap_test_class'] for r in diagnostics if r['model'] in ('R1', 'R2'))) ==
            saved_summary['R1_R2_gap_counts'], 'All comparison gap classifications')
    require(hashes(run) == before, 'Recorded campaign remains byte-identical')
    result = {'status': 'PASS', 'checks': checks, 'configurations': len(configurations),
              'timing_blocks': len(blocks), 'memory_blocks': len(memory),
              'warmup_trials': sum(t['phase'] == 'warmup' for t in trials),
              'measured_trials': sum(t['phase'] == 'measured' for t in trials),
              'solver_numerical_replays': numerical_replays, 'state_policy_rows': len(quality),
              'high_precision_configurations': len(configurations), 'unresolved_exact_mode_ties': ties,
              'quality_aggregate': aggregate, 'performance_measurements_created': 0,
              'timers_disabled_for_numerical_replay': True, 'recorded_files_unchanged': True}
    (output / 'verification_report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run.resolve(), args.output.resolve()), indent=2), flush=True)

if __name__ == '__main__':
    main()
