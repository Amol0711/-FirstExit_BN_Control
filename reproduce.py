#!/usr/bin/env python3
"""Run numerical tests, regenerate results and verify recorded comparisons."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}

def compare_exact(generated: Path, stored: Path, ignored: tuple[str, ...] = ()) -> int:
    actual = {p.name for p in generated.iterdir() if p.is_file()} - set(ignored)
    expected = {p.name for p in stored.iterdir() if p.is_file()} - set(ignored)
    if actual != expected:
        raise AssertionError(f'Output coverage mismatch: missing={expected-actual}, extra={actual-expected}')
    for name in sorted(expected):
        if digest(generated / name) != digest(stored / name):
            raise AssertionError(f'Exact numerical comparison failed for {name}. No tolerance was silently substituted.')
    return len(expected)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == HERE or HERE in out.parents:
        parser.error('Output must be outside the package')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output must be absent or empty')
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'; logs.mkdir()
    before = inventory(HERE)
    frozen = json.loads((HERE / 'FILE_HASHES.json').read_text())
    observed = {k: v for k, v in before.items() if k != 'FILE_HASHES.json'}
    if observed != frozen:
        raise AssertionError('Package file coverage or hashes do not match FILE_HASHES.json')
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'PYTHONHASHSEED': '0', 'PYTHONPATH': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
    code = HERE / 'code'
    jobs = [
        ('unit_tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-v'], code),
        ('deterministic', [sys.executable, '-B', str(code / 'run_deterministic.py'), '--output', str(out / 'deterministic')], None),
        ('sampling', [sys.executable, '-B', str(code / 'run_sampling.py'), '--output', str(out / 'sampling')], None),
        ('sampling_replay', [sys.executable, '-B', str(code / 'replay_sampling.py'), '--output', str(out / 'sampling'),
                             '--report', str(out / 'sampling_replay.json')], None),
        ('comparison_verification', [sys.executable, '-B', str(code / 'verify_comparison.py'),
                                     '--run', str(HERE / 'results/comparison'), '--output', str(out / 'comparison_checks')], None),
    ]
    for name, command, cwd in jobs:
        print(f'Running {name}', flush=True)
        with (logs / f'{name}.log').open('w') as handle:
            process = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'{name} failed. Inspect {logs / (name + ".log")}')
    deterministic = compare_exact(out / 'deterministic', HERE / 'results/deterministic')
    sampling = compare_exact(out / 'sampling', HERE / 'results/sampling', ('execution.json', 'output_hashes.json'))
    generated_hashes = json.loads((out / 'sampling/output_hashes.json').read_text())
    if generated_hashes != {p.name: digest(p) for p in (out / 'sampling').iterdir()
                            if p.is_file() and p.name != 'output_hashes.json'}:
        raise AssertionError('Generated sampling hash coverage')
    generated_execution = json.loads((out / 'sampling/execution.json').read_text())
    recorded_execution = json.loads((HERE / 'results/sampling/execution.json').read_text())
    for field in ('results', 'state_records', 'status_counts', 'total_updates', 'total_scans',
                  'stored_reference_values_used_in_updates'):
        if generated_execution[field] != recorded_execution[field]:
            raise AssertionError(f'Sampling execution numerical field {field}')
    if inventory(HERE) != before:
        raise AssertionError('The package was modified during reproduction')
    result = {'status': 'PASS', 'deterministic_exact_files': deterministic,
              'sampling_exact_numerical_files': sampling, 'sampling_metadata_files': 2,
              'sampling_replay': json.loads((out / 'sampling_replay.json').read_text()),
              'comparison_verification': json.loads((out / 'comparison_checks/verification_report.json').read_text()),
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'numpy': importlib.metadata.version('numpy'), 'mpmath': importlib.metadata.version('mpmath')},
              'numerical_comparison': 'Exact file comparison on the tested runtime; environment metadata compared separately',
              'package_unchanged': True, 'performance_campaign_retimed': False}
    (out / 'REPRODUCTION.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'PASS: {deterministic} deterministic files; {sampling} sampling numerical files; comparison numerics and fixed measurements verified.', flush=True)

if __name__ == '__main__':
    main()
