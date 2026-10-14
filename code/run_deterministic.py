#!/usr/bin/env python3
"""Run all six deterministic numerical producers into a new output directory."""
from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path

SCRIPTS = ('run_exactness.py', 'run_recovery.py', 'run_regularization.py',
           'run_scaling.py', 'run_phenotype.py', 'run_library_checks.py')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    package = Path(__file__).resolve().parents[1]
    if output == package or package in output.parents:
        parser.error('Use an output directory outside the package')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    previous = sys.argv[:]
    try:
        for name in SCRIPTS:
            script = Path(__file__).with_name(name)
            print(f'Running {name}', flush=True)
            sys.argv = [str(script), '--output', str(output)]
            runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous
    print('Deterministic numerical outputs complete', flush=True)

if __name__ == '__main__':
    main()
