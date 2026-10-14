"""Regression tests for numerical entry points and stored-result integrity."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class NumericalInterfaceTests(unittest.TestCase):
    def test_sampling_source_integrity(self):
        lock = json.loads((CODE / 'sampling_source_lock.json').read_text())
        self.assertEqual(len(lock['sha256']), 5)
        for name, expected in lock['sha256'].items():
            self.assertEqual(sha(CODE / name), expected)

    def test_sampling_result_integrity(self):
        root = ROOT / 'results/sampling'
        expected = json.loads((root / 'output_hashes.json').read_text())
        actual = {p.name: sha(p) for p in root.iterdir() if p.is_file() and p.name != 'output_hashes.json'}
        self.assertEqual(actual, expected)

    def test_sampling_run_coverage(self):
        root = ROOT / 'results/sampling'
        settings = json.loads((CODE / 'sampling_settings.json').read_text())
        expected = {f'{m}_{s}_result.json' for m in settings['models'] for s in settings['seeds']}
        self.assertEqual(expected, {p.name for p in root.glob('*_result.json')})
        self.assertEqual(len(expected), 10)

    def test_sampling_budget_outcomes_are_retained(self):
        records = [json.loads(p.read_text()) for p in (ROOT / 'results/sampling').glob('*_result.json')]
        self.assertEqual(sum(r['status'] == 'BUDGET_EXHAUSTED' for r in records), 3)
        self.assertEqual(sum(r['status'] == 'RESIDUAL_TOLERANCE_MET' for r in records), 7)

    def test_sampling_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            sentinel = out / 'sentinel.txt'; sentinel.write_text('preserve')
            result = subprocess.run([sys.executable, '-B', str(CODE / 'run_sampling.py'), '--output', str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'preserve')
            self.assertEqual(list(out.iterdir()), [sentinel])

    def test_deterministic_refuses_package_output(self):
        result = subprocess.run([sys.executable, '-B', str(CODE / 'run_deterministic.py'), '--output', str(ROOT)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_recorded_comparison_coverage(self):
        root = ROOT / 'results/comparison'
        self.assertEqual(len(list((root / 'timing').glob('*.json'))), 532)
        self.assertEqual(len(list((root / 'memory').glob('*.json'))), 532)
        self.assertEqual(len(list((root / 'evaluation/high_precision').glob('*.json'))), 133)

    def test_recorded_signature_preserves_integer_state_order(self):
        from verify_comparison import recorded_signature
        import comparison_solvers as solvers
        result = {'config': {}, 'method': 'C-KL-MODE', 'outcomes': [],
                  'values': {}, 'solver': {}, 'mode_scores': {2: {'x': 1}, 10: {'x': 2}}}
        expected = solvers.digest(result)
        restored = json.loads(json.dumps(result))
        self.assertNotEqual(solvers.digest(restored), expected)
        self.assertEqual(recorded_signature(restored), expected)

    def test_fixed_measurement_quartiles(self):
        from evaluate_comparison import qstats
        self.assertEqual(qstats([1, 2, 3, 4]), {'median': 2.5, 'q1': 1.75, 'q3': 3.25})

if __name__ == '__main__':
    unittest.main()
