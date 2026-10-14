#!/usr/bin/env python3
"""Numerical regression tests for fixed-seed passive sampling and complete transition rows."""
import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch
from passive_sampling import TERMINAL, build_kernel, dense_reference, flip, induced_policy, policy_paths, sample
MODELS = json.loads(Path(__file__).with_name('sampling_models.json').read_text())
PARTIAL = {'n': 2, 'truth_table': {'00': '00', '01': '01', '10': '11', '11': '01'}, 'targets': ['00'], 'eligible_nodes': [1], 'weights_decimal': {'1': '0'}, 'state_cost_decimal': '1'}
ONE = {'n': 1, 'truth_table': {'0': '0', '1': '0'}, 'targets': ['0'], 'eligible_nodes': [], 'weights_decimal': {}, 'state_cost_decimal': '1'}

class SamplingTests(unittest.TestCase):

    def setUp(self):
        self.k = build_kernel(MODELS['R1'])

    def test_complete_truth_maps(self):
        for name, m in MODELS.items():
            for x, y in m['truth_table'].items():
                b = list(map(int, x))
                expected = f'{b[0] & b[1]}0' if name == 'R1' else f'0{b[0]}{b[1] | b[2]}'
                self.assertEqual(y, expected)

    def test_expected_graph_counts(self):
        self.assertEqual((len(self.k.states), self.k.edge_count), (3, 6))
        k = build_kernel(MODELS['R2'])
        self.assertEqual((len(k.states), k.edge_count), (7, 22))

    def test_row_sums_and_order(self):
        for name in MODELS:
            k = build_kernel(MODELS[name])
            self.assertEqual(tuple(sorted(k.states)), k.states)
            for row in k.rows.values():
                self.assertAlmostEqual(sum((e.probability for e in row)), 1, places=14)
                self.assertEqual([e.successor for e in row], sorted((e.successor for e in row)))

    def test_physical_representatives(self):
        for k in [self.k, build_kernel(MODELS['R2'])]:
            for x, row in k.rows.items():
                for e in row:
                    self.assertEqual(k.truth_table[flip(x, e.action)], e.physical_successor)
                    self.assertEqual(TERMINAL if e.physical_successor in k.targets else e.physical_successor, e.successor)

    def test_invalid_map_rejected(self):
        m = copy.deepcopy(MODELS['R1'])
        m['truth_table'].pop('11')
        with self.assertRaises(ValueError):
            build_kernel(m)

    def test_noninvariant_target_rejected(self):
        m = copy.deepcopy(MODELS['R1'])
        m['truth_table']['00'] = '10'
        with self.assertRaises(ValueError):
            build_kernel(m)

    def test_invalid_weight_rejected(self):
        m = copy.deepcopy(MODELS['R1'])
        m['weights_decimal']['1'] = '-1'
        with self.assertRaises(ValueError):
            build_kernel(m)

    def test_invalid_node_rejected(self):
        m = copy.deepcopy(MODELS['R1'])
        m['eligible_nodes'] = [1, 1]
        with self.assertRaises(ValueError):
            build_kernel(m)

    def test_zero_cost_rejected(self):
        m = copy.deepcopy(MODELS['R1'])
        m['state_cost_decimal'] = '0'
        with self.assertRaises(ValueError):
            build_kernel(m)

    def test_underflow_rejected(self):
        m = copy.deepcopy(MODELS['R1'])
        m['state_cost_decimal'] = '10000'
        with self.assertRaises(ValueError):
            build_kernel(m)

    def test_integer_arguments(self):
        for key in ['seed', 'max_updates', 'scan_every', 'path_cap']:
            for value in [True, -1, 1.2]:
                args = dict(seed=1, max_updates=0)
                args[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    sample(self.k, **args)

    def test_invalid_omega(self):
        for v in [0.5, 0, 1.01, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                sample(self.k, seed=1, omega=v, max_updates=0)

    def test_invalid_tolerance(self):
        for v in [0, -1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                sample(self.k, seed=1, tolerance=v, max_updates=0)

    def test_zero_budget(self):
        r = sample(self.k, seed=1, max_updates=0)
        self.assertEqual(r['status'], 'BUDGET_EXHAUSTED')
        self.assertEqual(r['work']['sample_updates'], 0)
        self.assertEqual(r['work']['residual_scans'], 1)
        self.assertEqual(sum(r['visits'].values()), 0)

    def test_empty_basin(self):
        m = copy.deepcopy(ONE)
        m['truth_table']['1'] = '1'
        k = build_kernel(m)
        r = sample(k, seed=1)
        self.assertEqual(r['status'], 'EMPTY_TRANSIENT_BASIN')
        self.assertEqual(r['z'], {TERMINAL: 1.0})
        self.assertEqual(r['work']['episodes'], 0)
        self.assertEqual(r['error_bound_estimate'], 0)

    def test_one_step(self):
        k = build_kernel(ONE)
        r = sample(k, seed=1, scan_every=1, max_updates=10)
        self.assertEqual(r['status'], 'RESIDUAL_TOLERANCE_MET')
        self.assertEqual(r['work']['sample_updates'], 1)
        self.assertEqual(r['z']['1'], math.exp(-1))
        self.assertEqual(r['work']['target_events'], 1)

    def test_success_at_hard_cap(self):
        r = sample(build_kernel(ONE), seed=1, scan_every=7, max_updates=1)
        self.assertEqual(r['status'], 'RESIDUAL_TOLERANCE_MET')
        self.assertEqual(r['work']['residual_scans'], 1)

    def test_incomplete_visitation_prevents_success(self):
        r = sample(self.k, seed=1, max_updates=1, scan_every=1, tolerance=100)
        self.assertEqual(r['status'], 'BUDGET_EXHAUSTED')

    def test_nonmultiple_final_scan(self):
        history = []
        r = sample(self.k, seed=1, max_updates=17, scan_every=6, tolerance=1e-20, observer=history.append)
        self.assertEqual([v['updates'] for v in history], [6, 12, 17])
        self.assertEqual(r['work']['residual_scans'], 3)

    def test_scan_count_is_actual_residual_count(self):
        from passive_sampling import Kernel
        old = Kernel.residual
        calls = []

        def traced(k, z):
            calls.append(1)
            return old(k, z)
        with patch.object(Kernel, 'residual', traced):
            r = sample(self.k, seed=1, max_updates=17, scan_every=6, tolerance=1e-20)
        self.assertEqual(len(calls), r['work']['residual_scans'])

    def test_checkpoint_payload(self):
        h = []
        r = sample(self.k, seed=2, max_updates=20, scan_every=4, tolerance=1e-20, observer=h.append)
        self.assertEqual(h[-1]['z'], r['z'])
        self.assertEqual(h[-1]['visits'], r['visits'])
        self.assertEqual(h[-1]['work'], r['work'])
        self.assertEqual(h[-1]['rng_state_sha256'], r['last_rng_state_sha256'])

    def test_observer_isolation(self):
        a = sample(self.k, seed=2, max_updates=20, scan_every=4)

        def malicious(s):
            s['z'].clear()
            s['work'].clear()
        b = sample(self.k, seed=2, max_updates=20, scan_every=4, observer=malicious)
        self.assertEqual(a, b)

    def test_repeatability(self):
        a = sample(self.k, seed=19, max_updates=150, scan_every=17)
        self.assertEqual(a, sample(self.k, seed=19, max_updates=150, scan_every=17))

    def test_no_reference_call_in_updates(self):
        with patch('passive_sampling.dense_reference', side_effect=AssertionError('oracle used')):
            sample(self.k, seed=3, max_updates=100)

    def test_counts_and_exclusive_episode_ends(self):
        for name in MODELS:
            r = sample(build_kernel(MODELS[name]), seed=17, max_updates=203, scan_every=17, tolerance=1e-20)
            w = r['work']
            self.assertEqual(w['sample_updates'], sum(r['visits'].values()))
            self.assertEqual(w['episodes'], sum(r['restarts'].values()))
            self.assertEqual(w['episodes'], sum((w[k] for k in ['target_events', 'zero_boundary_events', 'cap_events', 'interrupted_episodes'])))
            self.assertEqual(r['z'][TERMINAL], 1.0)
            self.assertTrue(all((0 <= v <= 1 for v in r['z'].values())))

    def test_partial_basin_no_renormalization(self):
        k = build_kernel(PARTIAL)
        self.assertEqual(k.states, ('10',))
        self.assertEqual(k.zero_boundary, frozenset({'11'}))
        z = dense_reference(k)
        self.assertEqual(z['10'], math.exp(-1) / 2)
        self.assertEqual(k.residual(z), 0)

    def test_zero_boundary_events(self):
        r = sample(build_kernel(PARTIAL), seed=9, max_updates=100, scan_every=100, tolerance=1e-20)
        self.assertGreater(r['work']['zero_boundary_events'], 0)
        self.assertNotIn('11', r['visits'])
        self.assertEqual(r['work']['sample_updates'], 100)

    def test_budget_interrupt(self):
        statuses = [sample(self.k, seed=s, max_updates=1, scan_every=1, tolerance=1e-20) for s in range(10)]
        self.assertTrue(any((r['work']['interrupted_episodes'] == 1 for r in statuses)))

    def test_path_caps(self):
        r = sample(self.k, seed=17, path_cap=1, max_updates=100, scan_every=100, tolerance=1e-20)
        self.assertGreater(r['work']['cap_events'], 0)
        self.assertEqual(r['work']['episodes'], r['work']['sample_updates'])

    def test_reference_residual(self):
        for name in MODELS:
            k = build_kernel(MODELS[name])
            z = dense_reference(k)
            self.assertLess(k.residual(z), 1e-15)

    def test_reference_modes_proper(self):
        for name in MODELS:
            k = build_kernel(MODELS[name])
            z = dense_reference(k)
            law, modes, undef = induced_policy(k, z)
            self.assertFalse(undef)
            self.assertTrue(all((p['proper'] for p in policy_paths(k, modes).values())))
            for row in law.values():
                self.assertAlmostEqual(sum(row.values()), 1, places=14)

    def test_policy_corruption_rejected(self):
        k = self.k
        _, modes, _ = induced_policy(k, dense_reference(k))
        modes['01']['physical_successor'] = '11'
        with self.assertRaises(ValueError):
            policy_paths(k, modes)

    def test_unknown_state_or_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.k.value({}, '01')
        with self.assertRaises(ValueError):
            self.k.value({'01': math.nan}, '01')

    def test_two_target_realization(self):
        m = {'n': 2, 'truth_table': {'00': '01', '01': '00', '10': '01', '11': '00'}, 'targets': ['00', '01'], 'eligible_nodes': [], 'weights_decimal': {}, 'state_cost_decimal': '1'}
        k = build_kernel(m)
        _, modes, _ = induced_policy(k, dense_reference(k))
        self.assertEqual(modes['10']['physical_successor'], '01')
        self.assertTrue(all((p['proper'] for p in policy_paths(k, modes).values())))
if __name__ == '__main__':
    unittest.main()
