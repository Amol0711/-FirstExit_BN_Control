#!/usr/bin/env python3
"""Numerical regression tests for comparison solvers, models and bookkeeping."""
import json
import math
import unittest
from array import array
from pathlib import Path
from boolean_networks import BNCase, make_cases, phenotype9_case
from comparison_solvers import Graph, SCALE, TERMINAL, BETA, configurations, validate_case, build_graph, basin, reverse_dijkstra, solve_dp, dp_sweep, reference_rows, operator, solve_kl, recover, verify_rollouts, stationary_proper, run_trial

class ComparisonTests(unittest.TestCase):

    def fixture(self):
        c = BNCase('fixture', 2, (1,), ('00',), lambda s: s, {1: 0})
        rows = {0: array('q', [1, 0, 1, 2, 1, 1]), 1: array('q', [1, 0, 2, 3, 1, 3]), 2: array('q', [3, 0, 4, 2, 1, 5]), 3: array('q', [3, 0, 4, 3, 1, 7])}
        return Graph(c, ((), (1,)), (0, 0), (0, 1, 2, 3), frozenset(), rows, False, 8)

    def test_published_fixture_all_values_and_tie(self):
        g = self.fixture()
        v = {0: 4, 1: 7, 2: 0, 3: 3, TERMINAL: 0}
        rows = [[4, 7, 0, 3]]
        policies = []
        for row in g.rows.values():
            for j in range(2, len(row), 3):
                row[j] -= SCALE
        for _ in range(3):
            v, p = dp_sweep(g, v)
            rows.insert(0, [v[x] for x in g.states])
            policies.append(p)
        self.assertEqual(rows, [[9, 10, 15, 15], [6, 8, 10, 11], [1, 6, 5, 7], [4, 7, 0, 3]])
        last = {x: rows[1][x] for x in g.states}
        row = g.rows[2]
        s = [SCALE + row[j + 2] + last[row[j]] for j in range(0, len(row), 3)]
        self.assertEqual(s, [15, 15])

    def test_configuration_count_and_V1_exclusion(self):
        p = json.loads((Path(__file__).parent / 'comparison_settings.json').read_text())
        c = configurations(p)
        self.assertEqual(len(c), 133)
        self.assertNotIn('V1', [r['model'] for r in c])

    def test_all_existing_targets(self):
        for case in make_cases().values():
            validate_case(case)
        for r in [1, 2]:
            validate_case(phenotype9_case(tuple(range(1, 10)), {i: 1.0 for i in range(1, 10)}, r))

    def test_noninvariant_target_rejected(self):
        with self.assertRaises(ValueError):
            validate_case(BNCase('bad', 1, (), ('0',), lambda s: '1', {}))

    def test_one_step_known_solution(self):
        c = BNCase('direct', 2, (1, 2), ('00',), lambda s: '00', {1: 0.4, 2: 1.0})
        for compact in [False, True]:
            g = build_graph(c, compact)
            b = basin(g)
            d = reverse_dijkstra(g, b)
            self.assertTrue(all((d['values'][x] == 10 for x in b)))
            full = solve_dp(g, b, False)
            early = solve_dp(g, b, True)
            self.assertEqual(d['values'], early['values'])
            self.assertEqual(full['values'], early['values'])
            ref = reference_rows(build_graph(c, True), b)
            z = solve_kl(build_graph(c, True), b, ref)
            self.assertTrue(all((abs(z['values'][x] - math.exp(-1)) < 1e-13 for x in b)))

    def test_exact_cost_ties_not_rounded(self):
        c = make_cases()['R1'].with_weights({1: 0.2, 2: 0.4})
        g = build_graph(c, True)
        self.assertEqual(g.rho10, (0, 2, 6, 4))
        self.assertEqual(g.actions, ((), (1,), (1, 2), (2,)))

    def test_partial_basin_no_conditioning(self):
        c = BNCase('partial', 2, (1,), ('00',), lambda s: s, {1: 0})
        rows = {3: array('q', [TERMINAL, 0, 0, 1, 1, 0]), 2: array('q', [2, 0, 0]), 1: array('q', [1, 0, 0])}
        g = Graph(c, ((), (1,)), (0, 0), (3, 2, 1), frozenset([0]), rows, True, 4)
        b = basin(g)
        self.assertEqual(b, (3,))
        ref = reference_rows(g, b)
        self.assertEqual(list(ref[3]), [0.5, 0.5])
        s = solve_kl(g, b, ref)
        self.assertAlmostEqual(s['values'][3], math.exp(-1) / 2, places=14)
        d = reverse_dijkstra(g, b)
        self.assertEqual(d['values'][3], 10)
        dp = solve_dp(g, b, False)
        self.assertGreater(dp['values'][1], dp['penalty10'])
        self.assertEqual(dp['values'][3], 10)

    def test_multi_target_physical_arrival(self):
        c = make_cases()['V3']
        g = build_graph(c, True)
        b = basin(g)
        p = recover(g, b, 'C-SP', reverse_dijkstra(g, b))
        out = verify_rollouts(g, b, p)
        self.assertTrue(all((r['status'] == 'PROPER' for r in out)))
        arrivals = {r['path'][-1][-1] for r in out}
        self.assertIn(6, arrivals)

    def test_improper_cycle_retained(self):
        c = BNCase('identity', 2, (), ('00',), lambda s: s, {})
        g = build_graph(c, True)
        p = {'kind': 'stationary', 'policy': {1: (1, 0)}}
        out = verify_rollouts(g, (1,), p)
        self.assertEqual(out[0]['status'], 'IMPROPER')
        self.assertIsNone(out[0]['cost10'])
        self.assertFalse(stationary_proper(p['policy'], (1,)))

    def test_zero_sweep_budget_is_failure(self):
        g = build_graph(make_cases()['R1'], True)
        b = basin(g)
        ref = reference_rows(g, b)
        self.assertEqual(solve_kl(g, b, ref, max_sweeps=0)['status'], 'SWEEP_BUDGET_EXHAUSTED')

    def test_raw_compressed_same_values(self):
        for name in ['V2', 'V3', 'V4', 'R1', 'R2']:
            c = make_cases()[name]
            a = build_graph(c, False)
            b = build_graph(c, True)
            self.assertEqual(reverse_dijkstra(a, basin(a))['values'], reverse_dijkstra(b, basin(b))['values'])

    def test_full_and_early_DP_and_physical_rollouts(self):
        for name in ['V2', 'V3', 'V4', 'R1', 'R2']:
            g = build_graph(make_cases()[name], True)
            b = basin(g)
            full = solve_dp(g, b, False)
            early = solve_dp(g, b, True)
            self.assertEqual(full['values'], early['values'])
            self.assertEqual(full['values'], reverse_dijkstra(g, b)['values'])
            for solved in [full, early]:
                r = recover(g, b, 'TW-DP-H', solved)
                o = verify_rollouts(g, b, r)
                self.assertTrue(all((v['cost10'] == solved['values'][v['initial']] for v in o)))

    def test_payload_byte_counts(self):
        r = run_trial({'id': 'R1', 'model': 'R1', 'weights': ['0.4', '1.0']}, 'C-SP', True)
        self.assertEqual(r['memory']['graph_payload_bytes'], 3 * 8 * r['E_all'])
        self.assertEqual(r['C_successor_calls'], r['N0'] * r['A'])

    def test_timeindexed_policy_does_not_false_cycle(self):
        c = make_cases()['V3']
        g = build_graph(c, True)
        b = basin(g)
        d = solve_dp(g, b, False)
        p = recover(g, b, 'TW-DP-H', d)
        self.assertEqual(p['logical_policy_entries'], len(b) ** 2)
        self.assertEqual(len(verify_rollouts(g, b, p)), len(b))
if __name__ == '__main__':
    unittest.main(verbosity=2)
