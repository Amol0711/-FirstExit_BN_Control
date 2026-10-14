#!/usr/bin/env python3
"""Numerical regression tests for the synthetic V1 fixed-point input."""
import importlib.util
import itertools
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from boolean_networks import BNCase, DAGGER, bits_to_delta_index, build_successor_model, deterministic_values, flip_state, kl_values, make_cases, map_recovered_policy, map_recovered_rollout, rational_deterministic_values, raw_deterministic_values
STATES = tuple((f'{i:03b}' for i in range(8)))
EXPECTED_UPDATE = ('100', '100', '001', '001', '110', '100', '110', '001')
OLD_UPDATE = ('100', '100', '001', '001', '110', '100', '011', '001')
TARGET = '110'

def raw_policy_minima():
    actions = (0, 4, 2, 6)
    transients = tuple((i for i in range(8) if i != 6))
    targets = tuple((int(s, 2) for s in EXPECTED_UPDATE))
    successors = {i: tuple((targets[i ^ a] for a in actions)) for i in transients}
    best = {i: math.inf for i in transients}
    proper = 0
    examined = 0
    for indices in itertools.product(range(4), repeat=7):
        policy = {i: successors[i][j] for i, j in zip(transients, indices)}
        all_proper = True
        for start in transients:
            seen = set()
            x = start
            steps = 0
            while x != 6 and x not in seen:
                seen.add(x)
                x = policy[x]
                steps += 1
            if x == 6:
                best[start] = min(best[start], steps)
            else:
                all_proper = False
        proper += int(all_proper)
        examined += 1
    return (best, examined, proper)

class V1FixedPointTests(unittest.TestCase):

    def setUp(self):
        self.case = make_cases()['V1']

    def test_complete_truth_table(self):
        self.assertEqual(tuple((self.case.update(s) for s in STATES)), EXPECTED_UPDATE)

    def test_only_one_uncontrolled_row_replaced(self):
        differences = [s for s, old in zip(STATES, OLD_UPDATE) if self.case.update(s) != old]
        self.assertEqual(differences, ['110'])

    def test_invariant_singleton_and_marker(self):
        self.assertEqual(self.case.target, (TARGET,))
        self.assertEqual(self.case.update(TARGET), TARGET)
        self.assertEqual(''.join((TARGET[i - 1] for i in (1, 2, 3))), '110')
        self.assertEqual(len(self.case.target_set), 1)

    def test_original_witness_and_controlled_target_are_distinct(self):
        self.assertEqual(OLD_UPDATE[6], '011')
        self.assertEqual(OLD_UPDATE[6 ^ 2], TARGET)

    def test_family_cost_and_state_index_preserved(self):
        self.assertEqual(self.case.n, 3)
        self.assertEqual(self.case.flip_nodes, (1, 2))
        self.assertEqual(self.case.actions, [(), (1,), (2,), (1, 2)])
        self.assertTrue(all((self.case.rho(a) == 0 for a in self.case.actions)))
        self.assertEqual(bits_to_delta_index(TARGET), 2)
        self.assertEqual([bits_to_delta_index(s) for s in STATES], list(range(8, 0, -1)))

    def test_changed_controlled_edges_are_not_just_target_absorption(self):
        changed = []
        for s in self.case.transient_states:
            for a in self.case.actions:
                pre = flip_state(s, a)
                if OLD_UPDATE[int(pre, 2)] != self.case.update(pre):
                    changed.append((s, a))
        self.assertEqual(changed, [('000', (1, 2)), ('010', (1,)), ('100', (2,))])

    def test_exactness_against_exhaustive_raw_policies(self):
        best, examined, proper = raw_policy_minima()
        self.assertEqual(examined, 4 ** 7)
        self.assertGreater(proper, 0)
        comp = deterministic_values(self.case)
        raw = raw_deterministic_values(self.case)
        rational = rational_deterministic_values(self.case)
        for i, value in best.items():
            s = f'{i:03b}'
            self.assertEqual(comp[s], value)
            self.assertEqual(raw[s], value)
            self.assertEqual(rational[s], value)

    def test_recovered_policy_and_post_target_uncontrolled_stay(self):
        _, costs = kl_values(self.case)
        values = deterministic_values(self.case)
        for s in STATES:
            rows, _ = map_recovered_rollout(self.case, s, values_kl=costs)
            self.assertEqual(rows[-1]['state_bits'], TARGET)
            self.assertEqual(rows[-1]['t'], values[s])
            x = rows[-1]['state_bits']
            for _ in range(5):
                x = self.case.update(x)
                self.assertEqual(x, TARGET)

    def test_representatives_use_retained_tie_order(self):
        m = build_successor_model(self.case)
        for s in self.case.transient_states:
            for y in m.ysets[s]:
                actions = [a for a in self.case.actions if self.case.terminalized_successor(s, a) == y]
                expected = min(actions, key=lambda a: (self.case.rho(a), len(a), a))
                self.assertEqual(m.reps[s, y], expected)
        self.assertEqual(m.reps['010', DAGGER], (1,))

    def test_table_producer_rejects_noninvariant_target(self):
        path = Path(__file__).parent / 'run_exactness.py'
        spec = importlib.util.spec_from_file_location('v1_table_producer', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cases = make_cases()
        cases['V1'] = BNCase('V1', 3, (1, 2), ('110',), lambda s: OLD_UPDATE[int(s, 2)], {})
        with patch.object(mod, 'make_cases', return_value=cases):
            with self.assertRaisesRegex(ValueError, 'target.*invariant'):
                mod.build_rows()

    def test_all_retained_small_targets_remain_invariant_minimal(self):
        for case in make_cases().values():
            T = case.target_set
            self.assertTrue(T)
            self.assertTrue(all((case.update(t) in T for t in T)))
            for size in range(1, len(T)):
                for subset in itertools.combinations(T, size):
                    self.assertFalse(all((case.update(s) in subset for s in subset)))
if __name__ == '__main__':
    unittest.main()
