#!/usr/bin/env python3
"""Numerical regression tests for decimal-weight gaps, ties and properness-sensitive diagnostics."""
import math
import unittest
from fractions import Fraction as F
from boolean_networks import BNCase, DAGGER, build_successor_model, rational_deterministic_values, rational_action_cost, strict_successor_gap, numerical_gap_test, make_cases, recovery_evaluation, recovery_evaluations, kl_values, map_recovered_successor, map_recovered_rollout

class GapTests(unittest.TestCase):

    def test_unique_minimum(self):
        self.assertEqual(strict_successor_gap({'a': F(2), 'b': F(5)}), F(3))

    def test_all_tied_minima_excluded(self):
        self.assertEqual(strict_successor_gap({'a': F(2), 'b': F(2), 'c': F(5)}), F(3))

    def test_all_successors_tied(self):
        self.assertEqual(strict_successor_gap({'a': F(2), 'b': F(2)}), math.inf)

    def test_single_successor(self):
        self.assertEqual(strict_successor_gap({'a': F(2)}), math.inf)

    def test_near_tie_is_not_collapsed(self):
        self.assertEqual(strict_successor_gap({'a': F(1), 'b': F(1) + F(1, 10 ** 14)}), F(1, 10 ** 14))

    def test_empty_set_rejected(self):
        with self.assertRaises(ValueError):
            strict_successor_gap({})

    def test_inexact_gap_input_rejected(self):
        with self.assertRaises(TypeError):
            strict_successor_gap({'a': 0.1})

    def test_strict_boundary(self):
        self.assertFalse(numerical_gap_test(F(1), 0.5, 0))
        self.assertTrue(numerical_gap_test(F(1), 0.49, 0))
        self.assertFalse(numerical_gap_test(F(1), 0.51, 0))

    def test_numerical_guard_is_conservative(self):
        self.assertFalse(numerical_gap_test(F(1), 0.5 - 1e-11))
        self.assertTrue(numerical_gap_test(F(1), 0.5 - 1e-08))

    def test_infinite_gap(self):
        self.assertTrue(numerical_gap_test(math.inf, 100000.0))

    def test_invalid_mismatch_and_guard(self):
        for value in [math.nan, math.inf, -1.0]:
            with self.assertRaises(ValueError):
                numerical_gap_test(F(1), value)
        with self.assertRaises(ValueError):
            numerical_gap_test(F(1), 0.1, -1)

class DomainAndPolicyTests(unittest.TestCase):

    def test_decimal_arithmetic(self):
        c = make_cases()['R1'].with_weights({1: 0.1, 2: 0.2})
        self.assertEqual(rational_action_cost(c, (1, 2)), F(3, 10))

    def test_negative_weight_rejected(self):
        c = make_cases()['R1'].with_weights({1: -0.1})
        with self.assertRaises(ValueError):
            rational_deterministic_values(c)

    def test_nonfinite_weight_rejected(self):
        c = make_cases()['R1'].with_weights({1: math.inf})
        with self.assertRaises(ValueError):
            rational_deterministic_values(c)

    def test_representative_tie_order(self):
        c = make_cases()['V2']
        m = build_successor_model(c)
        self.assertTrue(all((a == () for a in m.reps.values())))

    def test_modal_tie_order(self):
        c = make_cases()['R1']
        m = build_successor_model(c)
        exact = rational_deterministic_values(c)
        values = {s: 0.0 for s in c.states}
        values[DAGGER] = 0.0
        values['10'] = 0.4
        self.assertEqual(map_recovered_successor(c, '11', values, m, exact)[0], '10')

    @staticmethod
    def trap_case():
        mapping = {'000': '000', '100': '000', '001': '001', '101': '001', '010': '000', '110': '001', '011': '001', '111': '001'}
        return BNCase('trap', 3, (1,), ('000',), mapping.__getitem__, {1: 0.4})

    def test_domain_and_unreachable_successors(self):
        c = self.trap_case()
        m = build_successor_model(c)
        v = rational_deterministic_values(c)
        z, j = kl_values(c)
        self.assertEqual(v['001'], math.inf)
        self.assertEqual(z['001'], 0)
        self.assertIn('001', m.ysets['010'])
        d = recovery_evaluation(c, '010', values_kl=j, model=m, values_det_exact=v)
        self.assertEqual(d.Delta, math.inf)
        self.assertEqual(d.map_successor, DAGGER)
        self.assertAlmostEqual(d.epsilon, 0)

    def test_unreachable_ghost_value_ignored(self):
        c = self.trap_case()
        m = build_successor_model(c)
        v = rational_deterministic_values(c)
        _, j = kl_values(c)
        j['001'] = 0.01
        d = recovery_evaluation(c, '010', values_kl=j, model=m, values_det_exact=v)
        self.assertEqual(d.Delta, math.inf)
        self.assertEqual(d.map_successor, DAGGER)

    def test_unreachable_state_rejected(self):
        c = self.trap_case()
        with self.assertRaises(ValueError):
            recovery_evaluation(c, '001')

    def test_missing_reachable_kl_value_rejected(self):
        c = make_cases()['R1']
        _, j = kl_values(c)
        j[DAGGER] = math.inf
        with self.assertRaises(ValueError):
            recovery_evaluation(c, '11', values_kl=j)

    def test_policy_may_cycle_despite_positive_desirability(self):
        c = BNCase('identity', 2, (1, 2), ('00',), lambda s: s, {1: 5.0, 2: 5.0}, ((), (1,), (2,)))
        m = build_successor_model(c)
        v = rational_deterministic_values(c)
        z, j = kl_values(c)
        self.assertTrue(all((z[s] > 0 for s in c.states)))
        self.assertEqual(map_recovered_successor(c, '11', j, m, v)[0], '11')
        d = recovery_evaluation(c, '11', values_kl=j, model=m, values_det_exact=v)
        self.assertLessEqual(d.local_loss, d.two_epsilon + 1e-12)
        with self.assertRaises(RuntimeError):
            map_recovered_rollout(c, '11', values_kl=j, model=m, values_det=v)

    def test_positive_near_optimal_loss_is_not_zeroed(self):
        c = make_cases()['R1'].with_weights({1: 1.000000000001, 2: 2.0})
        m = build_successor_model(c)
        v = rational_deterministic_values(c)
        j = {y: 0.0 for y in c.states}
        j[DAGGER] = 0.0
        j['10'] = 2.0
        d = recovery_evaluation(c, '11', values_kl=j, model=m, values_det_exact=v)
        self.assertEqual(d.local_loss, 1e-12)
        self.assertFalse(d.map_is_deterministic_optimal)
        self.assertEqual(d.Delta, 1e-12)

    def test_all_target_boundary(self):
        c = BNCase('all_target', 1, (), ('0', '1'), lambda s: '1' if s == '0' else '0', {})
        v = rational_deterministic_values(c)
        z, j = kl_values(c)
        self.assertTrue(all((a == F(0) for a in v.values())))
        self.assertTrue(all((a == 1.0 for a in z.values())))
        self.assertEqual(recovery_evaluations(c), [])

    def test_tied_modes_at_gap_equality(self):
        phi = {'a': F(0), 'b': F(1)}
        psi = {'a': F(1, 2), 'b': F(1, 2)}
        mismatch = max((abs(psi[y] - phi[y]) for y in phi))
        self.assertFalse(numerical_gap_test(strict_successor_gap(phi), float(mismatch), 0))
        self.assertTrue(all((phi[y] - min(phi.values()) <= 2 * mismatch for y in psi)))

    def test_unweighted_small_cases(self):
        for name in ['V2', 'V3', 'V4']:
            ds = recovery_evaluations(make_cases()[name])
            self.assertTrue(all((d.map_is_deterministic_optimal for d in ds)))
if __name__ == '__main__':
    unittest.main()
