#!/usr/bin/env python3
"""Numerical regression tests for solver backends, restarts, boundaries and policy validation."""
import math
import unittest
from unittest.mock import patch
import boolean_networks as bn
import numerical_solvers as co

def fixture(name, table, target=('00',), nodes=(1,), weights=None):
    return bn.BNCase(name, len(next(iter(table))), nodes, target, table.__getitem__, {i: 0.0 for i in nodes} if weights is None else weights)

def partial():
    return fixture('partial', {'00': '00', '01': '01', '10': '11', '11': '11'})

def chain():
    return fixture('chain', {'00': '00', '01': '10', '10': '00', '11': '11'})

def targets():
    return fixture('two_cycle', {'00': '01', '01': '00', '10': '01', '11': '00'}, ('00', '01'))

class DomainTests(unittest.TestCase):

    def test_graph_basin_without_optimization(self):
        with patch.object(bn, 'deterministic_values', side_effect=AssertionError('oracle forbidden')):
            self.assertEqual(bn.controllable_basin(partial()), {'10'})

    def test_zero_endpoint_retains_probability(self):
        k = co.make_sparse_kernel(partial())
        self.assertEqual(k.zero_boundary, frozenset({'11'}))
        self.assertEqual([(e.successor, e.probability) for e in k.rows['10']], [('11', 0.5), (bn.DAGGER, 0.5)])
        z = co.solve_desirability(k).z
        self.assertAlmostEqual(z['10'], math.exp(-1) / 2, places=15)
        self.assertGreater(abs(z['10'] - math.exp(-1)), 0.1)

    def test_full_state_boundary_arrays(self):
        z, j = bn.kl_values(partial())
        for x in ['01', '11']:
            self.assertEqual(z[x], 0)
            self.assertEqual(j[x], math.inf)
        self.assertEqual(z['00'], 1)

    def test_no_unreachable_successor_recovery(self):
        p = bn.map_recovered_policy(partial())
        self.assertEqual(set(p), {'10'})
        self.assertEqual(p['10'][0], bn.DAGGER)

    def test_unresolved_reachable_cost_rejected(self):
        c = chain()
        m = bn.build_successor_model(c)
        with self.assertRaises(ValueError):
            bn.map_recovered_successor(c, '01', {bn.DAGGER: 0, '10': math.inf, '01': math.inf, '11': math.inf}, m)

    def test_disconnected_domain_not_numerical_cutoff(self):
        with patch.object(bn, 'deterministic_values', side_effect=AssertionError('forbidden')):
            self.assertEqual(co.make_sparse_kernel(partial()).states, ('10',))

    def test_empty_transient_basin(self):
        c = fixture('empty', {'00': '00', '01': '01', '10': '10', '11': '11'}, nodes=())
        k = co.make_sparse_kernel(c)
        self.assertEqual(k.states, ())
        self.assertEqual(co.solve_desirability(k).status, 'EMPTY_TRANSIENT_BASIN')
        self.assertEqual(co.z_learning(k).iterations, 0)

    def test_all_target(self):
        c = fixture('all_target', {'0': '1', '1': '0'}, ('0', '1'), nodes=())
        k = co.make_sparse_kernel(c)
        self.assertEqual(k.states, ())
        self.assertTrue(co.solve_desirability(k).converged)

    def test_lean_builder_discards_classes(self):
        c = targets()
        full = bn.build_successor_model(c)
        lean = bn.build_successor_model(c, retain_classes=False)
        self.assertEqual(lean.classes, {})
        self.assertEqual(full.ysets, lean.ysets)
        self.assertEqual(full.reps, lean.reps)
        k = co.make_sparse_kernel(c)
        self.assertEqual(k.construction['retained_action_class_labels'], 0)
        self.assertEqual(k.construction['action_evaluations'], 4)

    def test_partial_controllability_metrics(self):
        model = bn.build_successor_model(partial())
        self.assertEqual(model.N_transient, 3)
        self.assertEqual(model.N_B, 1)
        self.assertEqual(model.E_0, 6)
        self.assertEqual(model.E_B, 2)
        self.assertEqual(model.construction_edge_count, 6)
        self.assertEqual(model.raw_edge_count, 2)

    def test_empty_basin_ratio_is_undefined(self):
        c = fixture('empty', {'00': '00', '01': '01', '10': '10', '11': '11'}, nodes=())
        model = bn.build_successor_model(c)
        self.assertTrue(math.isnan(model.compression_factor))

    def test_physical_multitarget_modal_rollout(self):
        c = targets()
        rows, _ = bn.map_recovered_rollout(c, '10', max_steps=1)
        self.assertEqual(rows[-1]['state_bits'], '01')
        self.assertEqual(rows[-1]['state_delta'], bn.bits_to_delta_index('01'))

    def test_physical_multitarget_deterministic_rollout(self):
        rows, _ = bn.deterministic_rollout(targets(), '10', max_steps=1)
        self.assertEqual(rows[-1]['state_bits'], '01')

    def test_physical_multitarget_pointer(self):
        k = co.make_sparse_kernel(targets())
        p = co.recover_from_desirability(k, co.solve_desirability(k).z)
        self.assertEqual(p.physical_successors['10'], '01')
        self.assertEqual(p.pointers['10'], bn.DAGGER)

    def test_index_convention(self):
        for n in range(1, 6):
            for x in bn.all_states(n):
                self.assertEqual(bn.delta_index_to_bits(bn.bits_to_delta_index(x), n), x)

    def test_checked_subset_path_uses_other_states(self):
        c = chain()
        p = {'01': ('10', ()), '10': (bn.DAGGER, ()), '11': ('11', ())}
        r = co.check_policy(c, p, checked_states=['00', '01'])
        self.assertEqual(r.status, 'CHECKED_DOMAIN_PASSES')
        self.assertIn('10', r.proper_states)
        self.assertNotIn('11', r.proper_states)

    def test_cycles_fail_only_the_checked_set(self):
        c = chain()
        p = {'01': ('10', ()), '10': (bn.DAGGER, ()), '11': ('11', ())}
        r = co.check_policy(c, p)
        self.assertEqual(r.failed_checked_states, frozenset({'11'}))
        self.assertEqual(r.status, 'CHECKED_DOMAIN_FAILS')

    def test_incomplete_policy_rejected(self):
        with self.assertRaises(ValueError):
            co.check_policy(chain(), {'01': ('10', ())})

    def test_wrong_physical_pointer_rejected(self):
        with self.assertRaises(ValueError):
            co.check_policy(targets(), {'10': ('10', ()), '11': (bn.DAGGER, ())})

    def test_invalid_checked_domain(self):
        with self.assertRaises(ValueError):
            co.check_policy(chain(), {'01': ('10', ()), '10': (bn.DAGGER, ())}, checked_states=['01'])

    def test_optional_check(self):
        p = co.check_policy(chain(), {'01': ('10', ()), '10': (bn.DAGGER, ())}, verify=False)
        self.assertEqual(p.status, 'NOT_REQUESTED')

    def test_no_deterministic_oracle_in_recovery(self):
        c = chain()
        _, j = bn.kl_values(c)
        with patch.object(bn, 'deterministic_values', side_effect=AssertionError('forbidden')):
            with patch.object(bn, 'graph_controllable_basin', wraps=bn.graph_controllable_basin) as counter:
                p = bn.map_recovered_policy(c, j)
                self.assertEqual(counter.call_count, 1)
                self.assertEqual(len(p), 3)

    def test_improper_modal_cycle(self):
        c = fixture('modal_cycle', {'0': '0', '1': '1'}, ('0',), (1,), {1: 2.0})
        k = co.make_sparse_kernel(c, state_cost=0.1)
        r = co.solve_desirability(k)
        self.assertGreater(r.z['1'], 0)
        p = co.recover_from_desirability(k, r.z)
        self.assertEqual(p.status, 'CHECKED_DOMAIN_FAILS')
        self.assertEqual(p.pointers['1'], '1')

    def test_zero_step_rollout_budget(self):
        with self.assertRaises(RuntimeError):
            bn.map_recovered_rollout(targets(), '10', max_steps=0)
        with self.assertRaises(RuntimeError):
            bn.deterministic_rollout(targets(), '10', max_steps=0)

class SolverTests(unittest.TestCase):

    def test_sparse_python_dense_agreement(self):
        for c in [partial(), targets(), chain()]:
            k = co.make_sparse_kernel(c)
            a = co.solve_desirability(k, backend='python_dense')
            b = co.solve_desirability(k, backend='sparse_iteration')
            self.assertTrue(b.converged)
            self.assertLess(max((abs(a.z[x] - b.z[x]) for x in k.rows)), 2e-14)
            self.assertEqual(b.work['dense_matrix_entries'], 0)

    def test_numpy_dense_agreement(self):
        k = co.make_sparse_kernel(chain())
        a = co.solve_desirability(k, backend='python_dense')
        b = co.solve_desirability(k, backend='numpy_dense')
        for x in k.rows:
            self.assertAlmostEqual(a.z[x], b.z[x], places=14)

    def test_state_dependent_costs(self):
        k = co.make_sparse_kernel(partial(), state_cost={'01': 0.2, '10': 0.7, '11': 1.0})
        r = co.solve_desirability(k)
        self.assertAlmostEqual(r.z['10'], math.exp(-0.7) / 2, places=14)
        self.assertAlmostEqual(k.beta, math.exp(-0.2))

    def test_zero_sweep_budget_is_explicit(self):
        k = co.make_sparse_kernel(chain())
        r = co.solve_desirability(k, backend='sparse_iteration', max_sweeps=0)
        self.assertEqual(r.status, 'BUDGET_EXHAUSTED')
        self.assertEqual(r.iterations, 0)
        self.assertGreater(r.residual, 0)
        with self.assertRaises(co.SolverNotConverged):
            r.costs(k)

    def test_compatibility_failed_budget_raises(self):
        with self.assertRaises(co.SolverNotConverged):
            bn.kl_values(chain(), backend='sparse_iteration', max_iter=0)

    def test_bad_dense_output_not_clipped(self):
        k = co.make_sparse_kernel(partial())
        with patch.object(co, '_solve_linear_system_dense', return_value=[-1e-15]):
            with self.assertRaises(co.NumericalResolutionError):
                co.solve_desirability(k)

    def test_reachable_zero_not_called_unreachable(self):
        k = co.make_sparse_kernel(partial())
        with patch.object(co, '_solve_linear_system_dense', return_value=[0.0]):
            r = co.solve_desirability(k)
            self.assertEqual(r.status, 'NUMERICAL_RESOLUTION_FAILURE')
            self.assertEqual(k.states, ('10',))

    def test_no_silent_solver_fallback(self):
        with patch.object(co, '_solve_linear_system_dense', side_effect=ValueError('solver fault')):
            with self.assertRaisesRegex(ValueError, 'solver fault'):
                co.solve_desirability(co.make_sparse_kernel(partial()))

    def test_numpy_unavailable_auto_fallback(self):
        n = 8
        c = bn.BNCase('large', n, (), ('0' * n,), lambda s: '0' * n, {})
        with patch.dict('sys.modules', {'numpy': None}):
            r = co.solve_desirability(co.make_sparse_kernel(c))
        self.assertEqual(r.backend, 'sparse_iteration')
        self.assertTrue(r.converged)

    def test_residual_estimate_bound_on_closed_form(self):
        k = co.make_sparse_kernel(partial())
        z = {'10': 0.2, bn.DAGGER: 1}
        residual = k.residual(z)
        error = abs(z['10'] - math.exp(-1) / 2)
        self.assertLessEqual(error, residual / (1 - k.beta) + 1e-15)

    def test_residual_scan_cost_is_counted(self):
        k = co.make_sparse_kernel(chain())
        r = co.solve_desirability(k, backend='sparse_iteration')
        self.assertEqual(r.work['operator_edge_visits'], r.iterations * k.edge_count)
        self.assertEqual(r.work['residual_edge_visits'], (r.iterations + 1) * k.edge_count)

    def test_smoothing_zero_and_positive(self):
        c = partial()
        a = co.make_sparse_kernel(c, 0)
        b = co.make_sparse_kernel(c, 0.01)
        self.assertEqual(a.zero_boundary, b.zero_boundary)
        self.assertEqual(a.beta, b.beta)
        self.assertEqual(a.states, b.states)

    def test_nonuniform_smoothing_uses_actual_probabilities(self):
        c = fixture('weighted', {'00': '00', '01': '01', '10': '11', '11': '11'}, weights={1: 2.0})
        k = co.make_sparse_kernel(c, 0.1)
        terminal = next((e.probability for e in k.rows['10'] if e.successor == bn.DAGGER))
        r = co.solve_desirability(k)
        self.assertAlmostEqual(r.z['10'], math.exp(-1) * terminal, places=15)

    def test_feasible_probability_underflow_is_reported(self):
        c = fixture('huge', {'00': '00', '01': '01', '10': '11', '11': '11'}, weights={1: 1000})
        with self.assertRaises(co.NumericalResolutionError):
            co.make_sparse_kernel(c)

    def test_invalid_inputs(self):
        for epsilon in [-0.1, 1.0, float('nan')]:
            with self.assertRaises(ValueError):
                co.make_sparse_kernel(partial(), epsilon)
        for q in [0, -1, float('inf')]:
            with self.assertRaises(ValueError):
                co.make_sparse_kernel(partial(), state_cost=q)
        k = co.make_sparse_kernel(chain())
        for tol in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                co.solve_desirability(k, tolerance=tol)

    def test_missing_desirability_is_not_zero_boundary(self):
        k = co.make_sparse_kernel(chain())
        with self.assertRaises(ValueError):
            k.apply({})

class SamplingTests(unittest.TestCase):

    def test_independent_restarts_cover_states(self):
        k = co.make_sparse_kernel(chain())
        r = co.z_learning(k, episodes=300, path_cap=1, seed=29, tolerance=1e-30)
        self.assertEqual(sum(r.diagnostics['restarts'].values()), r.iterations)
        self.assertGreater(min(r.diagnostics['restarts'].values()), 0)
        self.assertEqual(r.work['sample_updates'], r.iterations)

    def test_cyclic_visitation_exactly_one_per_state(self):
        k = co.make_sparse_kernel(chain())
        r = co.z_learning(k, episodes=len(k.rows), path_cap=1, schedule='cyclic_restarts', tolerance=1e-30)
        self.assertTrue(all((v == 1 for v in r.diagnostics['visits'].values())))

    def test_sampling_reproducibility(self):
        k = co.make_sparse_kernel(chain())
        kw = dict(episodes=120, path_cap=2, seed=17, tolerance=1e-30)
        self.assertEqual(co.z_learning(k, **kw), co.z_learning(k, **kw))

    def test_zero_budget_is_not_convergence(self):
        r = co.z_learning(co.make_sparse_kernel(chain()), episodes=0)
        self.assertEqual(r.status, 'BUDGET_EXHAUSTED')
        self.assertEqual(r.work['sample_updates'], 0)
        self.assertEqual(r.work['residual_scans'], 1)
        self.assertGreater(r.residual, 0)

    def test_finite_budget_exhaustion(self):
        r = co.z_learning(co.make_sparse_kernel(chain()), episodes=1, path_cap=1, tolerance=1e-30)
        self.assertEqual(r.status, 'BUDGET_EXHAUSTED')

    def test_zero_boundary_sampling_not_conditioned(self):
        k = co.make_sparse_kernel(partial())
        r = co.z_learning(k, episodes=400, path_cap=20, tolerance=1e-30, seed=9)
        self.assertGreater(r.work['zero_boundary_events'], 0)
        self.assertGreater(r.work['target_events'], 0)
        self.assertEqual(r.work['sample_updates'], 400)
        self.assertEqual(r.work['zero_boundary_events'] + r.work['target_events'], 400)

    def test_single_successor_finite_stop(self):
        c = fixture('single', {'00': '00', '01': '00', '10': '00', '11': '00'}, nodes=())
        k = co.make_sparse_kernel(c)
        r = co.z_learning(k, episodes=3, path_cap=1, check_every=1, schedule='cyclic_restarts')
        self.assertTrue(r.converged)
        self.assertEqual(r.iterations, 3)
        self.assertEqual(r.residual, 0)

    def test_undefined_finite_policy_row_is_reported(self):
        k = co.make_sparse_kernel(chain())
        law, undefined = co.induced_law(k, {x: 0.0 for x in k.rows})
        self.assertIn('01', undefined)
        self.assertNotIn('01', law)
        self.assertEqual(law['10'][bn.DAGGER], 1)

    def test_sample_counts_and_scan_counts(self):
        k = co.make_sparse_kernel(chain())
        r = co.z_learning(k, episodes=11, path_cap=3, check_every=4, tolerance=1e-30)
        self.assertEqual(r.work['residual_scans'], 3)
        self.assertEqual(r.work['residual_edge_visits'], 3 * k.edge_count)
        self.assertEqual(sum(r.diagnostics['visits'].values()), r.work['sample_updates'])
        self.assertLessEqual(r.work['sample_updates'], 33)

    def test_invalid_sampling_settings(self):
        k = co.make_sparse_kernel(chain())
        for kwargs in [dict(path_cap=0), dict(check_every=0), dict(episodes=-1), dict(omega=0.5), dict(omega=2), dict(schedule='guess')]:
            with self.assertRaises(ValueError):
                co.z_learning(k, **kwargs)
if __name__ == '__main__':
    unittest.main()
