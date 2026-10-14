#!/usr/bin/env python3
"""Explicit dense and sparse numerical backends, finite sampling interfaces and sparse policy validation."""
from __future__ import annotations
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from boolean_networks import Action, BNCase, DAGGER, State, SuccessorModel, bits_to_delta_index, build_successor_model, flip_state, graph_controllable_basin, state_sort_key, _solve_linear_system_dense

class NumericalResolutionError(RuntimeError):
    pass

class SolverNotConverged(RuntimeError):

    def __init__(self, result: 'SolveResult') -> None:
        self.result = result
        super().__init__(f'{result.backend}: {result.status}; residual={result.residual:.6g}')

@dataclass(frozen=True)
class Edge:
    successor: State
    probability: float
    action: Action
    cost: float
    physical_successor: State

@dataclass(frozen=True)
class SparseKernel:
    case: BNCase
    rows: dict[State, tuple[Edge, ...]]
    zero_boundary: frozenset[State]
    discounts: dict[State, float]
    beta: float
    epsilon: float
    construction: dict[str, int]

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(self.rows)

    @property
    def edge_count(self) -> int:
        return sum((len(row) for row in self.rows.values()))

    def value(self, z: Mapping[State, float], y: State) -> float:
        if y == DAGGER:
            return 1.0
        if y in self.zero_boundary:
            return 0.0
        if y not in self.rows or y not in z:
            raise ValueError(f'Missing/unknown desirability state {y!r}')
        value = float(z[y])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f'Invalid desirability at {y!r}: {value!r}')
        return value

    def apply(self, z: Mapping[State, float]) -> dict[State, float]:
        return {x: self.discounts[x] * sum((e.probability * self.value(z, e.successor) for e in row)) for x, row in self.rows.items()}

    def residual(self, z: Mapping[State, float]) -> float:
        image = self.apply(z)
        return max((abs(z[x] - image[x]) for x in self.rows), default=0.0)

def make_sparse_kernel(case: BNCase, epsilon: float=0.0, *, model: SuccessorModel | None=None, state_cost: float | Mapping[State, float]=1.0) -> SparseKernel:
    if not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
        raise ValueError('epsilon must be finite and in [0,1)')
    if not case.target or len(set(case.target)) != len(case.target):
        raise ValueError('Targets must be nonempty and distinct')
    if not case.actions or () not in case.actions:
        raise ValueError('The finite action family must include the empty action')
    model = build_successor_model(case, retain_classes=False) if model is None else model
    if model.case != case:
        raise ValueError('Successor model does not belong to the supplied case')
    basin = graph_controllable_basin(model)
    discounts: dict[State, float] = {}
    qs: dict[State, float] = {}
    for x in model.ysets:
        q = float(state_cost[x] if isinstance(state_cost, Mapping) else state_cost)
        if not math.isfinite(q) or q <= 0:
            raise ValueError('Transient state costs must be finite and strictly positive')
        qs[x] = q
    beta = math.exp(-min(qs.values())) if qs else 0.0
    if qs and (not 0 < beta < 1):
        raise NumericalResolutionError('The discount cannot be resolved inside (0,1)')
    rows: dict[State, tuple[Edge, ...]] = {}
    zero: set[State] = set()
    physical_checks = 0
    for x in model.ysets:
        if x not in basin:
            continue
        discounts[x] = math.exp(-qs[x])
        if discounts[x] == 0:
            raise NumericalResolutionError(f'State-cost exponential underflow at {x}')
        ys = model.ysets[x]
        weights = [math.exp(-model.rhostar[x, y]) for y in ys]
        if any((w == 0 or not math.isfinite(w) for w in weights)):
            raise NumericalResolutionError(f'Feasible passive probability underflow at {x}')
        gamma = sum(weights)
        out: list[Edge] = []
        for y, w in zip(ys, weights):
            p = (1.0 - epsilon) * (w / gamma) + epsilon / len(ys)
            a = model.reps[x, y]
            physical = case.update(flip_state(x, a))
            physical_checks += 1
            if len(physical) != case.n or any((c not in '01' for c in physical)):
                raise ValueError('Boolean update returned an invalid physical state')
            expected = DAGGER if physical in case.target_set else physical
            if expected != y:
                raise ValueError('Representative action does not realize its successor')
            if y != DAGGER and y not in basin:
                zero.add(y)
            out.append(Edge(y, p, a, model.rhostar[x, y], physical))
        if not math.isclose(sum((e.probability for e in out)), 1.0, rel_tol=0, abs_tol=2e-14):
            raise NumericalResolutionError('Passive row normalization failed')
        rows[x] = tuple(out)
    n0 = len(model.ysets)
    e0 = sum((len(row) for row in model.ysets.values()))
    construction = {'all_transient_states': n0, 'all_compressed_edges': e0, 'action_evaluations': n0 * len(case.actions), 'basin_reverse_edges': e0, 'basin_states': len(basin), 'retained_edges': sum((len(row) for row in rows.values())), 'zero_boundary_endpoints': len(zero), 'representative_physical_checks': physical_checks, 'retained_action_class_labels': sum((len(v) for v in model.classes.values()))}
    return SparseKernel(case, rows, frozenset(zero), discounts, beta, epsilon, construction)

@dataclass
class SolveResult:
    z: dict[State, float]
    status: str
    backend: str
    residual: float
    error_bound_estimate: float
    iterations: int
    work: dict[str, int]
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status in {'RESIDUAL_TOLERANCE_MET', 'EMPTY_TRANSIENT_BASIN'}

    def costs(self, kernel: SparseKernel) -> dict[State, float]:
        if not self.converged:
            raise SolverNotConverged(self)
        if any((self.z[x] <= 0 for x in kernel.rows)):
            raise NumericalResolutionError('Zero numerical value on a graph-reachable state')
        return {**{x: -math.log(self.z[x]) for x in kernel.rows}, DAGGER: 0.0}

def _positive_int(name: str, value: int, *, allow_zero: bool=False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {('nonnegative' if allow_zero else 'positive')} integer")

def solve_desirability(kernel: SparseKernel, *, backend: str='auto', tolerance: float=1e-14, max_sweeps: int=100000) -> SolveResult:
    """Solve a finite desirability system using the requested numerical backend and explicit termination status."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    _positive_int('max_sweeps', max_sweeps, allow_zero=True)
    valid = {'auto', 'python_dense', 'numpy_dense', 'sparse_iteration'}
    if backend not in valid:
        raise ValueError(f'backend must be one of {sorted(valid)}')
    n, ecount = (len(kernel.rows), kernel.edge_count)
    if not n:
        return SolveResult({DAGGER: 1.0}, 'EMPTY_TRANSIENT_BASIN', 'none', 0, 0, 0, {'operator_edge_visits': 0, 'residual_edge_visits': 0, 'dense_matrix_entries': 0, 'dense_augmented_entries': 0})
    if backend == 'auto':
        if n <= 128:
            backend = 'python_dense'
        else:
            try:
                import numpy
            except ImportError:
                backend = 'sparse_iteration'
            else:
                backend = 'numpy_dense'
    work = {'operator_edge_visits': 0, 'residual_edge_visits': 0, 'dense_matrix_entries': 0, 'dense_augmented_entries': 0}
    z = {x: 1.0 for x in kernel.rows}
    iterations = 0
    if backend == 'sparse_iteration':
        for iterations in range(1, max_sweeps + 1):
            z = kernel.apply(z)
            work['operator_edge_visits'] += ecount
            residual = kernel.residual(z)
            work['residual_edge_visits'] += ecount
            if residual / (1.0 - kernel.beta) <= tolerance:
                break
    else:
        states = kernel.states
        index = {x: i for i, x in enumerate(states)}
        if backend == 'numpy_dense':
            import numpy as np
            a = np.eye(n, dtype=float)
            b = np.zeros(n, dtype=float)
        else:
            a = [[float(i == j) for j in range(n)] for i in range(n)]
            b = [0.0] * n
            work['dense_augmented_entries'] = n * (n + 1)
        work['dense_matrix_entries'] = n * n
        for x, row in kernel.rows.items():
            i = index[x]
            for edge in row:
                coeff = kernel.discounts[x] * edge.probability
                if edge.successor == DAGGER:
                    b[i] += coeff
                elif edge.successor in index:
                    a[i][index[edge.successor]] -= coeff
        work['operator_edge_visits'] = ecount
        sol = np.linalg.solve(a, b) if backend == 'numpy_dense' else _solve_linear_system_dense(a, b)
        z = {x: float(sol[i]) for x, i in index.items()}
        iterations = 1
    z[DAGGER] = 1.0
    if any((not math.isfinite(z[x]) or not 0 <= z[x] <= 1 for x in kernel.rows)):
        raise NumericalResolutionError('Solver returned a nonfinite or out-of-range desirability')
    residual = kernel.residual(z)
    work['residual_edge_visits'] += ecount
    bound = residual / (1.0 - kernel.beta)
    status = 'RESIDUAL_TOLERANCE_MET' if bound <= tolerance else 'BUDGET_EXHAUSTED'
    if backend != 'sparse_iteration' and status == 'BUDGET_EXHAUSTED':
        status = 'RESIDUAL_TOLERANCE_NOT_MET'
    if any((z[x] == 0 for x in kernel.rows)):
        status = 'NUMERICAL_RESOLUTION_FAILURE'
    return SolveResult(z, status, backend, residual, bound, iterations, work, {'tolerance': tolerance, 'domain_source': 'graph_reachability', 'bound_kind': 'floating_residual_estimate_not_interval_certificate'})

def induced_law(kernel: SparseKernel, z: Mapping[State, float]) -> tuple[dict, tuple[State, ...]]:
    law: dict[State, dict[State, float]] = {}
    undefined: list[State] = []
    for x, row in kernel.rows.items():
        weights = {e.successor: e.probability * kernel.value(z, e.successor) for e in row}
        total = sum(weights.values())
        if total <= 0:
            undefined.append(x)
        else:
            law[x] = {y: w / total for y, w in weights.items()}
    return (law, tuple(undefined))

def _sample(row: tuple[Edge, ...], rng: random.Random) -> tuple[State, int]:
    draw = rng.random()
    acc = 0.0
    for reads, edge in enumerate(row, 1):
        acc += edge.probability
        if draw < acc:
            return (edge.successor, reads)
    return (row[-1].successor, len(row))

def z_learning(kernel: SparseKernel, *, episodes: int=10000, path_cap: int=20, check_every: int=50, omega: float=0.75, tolerance: float=1e-06, seed: int=0, schedule: str='independent_restarts') -> SolveResult:
    """Run the episode-budgeted numerical sampling interface with explicit restart, visit and residual-scan accounting."""
    _positive_int('episodes', episodes, allow_zero=True)
    _positive_int('path_cap', path_cap)
    _positive_int('check_every', check_every)
    if not math.isfinite(omega) or not 0.5 < omega <= 1:
        raise ValueError('omega must be in (1/2,1]')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if schedule not in {'independent_restarts', 'cyclic_restarts'}:
        raise ValueError('Unknown restart schedule')
    rng = random.Random(seed)
    states = kernel.states
    z = {x: 1.0 for x in states}
    z[DAGGER] = 1.0
    visits = {x: 0 for x in states}
    starts = {x: 0 for x in states}
    work = {'sample_updates': 0, 'sampler_edge_visits': 0, 'residual_scans': 0, 'residual_edge_visits': 0, 'policy_edge_visits': 0, 'target_events': 0, 'zero_boundary_events': 0, 'cap_events': 0}
    i, residual, bound = (0, 0.0, 0.0)
    status = 'EMPTY_TRANSIENT_BASIN' if not states else 'BUDGET_EXHAUSTED'
    for i in range(1, episodes + 1) if states else ():
        x = rng.choice(states) if schedule == 'independent_restarts' else states[(i - 1) % len(states)]
        starts[x] += 1
        for step in range(path_cap):
            y, reads = _sample(kernel.rows[x], rng)
            visits[x] += 1
            alpha = visits[x] ** (-omega)
            z[x] = (1.0 - alpha) * z[x] + alpha * kernel.discounts[x] * kernel.value(z, y)
            work['sample_updates'] += 1
            work['sampler_edge_visits'] += reads
            if y == DAGGER:
                work['target_events'] += 1
                break
            if y in kernel.zero_boundary:
                work['zero_boundary_events'] += 1
                break
            x = y
        else:
            work['cap_events'] += 1
        if i % check_every == 0 or i == episodes:
            residual = kernel.residual(z)
            work['residual_scans'] += 1
            work['residual_edge_visits'] += kernel.edge_count
            bound = residual / (1.0 - kernel.beta)
            if min(visits.values()) > 0 and bound <= tolerance:
                status = 'RESIDUAL_TOLERANCE_MET'
                break
    if states and (not work['residual_scans']):
        residual = kernel.residual(z)
        bound = residual / (1.0 - kernel.beta)
        work['residual_scans'] += 1
        work['residual_edge_visits'] += kernel.edge_count
    law, undefined = induced_law(kernel, z)
    work['policy_edge_visits'] = kernel.edge_count
    return SolveResult(z, status, 'passive_sampling', residual, bound, i, work, {'visits': visits, 'restarts': starts, 'seed': seed, 'schedule': schedule, 'omega': omega, 'path_cap': path_cap, 'check_every': check_every, 'tolerance': tolerance, 'policy': law, 'undefined_policy_rows': undefined, 'minimum_visits': min(visits.values(), default=0), 'bound_kind': 'floating_residual_estimate_not_interval_certificate'})

@dataclass
class PolicyCheck:
    policy: dict[State, tuple[State, Action]]
    physical_successors: dict[State, State]
    pointers: dict[State, State]
    proper_states: frozenset[State]
    failed_checked_states: frozenset[State]
    status: str
    edge_visits: int

def check_policy(case: BNCase, policy: Mapping[State, tuple[State, Action]], *, checked_states: Sequence[State] | None=None, verify: bool=True) -> PolicyCheck:
    """Validate actual physical successors and traverse the complete recovered graph before restricting the reported checked set."""
    pointers: dict[State, State] = {}
    physical: dict[State, State] = {}
    reverse: dict[State, list[State]] = {DAGGER: []}
    allowed = set(case.actions)
    for x, (y, a) in policy.items():
        if x in case.target_set or len(x) != case.n or any((c not in '01' for c in x)):
            raise ValueError('Policy keys must be transient Boolean states')
        if a not in allowed:
            raise ValueError('Inadmissible recovered action')
        real = case.update(flip_state(x, a))
        terminalized = DAGGER if real in case.target_set else real
        if terminalized != y:
            raise ValueError('Recovered action does not realize the stored successor')
        if y != DAGGER and y not in policy:
            raise ValueError('Incomplete policy along a reachable policy path')
        physical[x] = real
        pointers[x] = y
        reverse.setdefault(y, []).append(x)
    checked = set(policy) | case.target_set if checked_states is None else set(checked_states)
    if not case.target_set <= checked or not checked <= set(policy) | case.target_set:
        raise ValueError('Checked set must contain targets and lie in the complete policy domain')
    if not verify:
        return PolicyCheck(dict(policy), physical, pointers, frozenset(), frozenset(), 'NOT_REQUESTED', 0)
    reached, queue, count = ({DAGGER}, deque([DAGGER]), 0)
    while queue:
        y = queue.popleft()
        for x in reverse.get(y, ()):
            count += 1
            if x not in reached:
                reached.add(x)
                queue.append(x)
    proper = reached - {DAGGER} | case.target_set
    failed = checked - proper
    return PolicyCheck(dict(policy), physical, pointers, frozenset(proper), frozenset(failed), 'CHECKED_DOMAIN_PASSES' if not failed else 'CHECKED_DOMAIN_FAILS', count)

def recover_from_desirability(kernel: SparseKernel, z: Mapping[State, float], *, checked_states: Sequence[State] | None=None, verify: bool=True) -> PolicyCheck:
    policy: dict[State, tuple[State, Action]] = {}
    for x, row in kernel.rows.items():
        finite = [e for e in row if e.successor == DAGGER or e.successor in kernel.rows]
        if not finite or any((kernel.value(z, e.successor) <= 0 for e in finite)):
            raise NumericalResolutionError('Cannot resolve every reachable continuation cost')
        best = min(finite, key=lambda e: (-math.log(e.probability) - math.log(kernel.value(z, e.successor)), state_sort_key(e.successor, kernel.case.n), len(e.action), e.action))
        policy[x] = (best.successor, best.action)
    return check_policy(kernel.case, policy, checked_states=checked_states, verify=verify)
