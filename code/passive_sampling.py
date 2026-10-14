#!/usr/bin/env python3
"""Finite passive sampling on explicit finite-state transition rows. Stored numerical reference results are not read by the update routine."""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Callable, Mapping
VERSION = 'passive-sampling-1.0'
TERMINAL = 'dagger'

@dataclass(frozen=True)
class Edge:
    successor: str
    probability: float
    action: tuple[int, ...]
    cost: float
    physical_successor: str

@dataclass(frozen=True)
class Kernel:
    states: tuple[str, ...]
    rows: Mapping[str, tuple[Edge, ...]]
    discounts: Mapping[str, float]
    zero_boundary: frozenset[str]
    beta: float
    n: int
    targets: frozenset[str]
    truth_table: Mapping[str, str]
    weights: Mapping[int, Fraction]

    @property
    def edge_count(self) -> int:
        return sum((len(row) for row in self.rows.values()))

    def value(self, z: Mapping[str, float], y: str) -> float:
        if y == TERMINAL:
            return 1.0
        if y in self.zero_boundary:
            return 0.0
        if y not in self.rows or y not in z:
            raise ValueError(f'Unknown/missing state {y!r}')
        v = float(z[y])
        if not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError('Desirability must be finite and in [0,1]')
        return v

    def residual(self, z: Mapping[str, float]) -> float:
        return max((abs(z[x] - self.discounts[x] * sum((e.probability * self.value(z, e.successor) for e in self.rows[x]))) for x in self.states), default=0.0)

def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')

def flip(x: str, action: tuple[int, ...]) -> str:
    bits = list(x)
    if len(set(action)) != len(action) or any((not 1 <= i <= len(x) for i in action)):
        raise ValueError('Invalid flip action')
    for i in action:
        bits[i - 1] = '1' if bits[i - 1] == '0' else '0'
    return ''.join(bits)

def build_kernel(model: Mapping) -> Kernel:
    """Validate a complete Boolean truth table and build full passive transition rows."""
    n = model['n']
    _integer('n', n, 1)
    states = tuple((format(i, f'0{n}b') for i in range(2 ** n)))
    table = dict(model['truth_table'])
    if set(table) != set(states) or any((y not in table for y in table.values())):
        raise ValueError('A complete, closed Boolean truth table is required')
    targets = frozenset(model['targets'])
    if not targets or not targets <= set(states) or any((table[x] not in targets for x in targets)):
        raise ValueError('Targets must be a nonempty invariant subset')
    nodes = tuple(model['eligible_nodes'])
    if len(set(nodes)) != len(nodes) or any((type(i) is not int or not 1 <= i <= n for i in nodes)):
        raise ValueError('Invalid eligible-node list')
    weights = {int(i): Fraction(str(v)) for i, v in model['weights_decimal'].items()}
    if set(weights) != set(nodes) or any((v < 0 for v in weights.values())):
        raise ValueError('Every eligible node needs a nonnegative weight')
    actions = tuple((a for k in range(len(nodes) + 1) for a in combinations(sorted(nodes), k)))
    q = float(model['state_cost_decimal'])
    beta = math.exp(-q)
    if not math.isfinite(q) or q <= 0 or (not 0 < beta < 1):
        raise ValueError('State-cost exponential must be resolvable in (0,1)')
    reps: dict[str, dict[str, tuple]] = {}
    reverse: dict[str, list[str]] = {TERMINAL: []}
    for x in states:
        if x in targets:
            continue
        rows = {}
        for a in actions:
            physical = table[flip(x, a)]
            y = TERMINAL if physical in targets else physical
            cost = sum((weights[i] for i in a), Fraction(0))
            key = (cost, len(a), a)
            if y not in rows or key < rows[y][0]:
                rows[y] = (key, physical)
        reps[x] = rows
        for y in rows:
            reverse.setdefault(y, []).append(x)
    reached = {TERMINAL}
    queue = deque([TERMINAL])
    while queue:
        for x in reverse.get(queue.popleft(), ()):
            if x not in reached:
                reached.add(x)
                queue.append(x)
    transient = tuple((x for x in states if x in reached))
    rows = {}
    zero = set()
    for x in transient:
        ys = sorted(reps[x])
        scores = [math.exp(-float(reps[x][y][0][0])) for y in ys]
        if any((w <= 0 for w in scores)):
            raise ValueError('Feasible reference weight underflow')
        total = sum(scores)
        row = []
        for y, w in zip(ys, scores):
            key, physical = reps[x][y]
            row.append(Edge(y, w / total, key[2], float(key[0]), physical))
            if y != TERMINAL and y not in reached:
                zero.add(y)
        rows[x] = tuple(row)
    return Kernel(transient, rows, {x: beta for x in transient}, frozenset(zero), beta if transient else 0.0, n, targets, table, weights)

def dense_reference(kernel: Kernel) -> dict[str, float]:
    """Solve a finite reference system by partial-pivot Gaussian elimination, outside sampled updates."""
    states = kernel.states
    idx = {x: i for i, x in enumerate(states)}
    n = len(states)
    a = [[float(i == j) for j in range(n)] + [0.0] for i in range(n)]
    for x in states:
        i = idx[x]
        for e in kernel.rows[x]:
            v = kernel.discounts[x] * e.probability
            if e.successor == TERMINAL:
                a[i][-1] += v
            elif e.successor in idx:
                a[i][idx[e.successor]] -= v
    for c in range(n):
        pivot = max(range(c, n), key=lambda i: abs(a[i][c]))
        if a[pivot][c] == 0:
            raise ArithmeticError('Singular system')
        a[c], a[pivot] = (a[pivot], a[c])
        for i in range(c + 1, n):
            ratio = a[i][c] / a[c][c]
            for j in range(c, n + 1):
                a[i][j] -= ratio * a[c][j]
    v = [0.0] * n
    for i in range(n - 1, -1, -1):
        v[i] = (a[i][-1] - sum((a[i][j] * v[j] for j in range(i + 1, n)))) / a[i][i]
    return {**dict(zip(states, v)), TERMINAL: 1.0}

def induced_policy(kernel: Kernel, z: Mapping[str, float]) -> tuple[dict, dict, list[str]]:
    laws, modes, undefined = ({}, {}, [])
    for x in kernel.states:
        vals = {e.successor: e.probability * kernel.value(z, e.successor) for e in kernel.rows[x]}
        total = sum(vals.values())
        if total <= 0:
            undefined.append(x)
            continue
        laws[x] = {y: v / total for y, v in vals.items()}
        feasible = [e for e in kernel.rows[x] if kernel.value(z, e.successor) > 0]
        e = min(feasible, key=lambda e: (-math.log(e.probability) - math.log(kernel.value(z, e.successor)), 2 ** kernel.n + 1 if e.successor == TERMINAL else 2 ** kernel.n - int(e.successor, 2), len(e.action), e.action))
        modes[x] = {'successor': e.successor, 'action': list(e.action), 'physical_successor': e.physical_successor}
    return (laws, modes, undefined)

def policy_paths(kernel: Kernel, modes: Mapping) -> dict:
    answer = {}
    for start in kernel.states:
        x, seen, steps, effort = (start, set(), 0, Fraction(0))
        while x != TERMINAL and x not in seen and (x in modes):
            seen.add(x)
            a = tuple(modes[x]['action'])
            if any((i not in kernel.weights for i in a)):
                raise ValueError('Policy contains an ineligible action')
            physical = kernel.truth_table[flip(x, a)]
            y = TERMINAL if physical in kernel.targets else physical
            if y != modes[x]['successor'] or physical != modes[x]['physical_successor']:
                raise ValueError('Policy successor/action inconsistency')
            effort += sum((kernel.weights[i] for i in a), Fraction(0))
            x = y
            steps += 1
        answer[start] = {'proper': x == TERMINAL, 'hitting_time': steps if x == TERMINAL else None, 'intervention_cost': str(effort) if x == TERMINAL else None}
    return answer

def sample(kernel: Kernel, *, seed: int, max_updates: int=1000000, path_cap: int | None=None, scan_every: int=1000, omega: float=0.75, tolerance: float=0.001, observer: Callable[[dict], None] | None=None) -> dict:
    """Apply uniform restarts, inverse-CDF successor sampling and state-wise stepsizes with a hard update cap. A successful residual scan may interrupt an episode. All event counters remain explicit; residual bounds are floating estimates."""
    _integer('max_updates', max_updates, 0)
    _integer('scan_every', scan_every, 1)
    _integer('seed', seed, 0)
    path_cap = max(1, len(kernel.states)) if path_cap is None else path_cap
    _integer('path_cap', path_cap, 1)
    if not math.isfinite(omega) or not 0.5 < omega <= 1:
        raise ValueError('omega must be in (1/2,1]')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    states = kernel.states
    visits = {x: 0 for x in states}
    starts = {x: 0 for x in states}
    z = {**{x: 1.0 for x in states}, TERMINAL: 1.0}
    rng = random.Random(seed)
    work = dict.fromkeys(['sample_updates', 'episodes', 'sampler_edge_visits', 'residual_scans', 'residual_edge_visits', 'target_events', 'zero_boundary_events', 'cap_events', 'interrupted_episodes'], 0)
    last_scan = -1
    last_snapshot: dict = {}

    def scan(residual: float | None=None) -> dict:
        nonlocal last_scan, last_snapshot
        r = kernel.residual(z) if residual is None else residual
        work['residual_scans'] += 1
        work['residual_edge_visits'] += kernel.edge_count
        last_scan = work['sample_updates']
        last_snapshot = {'updates': last_scan, 'residual': r, 'error_bound_estimate': r / (1 - kernel.beta), 'z': dict(z), 'visits': dict(visits), 'restarts': dict(starts), 'work': dict(work), 'rng_state_sha256': hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()}
        if observer is not None:
            observer(json.loads(json.dumps(last_snapshot)))
        return last_snapshot
    status = 'BUDGET_EXHAUSTED'
    if not states:
        status = 'EMPTY_TRANSIENT_BASIN'
        scan()
    else:
        while work['sample_updates'] < max_updates:
            x = rng.choice(states)
            starts[x] += 1
            work['episodes'] += 1
            stop = False
            for step in range(path_cap):
                row = kernel.rows[x]
                draw, cumulative = (rng.random(), 0.0)
                chosen = row[-1]
                for reads, edge in enumerate(row, 1):
                    cumulative += edge.probability
                    if draw < cumulative:
                        chosen = edge
                        break
                y = chosen.successor
                visits[x] += 1
                alpha = visits[x] ** (-omega)
                z[x] = (1.0 - alpha) * z[x] + alpha * kernel.discounts[x] * kernel.value(z, y)
                work['sample_updates'] += 1
                work['sampler_edge_visits'] += reads
                ended = y == TERMINAL or y in kernel.zero_boundary or step + 1 == path_cap
                if y == TERMINAL:
                    work['target_events'] += 1
                elif y in kernel.zero_boundary:
                    work['zero_boundary_events'] += 1
                elif step + 1 == path_cap:
                    work['cap_events'] += 1
                needs_scan = work['sample_updates'] % scan_every == 0 or work['sample_updates'] == max_updates
                if needs_scan:
                    r = kernel.residual(z)
                    success = min(visits.values()) > 0 and r / (1 - kernel.beta) <= tolerance
                    stop = success or work['sample_updates'] == max_updates
                    if stop and (not ended):
                        work['interrupted_episodes'] += 1
                    scan(r)
                    if success:
                        status = 'RESIDUAL_TOLERANCE_MET'
                if stop or ended:
                    break
                x = y
            if stop:
                break
        if last_scan != work['sample_updates']:
            scan()
    laws, modes, undefined = induced_policy(kernel, z)
    return {'implementation': VERSION, 'seed': seed, 'status': status, 'settings': {'max_updates': max_updates, 'path_cap': path_cap, 'scan_every': scan_every, 'omega': omega, 'tolerance': tolerance}, 'states': list(states), 'z': dict(z), 'visits': dict(visits), 'restarts': dict(starts), 'work': dict(work), 'residual': last_snapshot['residual'], 'error_bound_estimate': last_snapshot['error_bound_estimate'], 'last_rng_state_sha256': last_snapshot['rng_state_sha256'], 'law': laws, 'modes': modes, 'undefined_policy_rows': undefined, 'policy_paths': policy_paths(kernel, modes), 'bound_kind': 'floating_residual_estimate_not_interval_certificate'}
