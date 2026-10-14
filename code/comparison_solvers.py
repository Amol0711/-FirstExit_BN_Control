#!/usr/bin/env python3
"""Sparse comparison solvers and numerical result accounting. Deterministic costs are integer tenths; desirability iteration uses binary64 arithmetic."""
from __future__ import annotations
import hashlib
import heapq
import json
import math
import sys
from array import array
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter_ns
from typing import Any
from boolean_networks import BNCase, make_cases, phenotype9_case, flip_state
TERMINAL = -1
SCALE = 10
BETA = math.exp(-1.0)
METHODS = ('U-SP', 'C-SP', 'C-KL-MODE', 'TW-DP-H')

def configurations(settings: dict) -> list[dict]:
    c = settings['configurations']
    result = [{'id': k, 'model': k, 'weights': []} for k in c['small']]
    for model in c['recovery']:
        for w1 in c['weights_decimal']:
            for w2 in c['weights_decimal']:
                result.append({'id': f'{model}-w{w1}-{w2}', 'model': model, 'weights': [w1, w2]})
    result += [{'id': k, 'model': k, 'weights': ['1'] * 9} for k in c['phenotype']]
    if len(result) != c['locked_count'] or len({r['id'] for r in result}) != len(result):
        raise ValueError('Configuration count or identity mismatch')
    return result

def get_case(config: dict) -> BNCase:
    name = config['model']
    if name.startswith('P9-r'):
        return phenotype9_case(tuple(range(1, 10)), {i: 1.0 for i in range(1, 10)}, max_flip_size=int(name[-1]))
    case = make_cases()[name]
    if config.get('weights'):
        case = case.with_weights({i + 1: float(w) for i, w in enumerate(config['weights'])})
    return case

def costs10(case: BNCase, actions: tuple) -> tuple[int, ...]:
    out = []
    for action in actions:
        cost = sum((Fraction(str(case.weights.get(i, 0))) for i in action), Fraction()) * SCALE
        if cost.denominator != 1 or cost < 0:
            raise ValueError('Expected nonnegative exact-tenth intervention costs')
        out.append(int(cost))
    return tuple(out)

def validate_case(case: BNCase) -> dict:
    states = case.states
    target = case.target_set
    if not target or () not in case.actions or len(set(case.actions)) != len(case.actions):
        raise ValueError('Invalid target or admissible family')
    truth = {x: case.update(x) for x in states}
    if any((y not in truth for y in truth.values())):
        raise ValueError('Update outside state space')
    if any((truth[t] not in target for t in target)):
        raise ValueError('Target is not invariant')
    first = min(target)
    cycle = set()
    x = first
    while x not in cycle:
        cycle.add(x)
        x = truth[x]
    if x != first or cycle != target:
        raise ValueError('Target is not a single invariant cycle')
    markers = [i for i in range(case.n) if len({x[i] for x in target}) == 1]
    if not markers:
        raise ValueError('No common nonempty coordinate marker')
    acts = tuple(sorted(case.actions))
    costs10(case, acts)
    bound = (len(states) + 1) * (SCALE + max(costs10(case, acts)))
    if bound >= 2 ** 63:
        raise ValueError('Graph cost payload exceeds signed 64-bit range')
    binding = {'n': case.n, 'truth': truth, 'target': sorted(target), 'actions': acts, 'rho10': costs10(case, acts), 'q10': SCALE}
    return {'model_sha256': digest(binding), 'validation_update_calls': len(states), 'common_marker_indices': [i + 1 for i in markers], 'target_cycle_size': len(target)}

def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@dataclass
class Graph:
    case: BNCase
    actions: tuple
    rho10: tuple
    states: tuple[int, ...]
    targets: frozenset[int]
    rows: dict[int, array]
    compressed: bool
    calls: int

    def terminalize(self, physical: int) -> int:
        return TERMINAL if physical in self.targets else physical

    def edge_count(self) -> int:
        return sum((len(row) // 3 for row in self.rows.values()))

    def physical_next(self, x: int, a: int) -> int:
        bits = f'{x:0{self.case.n}b}'
        return int(self.case.update(flip_state(bits, self.actions[a])), 2)

def build_graph(case: BNCase, compressed: bool) -> Graph:
    actions = tuple(sorted(case.actions))
    rho = costs10(case, actions)
    targets = frozenset((int(t, 2) for t in case.target))
    states = tuple((x for x in reversed(range(2 ** case.n)) if x not in targets))
    rows = {}
    calls = 0
    for x in states:
        bits = f'{x:0{case.n}b}'
        if compressed:
            best = {}
            for a, action in enumerate(actions):
                physical = int(case.update(flip_state(bits, action)), 2)
                calls += 1
                y = TERMINAL if physical in targets else physical
                if y not in best or (rho[a], actions[a]) < (best[y][1], actions[best[y][0]]):
                    best[y] = (a, rho[a])
            data = []
            for y in sorted(best, key=lambda v: (v == TERMINAL, -v)):
                a, r = best[y]
                data.extend((y, a, r))
        else:
            data = []
            for a, action in enumerate(actions):
                physical = int(case.update(flip_state(bits, action)), 2)
                calls += 1
                y = TERMINAL if physical in targets else physical
                data.extend((y, a, rho[a]))
        rows[x] = array('q', data)
    return Graph(case, actions, rho, states, targets, rows, compressed, calls)

def basin(graph: Graph) -> tuple[int, ...]:
    reverse = {TERMINAL: []}
    for x, row in graph.rows.items():
        for j in range(0, len(row), 3):
            reverse.setdefault(row[j], []).append(x)
    found = {TERMINAL}
    todo = deque([TERMINAL])
    while todo:
        y = todo.popleft()
        for x in reverse.get(y, ()):
            if x not in found:
                found.add(x)
                todo.append(x)
    return tuple((x for x in graph.states if x in found))

def edge_key(graph: Graph, row: array, j: int) -> tuple:
    y, a = (row[j], row[j + 1])
    return (y == TERMINAL, -y, graph.actions[a])

def reverse_dijkstra(graph: Graph, domain: tuple[int, ...]) -> dict:
    domain_set = set(domain)
    reverse = {TERMINAL: []}
    scans = 0
    for x in domain:
        row = graph.rows[x]
        for j in range(0, len(row), 3):
            y = row[j]
            scans += 1
            if y == TERMINAL or y in domain_set:
                reverse.setdefault(y, []).append((x, SCALE + row[j + 2]))
    dist = {TERMINAL: 0}
    heap = [(0, TERMINAL)]
    settled = set()
    relaxed = 0
    pushes = 1
    while heap:
        d, y = heapq.heappop(heap)
        if y in settled:
            continue
        settled.add(y)
        for x, c in reverse.get(y, ()):
            relaxed += 1
            cand = d + c
            if x not in dist or cand < dist[x]:
                dist[x] = cand
                heapq.heappush(heap, (cand, x))
                pushes += 1
    if any((x not in dist for x in domain)):
        raise RuntimeError('Reverse reachability and cost solver disagree')
    return {'values': dist, 'status': 'EXACT_INTEGER_SOLUTION', 'sweeps': 0, 'edge_visits': scans + relaxed, 'heap_pushes': pushes, 'numeric_value_entries': len(dist)}

def reference_rows(graph: Graph, domain: tuple[int, ...]) -> dict[int, array]:
    if not graph.compressed:
        raise ValueError('Reference rows require distinct successors')
    rows = {}
    for x in domain:
        row = graph.rows[x]
        weights = [math.exp(-row[j + 2] / SCALE) for j in range(0, len(row), 3)]
        gamma = math.fsum(weights)
        rows[x] = array('d', (w / gamma for w in weights))
    return rows

def operator(graph: Graph, domain: tuple[int, ...], ref: dict, z: dict) -> dict:
    out = {TERMINAL: 1.0}
    for x in domain:
        row = graph.rows[x]
        out[x] = BETA * math.fsum((p * z.get(row[3 * j], 0.0) for j, p in enumerate(ref[x])))
    return out

def solve_kl(graph: Graph, domain: tuple[int, ...], ref: dict, tolerance: float=1e-12, max_sweeps: int=10000) -> dict:
    if tolerance <= 0 or max_sweeps < 0:
        raise ValueError('Invalid iteration parameters')
    z = {x: 1.0 for x in domain}
    z[TERMINAL] = 1.0
    edge_visits = 0
    nr = sum((len(ref[x]) for x in domain))
    residual = math.inf
    converged = not domain
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        nxt = operator(graph, domain, ref, z)
        check = operator(graph, domain, ref, nxt)
        edge_visits += 2 * nr
        residual = max((abs(nxt[x] - check[x]) for x in domain), default=0.0)
        z = nxt
        if residual / (1.0 - BETA) <= tolerance:
            converged = True
            break
    if not domain:
        residual = 0.0
    return {'values': z, 'status': 'NUMERICAL_RESIDUAL_MET' if converged else 'SWEEP_BUDGET_EXHAUSTED', 'sweeps': sweeps, 'residual': residual, 'error_estimate': residual / (1.0 - BETA), 'edge_visits': edge_visits, 'numeric_value_entries': 3 * (len(domain) + 1)}

def dp_sweep(graph: Graph, values: dict) -> tuple[dict, dict]:
    new = {TERMINAL: 0}
    actions = {}
    for x in graph.states:
        row = graph.rows[x]
        best = min(range(0, len(row), 3), key=lambda j: (SCALE + row[j + 2] + values[row[j]], edge_key(graph, row, j)))
        new[x] = SCALE + row[best + 2] + values[row[best]]
        actions[x] = (row[best], row[best + 1])
    return (new, actions)

def stationary_proper(policy: dict, domain: tuple[int, ...]) -> bool:
    reverse = {TERMINAL: []}
    for x in domain:
        if x not in policy:
            return False
        reverse.setdefault(policy[x][0], []).append(x)
    seen = {TERMINAL}
    q = deque([TERMINAL])
    while q:
        for x in reverse.get(q.popleft(), ()):
            if x not in seen:
                seen.add(x)
                q.append(x)
    return all((x in seen for x in domain))

def solve_dp(graph: Graph, domain: tuple[int, ...], early_stop: bool=True) -> dict:
    horizon = len(domain)
    cmax = SCALE + max(graph.rho10)
    penalty = (horizon + 1) * cmax
    values = {x: penalty for x in graph.states}
    values[TERMINAL] = 0
    layers = [None]
    fixed = False
    sweeps = 0
    for sweeps in range(1, horizon + 1):
        new, policy = dp_sweep(graph, values)
        layers.append(policy)
        fixed = all((new[x] == values[x] for x in new))
        values = new
        if early_stop and fixed and stationary_proper(policy, domain):
            break
    return {'values': values, 'layers': layers, 'horizon': horizon, 'penalty10': penalty, 'cmax10': cmax, 'fixed_point_early_stop': bool(early_stop and fixed and (sweeps < horizon)), 'sweeps': sweeps, 'edge_visits': sweeps * graph.edge_count(), 'full_horizon_edge_visits': horizon * graph.edge_count(), 'numeric_value_entries': 2 * (len(graph.states) + 1), 'status': 'EXACT_INTEGER_FINITE_HORIZON'}

def recover(graph: Graph, domain: tuple[int, ...], method: str, solved: dict) -> dict:
    near = 0
    entries = 0
    scores_out = {}
    if method == 'TW-DP-H':
        h = solved['horizon']
        stored = len(solved['layers']) - 1
        return {'kind': 'time_indexed', 'layers': solved['layers'], 'horizon': h, 'stored_layers': stored, 'near_tie_count': 0, 'logical_policy_entries': h * len(domain), 'stored_policy_entries': stored * len(graph.states)}
    policy = {}
    vals = solved['values']
    for x in domain:
        row = graph.rows[x]
        candidates = []
        for j in range(0, len(row), 3):
            y = row[j]
            if method == 'C-KL-MODE':
                zy = vals.get(y, 0.0)
                if zy <= 0 or not math.isfinite(zy):
                    continue
                score = row[j + 2] / SCALE - math.log(zy)
            else:
                if y not in vals:
                    continue
                score = row[j + 2] + vals[y]
            candidates.append((score, edge_key(graph, row, j), j))
        if not candidates:
            raise RuntimeError('No finite successor during recovery')
        candidates.sort()
        best = candidates[0]
        j = best[2]
        policy[x] = (row[j], row[j + 1])
        entries += 1
        if method == 'C-KL-MODE':
            threshold = 1e-10 * max(1.0, abs(best[0]))
            close = [t for t in candidates if t[0] - best[0] <= threshold]
            if len(close) > 1:
                near += 1
            scores_out[x] = {'best_score': best[0], 'near_tie_candidates': [row[t[2]] for t in close], 'chosen_score': best[0]}
    return {'kind': 'stationary', 'policy': policy, 'near_tie_count': near, 'mode_scores': scores_out, 'logical_policy_entries': entries, 'stored_policy_entries': entries}

def selected_edge(recovered: dict, x: int, t: int=0) -> tuple[int, int] | None:
    if recovered['kind'] == 'stationary':
        return recovered['policy'].get(x)
    remaining = recovered['horizon'] - t
    if remaining <= 0:
        return None
    k = min(remaining, recovered['stored_layers'])
    return recovered['layers'][k].get(x)

def verify_rollouts(graph: Graph, domain: tuple[int, ...], recovered: dict) -> list[dict]:
    records = []
    for initial in domain:
        x = initial
        seen = set()
        total = 0
        flips = 0
        flipped_nodes = 0
        t = 0
        path = []
        first = selected_edge(recovered, x, 0)
        status = 'PROPER'
        cap = recovered.get('horizon', len(domain))
        while x not in graph.targets:
            if t >= cap or (recovered['kind'] == 'stationary' and x in seen):
                status = 'IMPROPER'
                break
            seen.add(x)
            selection = selected_edge(recovered, x, t)
            if selection is None:
                status = 'MISSING_POLICY'
                break
            y, a = selection
            actual = graph.physical_next(x, a)
            if graph.terminalize(actual) != y:
                raise AssertionError('Recorded successor disagrees with physical update')
            total += SCALE + graph.rho10[a]
            flips += int(bool(graph.actions[a]))
            flipped_nodes += len(graph.actions[a])
            path.append([x, a, actual])
            x = actual
            t += 1
        records.append({'initial': initial, 'initial_bits': f'{initial:0{graph.case.n}b}', 'delta_index': 2 ** graph.case.n - initial, 'status': status, 'first_successor': None if first is None else first[0], 'first_action': None if first is None else first[1], 'cost10': total if status == 'PROPER' else None, 'hitting_time': t if status == 'PROPER' else None, 'nonempty_interventions': flips if status == 'PROPER' else None, 'flipped_nodes': flipped_nodes if status == 'PROPER' else None, 'path': path})
    return records

def deep_size(obj: Any, seen: set[int] | None=None) -> int:
    seen = set() if seen is None else seen
    ident = id(obj)
    if ident in seen:
        return 0
    seen.add(ident)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        return size + sum((deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items()))
    if isinstance(obj, (tuple, list, set, frozenset)):
        return size + sum((deep_size(v, seen) for v in obj))
    return size

def run_trial(config: dict, method: str, memory: bool=False) -> dict:
    if method not in METHODS:
        raise ValueError('Unknown comparison method')
    phase = {}
    t = perf_counter_ns()
    case = get_case(config)
    validation = validate_case(case)
    phase['model_validation_ns'] = perf_counter_ns() - t
    t = perf_counter_ns()
    graph = build_graph(case, method != 'U-SP')
    phase['graph_build_ns'] = perf_counter_ns() - t
    t = perf_counter_ns()
    domain = basin(graph)
    phase['basin_ns'] = perf_counter_ns() - t
    ref = {}
    t = perf_counter_ns()
    if method == 'C-KL-MODE':
        ref = reference_rows(graph, domain)
    phase['reference_ns'] = perf_counter_ns() - t if ref else 0
    t = perf_counter_ns()
    if method in ('U-SP', 'C-SP'):
        solved = reverse_dijkstra(graph, domain)
    elif method == 'C-KL-MODE':
        solved = solve_kl(graph, domain, ref)
    else:
        solved = solve_dp(graph, domain, early_stop=True)
    phase['solve_ns'] = perf_counter_ns() - t
    t = perf_counter_ns()
    recovered = recover(graph, domain, method, solved)
    phase['recovery_ns'] = perf_counter_ns() - t
    t = perf_counter_ns()
    outcomes = verify_rollouts(graph, domain, recovered)
    phase['properness_ns'] = perf_counter_ns() - t
    phase['total_no_diagnostic_ns'] = sum(phase.values())
    nedge = graph.edge_count()
    ebasin = sum((len(graph.rows[x]) // 3 for x in domain))
    values = {str(k): v for k, v in solved['values'].items()}
    response = {'config': config['id'], 'method': method, 'phases': phase, 'validation': validation, 'N_total': 2 ** case.n, 'N0': len(graph.states), 'N_B': len(domain), 'A': len(graph.actions), 'E_all': nedge, 'E_B': ebasin, 'C_successor_calls': graph.calls, 'actions': graph.actions, 'domain': domain, 'outcomes': outcomes, 'values': values, 'near_tie_count': recovered['near_tie_count'], 'mode_scores': recovered.get('mode_scores', {}), 'solver': {k: v for k, v in solved.items() if k not in ('values', 'layers')}, 'failure_status': 'OK' if solved['status'] != 'SWEEP_BUDGET_EXHAUSTED' else solved['status']}
    response['result_signature'] = digest({k: response[k] for k in ('config', 'method', 'outcomes', 'values', 'solver', 'mode_scores')})
    if memory:
        gs = sum((row.itemsize * len(row) for row in graph.rows.values()))
        rs = sum((row.itemsize * len(row) for row in ref.values()))
        response['memory'] = {'graph_payload_bytes': gs, 'reference_payload_bytes': rs, 'graph_row_records': nedge, 'reference_entries': sum((len(r) for r in ref.values())), 'value_numeric_entries': solved['numeric_value_entries'], 'value_numeric_equivalent_bytes': 8 * solved['numeric_value_entries'], 'value_live_allocation_bytes': deep_size(solved['values']), 'logical_policy_entries': recovered['logical_policy_entries'], 'stored_policy_entries': recovered['stored_policy_entries'], 'policy_numeric_equivalent_bytes': 16 * recovered['stored_policy_entries'], 'graph_allocation_bytes': deep_size((graph.rows, graph.states, graph.actions, graph.rho10, graph.targets)), 'reference_allocation_bytes': deep_size(ref), 'policy_allocation_bytes': deep_size(recovered), 'retained_controller_allocation_bytes': deep_size((graph.rows, graph.states, graph.actions, graph.rho10, graph.targets, ref, solved, recovered)), 'measurement_qualification': 'Payload arrays are allocated bytes; other numeric-equivalent fields exclude Python overhead. RSS measured separately.'}
    return response
