#!/usr/bin/env python3
"""Boolean-network models, successor compression, numerical solvers and data exporters. Node numbers are one-based; delta index 1 denotes the all-one state."""
from __future__ import annotations
import csv
import heapq
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import exp, log
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple
State = str
Action = Tuple[int, ...]
DAGGER = 'dagger'

def all_states(n: int) -> List[State]:
    if n <= 0:
        raise ValueError('n must be positive')
    return [''.join((str(b) for b in bits)) for bits in product([0, 1], repeat=n)]

def powerset(nodes: Sequence[int]) -> List[Action]:
    nodes = tuple(sorted(nodes))
    return [tuple((nodes[i] for i in range(len(nodes)) if mask >> i & 1)) for mask in range(1 << len(nodes))]

def actions_up_to(nodes: Sequence[int], max_size: int) -> List[Action]:
    if max_size < 0:
        raise ValueError('max_size must be nonnegative')
    nodes = tuple(sorted(nodes))
    out: List[Action] = [tuple()]
    for r in range(1, min(max_size, len(nodes)) + 1):
        out.extend((tuple(c) for c in combinations(nodes, r)))
    return out

def flip_state(s: State, action: Action) -> State:
    if any((c not in '01' for c in s)):
        raise ValueError(f'Expected bit-string state, got {s!r}')
    bits = [int(c) for c in s]
    for node in action:
        if node < 1 or node > len(bits):
            raise ValueError(f'Node {node} is out of range for state {s!r}')
        bits[node - 1] = 1 - bits[node - 1]
    return ''.join((str(b) for b in bits))

def delta_index_to_bits(index: int, n: int) -> State:
    if index < 1 or index > 2 ** n:
        raise ValueError(f'delta index {index} is out of range for n={n}')
    value = index - 1
    bits = []
    for k in range(n):
        encoded_bit = value >> n - 1 - k & 1
        bits.append(1 - encoded_bit)
    return ''.join((str(b) for b in bits))

def bits_to_delta_index(s: State) -> int:
    if any((c not in '01' for c in s)):
        raise ValueError(f'Expected bit-string state, got {s!r}')
    value = 0
    for c in s:
        value = (value << 1) + (1 - int(c))
    return value + 1

def state_sort_key(s: State, n: int) -> Tuple[int, int | str]:
    if s == DAGGER:
        return (1, 2 ** n + 1)
    return (0, bits_to_delta_index(s))

def action_to_text(action: Action) -> str:
    if not action:
        return 'empty'
    return '{' + ','.join((str(i) for i in action)) + '}'

def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None=None) -> None:
    """Write numerical rows with a deterministic tab-separated column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', newline='') as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

@dataclass(frozen=True)
class BNCase:
    name: str
    n: int
    flip_nodes: Tuple[int, ...]
    target: Tuple[State, ...]
    update: Callable[[State], State]
    weights: Dict[int, float]
    actions_override: Tuple[Action, ...] | None = None

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError('n must be positive')
        for t in self.target:
            if len(t) != self.n or any((c not in '01' for c in t)):
                raise ValueError(f'Invalid target state {t!r} for n={self.n}')
        for node in self.flip_nodes:
            if node < 1 or node > self.n:
                raise ValueError(f'flip node {node} is out of range for n={self.n}')
        if self.actions_override is not None:
            for action in self.actions_override:
                for node in action:
                    if node < 1 or node > self.n:
                        raise ValueError(f'action {action} has node outside 1..{self.n}')

    @property
    def actions(self) -> List[Action]:
        if self.actions_override is not None:
            return [tuple(a) for a in self.actions_override]
        return powerset(self.flip_nodes)

    @property
    def states(self) -> List[State]:
        return all_states(self.n)

    @property
    def target_set(self) -> set[State]:
        return set(self.target)

    @property
    def transient_states(self) -> List[State]:
        return [s for s in self.states if s not in self.target_set]

    def rho(self, action: Action) -> float:
        return sum((self.weights.get(node, 0.0) for node in action))

    def terminalized_successor(self, s: State, action: Action) -> State:
        y = self.update(flip_state(s, action))
        return DAGGER if y in self.target_set else y

    def with_actions(self, actions: Sequence[Action], name: str | None=None) -> 'BNCase':
        nodes = tuple(sorted({node for action in actions for node in action}))
        return BNCase(name=self.name if name is None else name, n=self.n, flip_nodes=nodes, target=self.target, update=self.update, weights=dict(self.weights), actions_override=tuple((tuple(a) for a in actions)))

    def with_weights(self, weights: Mapping[int, float], name: str | None=None) -> 'BNCase':
        return BNCase(name=self.name if name is None else name, n=self.n, flip_nodes=self.flip_nodes, target=self.target, update=self.update, weights=dict(weights), actions_override=self.actions_override)

@dataclass(frozen=True)
class SuccessorModel:
    case: BNCase
    ysets: Dict[State, List[State]]
    classes: Dict[Tuple[State, State], List[Action]]
    reps: Dict[Tuple[State, State], Action]
    rhostar: Dict[Tuple[State, State], float]

    @property
    def N_transient(self) -> int:
        return len(self.case.transient_states)

    @property
    def A(self) -> int:
        return len(self.case.actions)

    @property
    def E_0(self) -> int:
        return sum((len(row) for row in self.ysets.values()))

    @property
    def N_B(self) -> int:
        return len(graph_controllable_basin(self))

    @property
    def E_B(self) -> int:
        return sum((len(self.ysets[x]) for x in graph_controllable_basin(self)))

    @property
    def construction_edge_count(self) -> int:
        return self.N_transient * self.A

    @property
    def raw_edge_count(self) -> int:
        return self.N_B * self.A

    @property
    def compression_factor(self) -> float:
        return math.nan if self.E_B == 0 else self.raw_edge_count / self.E_B

    def successors(self, s: State) -> List[State]:
        return self.ysets[s]

    def edge_cost(self, s: State, y: State) -> float:
        return 1.0 + self.rhostar[s, y]

def build_successor_model(case: BNCase, *, retain_classes: bool=True) -> SuccessorModel:
    """Enumerate successor classes; choose representatives by exact decimal cost, cardinality, then lexicographic node order. Optional class retention controls redundant action storage."""
    actions = case.actions
    if not actions:
        raise ValueError('An admissible action family must be nonempty')
    costs = {a: rational_action_cost(case, a) for a in actions}
    ysets: Dict[State, List[State]] = {}
    classes: Dict[Tuple[State, State], List[Action]] = {}
    reps: Dict[Tuple[State, State], Action] = {}
    rhostar: Dict[Tuple[State, State], float] = {}
    for x in case.transient_states:
        seen: set[State] = set()
        for action in actions:
            y = case.terminalized_successor(x, action)
            key = (x, y)
            seen.add(y)
            if retain_classes:
                classes.setdefault(key, []).append(action)
            if key not in reps or (costs[action], len(action), action) < (costs[reps[key]], len(reps[key]), reps[key]):
                reps[key] = action
        ysets[x] = sorted(seen)
        for y in ysets[x]:
            key = (x, y)
            if retain_classes:
                classes[key].sort(key=lambda a: (costs[a], len(a), a))
            rhostar[key] = case.rho(reps[key])
    return SuccessorModel(case, ysets, classes, reps, rhostar)

def graph_controllable_basin(model: SuccessorModel) -> set[State]:
    """Find the transient basin by reverse reachability, without cost optimization."""
    from collections import deque
    reverse: Dict[State, List[State]] = {DAGGER: []}
    for x, ys in model.ysets.items():
        for y in ys:
            reverse.setdefault(y, []).append(x)
    reached = {DAGGER}
    queue = deque([DAGGER])
    while queue:
        y = queue.popleft()
        for x in reverse.get(y, ()):
            if x not in reached:
                reached.add(x)
                queue.append(x)
    return reached - {DAGGER}

def successor_data(case: BNCase):
    model = build_successor_model(case)
    return (model.ysets, model.classes, model.reps, model.rhostar)

def _dijkstra_to_target(states: Sequence[State], edge_iter: Callable[[], Iterator[Tuple[State, State, float]]], n: int) -> Dict[State, float]:
    dist: Dict[State, float] = {s: math.inf for s in states}
    dist[DAGGER] = 0.0
    predecessors: Dict[State, List[Tuple[State, float]]] = {DAGGER: []}
    for s in states:
        predecessors.setdefault(s, [])
    for s, y, cost in edge_iter():
        predecessors.setdefault(y, []).append((s, cost))
    heap: List[Tuple[float, Tuple[int, int | str], State]] = [(0.0, (1, 0), DAGGER)]
    settled: set[State] = set()
    while heap:
        d, _, y = heapq.heappop(heap)
        if y in settled:
            continue
        settled.add(y)
        if d > dist[y] + 1e-15:
            continue
        for s, cost in predecessors.get(y, []):
            cand = d + cost
            if cand + 1e-12 < dist[s]:
                dist[s] = cand
                heapq.heappush(heap, (cand, state_sort_key(s, n), s))
    return dist

def deterministic_values(case: BNCase, tol: float=1e-12, max_iter: int=10000):
    model = build_successor_model(case)

    def edges() -> Iterator[Tuple[State, State, float]]:
        for s in case.transient_states:
            for y in model.ysets[s]:
                yield (s, y, model.edge_cost(s, y))
    values = _dijkstra_to_target(case.transient_states, edges, case.n)
    for t in case.target:
        values[t] = 0.0
    values[DAGGER] = 0.0
    return values

def raw_deterministic_values(case: BNCase) -> Dict[State, float]:
    actions = case.actions

    def edges() -> Iterator[Tuple[State, State, float]]:
        for s in case.transient_states:
            for action in actions:
                yield (s, case.terminalized_successor(s, action), 1.0 + case.rho(action))
    values = _dijkstra_to_target(case.transient_states, edges, case.n)
    for t in case.target:
        values[t] = 0.0
    values[DAGGER] = 0.0
    return values

def controllable_basin(case: BNCase, values: Mapping[State, float] | None=None) -> set[State]:
    return graph_controllable_basin(build_successor_model(case, retain_classes=False))

def backward_controllable_basin(case: BNCase) -> set[State]:
    return controllable_basin(case)

def optimal_successors(case: BNCase, values: Mapping[State, float] | None=None, model: SuccessorModel | None=None, tol: float=1e-09) -> Dict[State, List[State]]:
    model = build_successor_model(case) if model is None else model
    values = deterministic_values(case) if values is None else values
    out: Dict[State, List[State]] = {}
    for s in case.transient_states:
        scores = [(model.rhostar[s, y] + values[y], y) for y in model.ysets[s]]
        best = min((v for v, _ in scores))
        out[s] = [y for v, y in scores if abs(v - best) <= tol]
    return out

def deterministic_rollout(case: BNCase, initial: State, max_steps: int=100, values: Dict[State, float] | None=None, successor_cache: Tuple[Dict[State, List[State]], Dict[Tuple[State, State], Action]] | None=None) -> Tuple[List[Dict[str, object]], int]:
    if successor_cache is None:
        model = build_successor_model(case)
        ysets, reps = (model.ysets, model.reps)
    else:
        ysets, reps = successor_cache
    if values is None:
        values = deterministic_values(case)
    if not math.isfinite(values.get(initial, math.inf)):
        raise RuntimeError(f'Initial state {initial!r} is not in the controllable basin')
    rows: List[Dict[str, object]] = []
    flips = 0
    s = initial
    for t in range(max_steps + 1):
        if s in case.target_set:
            rows.append({'t': t, 'state_bits': s, 'state_delta': bits_to_delta_index(s), 'action': 'target', 'successor_bits': 'target', 'successor_delta': bits_to_delta_index(s)})
            return (rows, flips)
        if t == max_steps:
            raise RuntimeError(f'Rollout from {initial!r} exhausted its step budget')

        def sort_key(y: State) -> Tuple[float, Tuple[int, int | str], str]:
            action = reps[s, y]
            return (1.0 + case.rho(action) + values[y], state_sort_key(y, case.n), action_to_text(action))
        y_best = min(ysets[s], key=sort_key)
        a_best = reps[s, y_best]
        if a_best:
            flips += 1
        rows.append({'t': t, 'state_bits': s, 'state_delta': bits_to_delta_index(s), 'action': action_to_text(a_best), 'successor_bits': DAGGER if y_best == DAGGER else y_best, 'successor_delta': DAGGER if y_best == DAGGER else bits_to_delta_index(y_best)})
        if y_best == DAGGER:
            target = case.update(flip_state(s, a_best))
            if target not in case.target_set:
                raise ValueError('Terminal representative does not enter the target')
            rows.append({'t': t + 1, 'state_bits': target, 'state_delta': bits_to_delta_index(target), 'action': 'target', 'successor_bits': 'target', 'successor_delta': bits_to_delta_index(target)})
            return (rows, flips)
        s = y_best
    raise RuntimeError(f'Rollout from {initial} did not reach target within {max_steps} steps')

def _solve_linear_system_dense(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    if n == 0:
        return []
    aug = [list(A[i]) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-15:
            raise ValueError('Singular linear system in LMDP solve')
        if pivot != col:
            aug[col], aug[pivot] = (aug[pivot], aug[col])
        piv = aug[col][col]
        inv = 1.0 / piv
        for j in range(col, n + 1):
            aug[col][j] *= inv
        for r in range(n):
            if r == col:
                continue
            fac = aug[r][col]
            if fac == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= fac * aug[col][j]
    return [aug[i][n] for i in range(n)]

def _finite_successors(case: BNCase, state: State, values_det: Mapping[State, float], values_kl: Mapping[State, float], model: SuccessorModel) -> List[State]:
    if state not in model.ysets:
        raise ValueError(f'State {state!r} is not a transient state of case {case.name}')
    return [y for y in model.ysets[state] if math.isfinite(values_det.get(y, math.inf)) and math.isfinite(values_kl.get(y, math.inf))]

def map_recovered_successor(case: BNCase, state: State, values_kl: Mapping[State, float], model: SuccessorModel | None=None, values_det: Mapping[State, float] | None=None, *, basin: set[State] | None=None) -> Tuple[State, Action]:
    model = build_successor_model(case) if model is None else model
    basin = graph_controllable_basin(model) if basin is None else basin
    if state not in basin:
        raise ValueError(f'State {state!r} is outside the target-reachable basin')
    y_f = [y for y in model.ysets[state] if y == DAGGER or y in basin]
    if any((not math.isfinite(values_kl.get(y, math.inf)) for y in y_f)):
        raise ValueError('Unresolved numerical KL cost on a graph-reachable successor')
    if not y_f:
        raise ValueError(f'State {state!r} has no finite-value MAP successor in case {case.name}')

    def sort_key(y: State) -> Tuple[float, Tuple[int, int | str], str]:
        action = model.reps[state, y]
        return (model.rhostar[state, y] + values_kl[y], state_sort_key(y, case.n), action_to_text(action))
    y_hat = min(y_f, key=sort_key)
    return (y_hat, model.reps[state, y_hat])

def map_recovered_policy(case: BNCase, values_kl: Mapping[State, float] | None=None, model: SuccessorModel | None=None, states: Sequence[State] | None=None, values_det: Mapping[State, float] | None=None) -> Dict[State, Tuple[State, Action]]:
    model = build_successor_model(case) if model is None else model
    basin = graph_controllable_basin(model)
    if values_kl is None:
        _, values_kl = kl_values(case)
    if states is None:
        states = sorted(basin, key=bits_to_delta_index)
    return {s: map_recovered_successor(case, s, values_kl, model, values_det, basin=basin) for s in states}

def map_recovered_rollout(case: BNCase, initial: State, max_steps: int=100, values_kl: Mapping[State, float] | None=None, model: SuccessorModel | None=None, values_det: Mapping[State, float] | None=None) -> Tuple[List[Dict[str, object]], int]:
    model = build_successor_model(case) if model is None else model
    basin = graph_controllable_basin(model)
    if initial not in case.target_set and initial not in basin:
        raise ValueError(f'Initial state {initial!r} is outside the finite-value basin')
    if values_kl is None:
        _, values_kl = kl_values(case)
    rows: List[Dict[str, object]] = []
    flips = 0
    s = initial
    seen: set[State] = set()
    for t in range(max_steps + 1):
        if s in case.target_set:
            rows.append({'t': t, 'state_bits': s, 'state_delta': bits_to_delta_index(s), 'action': 'target', 'successor_bits': 'target', 'successor_delta': bits_to_delta_index(s)})
            return (rows, flips)
        if s in seen:
            raise RuntimeError(f'MAP-recovered rollout from {initial!r} entered a cycle at {s!r}')
        if t == max_steps:
            raise RuntimeError(f'MAP-recovered rollout from {initial!r} exhausted its step budget')
        seen.add(s)
        y_hat, a_hat = map_recovered_successor(case, s, values_kl, model, values_det, basin=basin)
        if a_hat:
            flips += 1
        rows.append({'t': t, 'state_bits': s, 'state_delta': bits_to_delta_index(s), 'action': action_to_text(a_hat), 'successor_bits': DAGGER if y_hat == DAGGER else y_hat, 'successor_delta': DAGGER if y_hat == DAGGER else bits_to_delta_index(y_hat)})
        if y_hat == DAGGER:
            target = case.update(flip_state(s, a_hat))
            if target not in case.target_set:
                raise ValueError('Terminal representative does not enter the target')
            rows.append({'t': t + 1, 'state_bits': target, 'state_delta': bits_to_delta_index(target), 'action': 'target', 'successor_bits': 'target', 'successor_delta': bits_to_delta_index(target)})
            return (rows, flips)
        s = y_hat
    raise RuntimeError(f'MAP-recovered rollout from {initial!r} did not reach target within {max_steps} steps')

def kl_values(case: BNCase, epsilon: float=0.0, tol: float=1e-14, max_iter: int=100000, *, backend: str='auto'):
    """Return full-state arrays through the numerical solver interface. Basin-row probabilities retain weights to zero-boundary endpoints; an unsuccessful residual check raises SolverNotConverged."""
    from numerical_solvers import make_sparse_kernel, solve_desirability
    kernel = make_sparse_kernel(case, epsilon)
    result = solve_desirability(kernel, backend=backend, tolerance=tol, max_sweeps=max_iter)
    costs = result.costs(kernel)
    z = {s: 0.0 for s in case.states}
    z.update(result.z)
    z.update({t: 1.0 for t in case.target})
    jkl = {s: math.inf for s in case.states}
    jkl.update(costs)
    jkl.update({t: 0.0 for t in case.target})
    return (z, jkl)
ExactValue = Fraction | float

def rational_action_cost(case: BNCase, action: Action) -> Fraction:
    value = Fraction(0)
    for node in action:
        weight = case.weights.get(node, 0)
        try:
            term = weight if isinstance(weight, Fraction) else Fraction(str(weight))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f'Non-finite or invalid weight for node {node}: {weight!r}') from exc
        if term < 0:
            raise ValueError(f'Negative intervention weight at node {node}')
        value += term
    return value

def rational_deterministic_values(case: BNCase) -> Dict[State, ExactValue]:
    dist: Dict[State, ExactValue] = {s: math.inf for s in case.states}
    dist.update({s: Fraction(0) for s in case.target})
    dist[DAGGER] = Fraction(0)
    edges = {x: [(case.terminalized_successor(x, a), Fraction(1) + rational_action_cost(case, a)) for a in case.actions] for x in case.transient_states}
    for _ in case.transient_states:
        nxt = dict(dist)
        for x, row in edges.items():
            candidates = [cost + dist[y] for y, cost in row if isinstance(dist[y], Fraction)]
            if candidates:
                nxt[x] = min(candidates)
        if nxt == dist:
            return nxt
        dist = nxt
    return dist

def strict_successor_gap(costs: Mapping[State, Fraction]) -> ExactValue:
    if not costs:
        raise ValueError('A successor-gap check requires a nonempty finite set')
    if any((not isinstance(v, Fraction) for v in costs.values())):
        raise TypeError('Exact continuation costs must be Fraction values')
    best = min(costs.values())
    worse = [v for v in costs.values() if v > best]
    return min(worse) - best if worse else math.inf

def numerical_gap_test(gap: ExactValue, mismatch: float, tolerance: float=1e-09) -> bool:
    if not math.isfinite(mismatch) or mismatch < 0:
        raise ValueError('Mismatch must be finite and nonnegative')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Comparison tolerance must be finite and nonnegative')
    if gap == math.inf:
        return True
    if gap < 0 or not math.isfinite(gap):
        raise ValueError('Gap must be nonnegative or positive infinity')
    return float(gap) > 2.0 * mismatch + tolerance

@dataclass(frozen=True)
class RecoveryDiagnostic:
    case: str
    state: State
    state_delta: int
    deterministic_successor: State
    map_successor: State
    representative_action: Action
    Delta: float
    epsilon: float
    two_epsilon: float
    local_loss: float
    exact_transfer_condition_met: bool
    map_is_deterministic_optimal: bool
    compression_factor_state: float

    def as_row(self) -> Dict[str, object]:
        return {'case': self.case, 'state': self.state, 'state_delta': self.state_delta, 'y_D_star': self.deterministic_successor, 'y_hat': self.map_successor, 'V_hat': action_to_text(self.representative_action), 'Delta': f'{self.Delta:.12g}' if math.isfinite(self.Delta) else 'inf', 'epsilon': f'{self.epsilon:.12g}', 'two_epsilon': f'{self.two_epsilon:.12g}', 'local_loss': f'{self.local_loss:.12g}', 'exact_transfer_condition_met': str(self.exact_transfer_condition_met), 'map_is_deterministic_optimal': str(self.map_is_deterministic_optimal), 'compression_factor_state': f'{self.compression_factor_state:.12g}'}

def recovery_evaluation(case: BNCase, state: State, values_det: Mapping[State, float] | None=None, values_kl: Mapping[State, float] | None=None, model: SuccessorModel | None=None, tol: float=1e-09, *, values_det_exact: Mapping[State, ExactValue] | None=None) -> RecoveryDiagnostic:
    model = build_successor_model(case) if model is None else model
    values_det_exact = rational_deterministic_values(case) if values_det_exact is None else values_det_exact
    if values_kl is None:
        _, values_kl = kl_values(case)
    if state not in model.ysets:
        raise ValueError(f'State {state!r} is not transient in {case.name}')
    y_f = [y for y in model.ysets[state] if isinstance(values_det_exact.get(y), Fraction)]
    if not y_f:
        raise ValueError(f'State {state!r} has no finite-value successors in case {case.name}')
    if any((not math.isfinite(values_kl.get(y, math.inf)) for y in y_f)):
        raise ValueError('KL solve has a non-finite value on the reachable successor domain')
    if values_det is not None and any((not math.isfinite(values_det.get(y, math.inf)) for y in y_f)):
        raise ValueError('Supplied deterministic domain disagrees with the exact raw-action oracle')
    phi: Dict[State, Fraction] = {}
    for y in y_f:
        cost = min((rational_action_cost(case, a) for a in model.classes[state, y]))
        if rational_action_cost(case, model.reps[state, y]) != cost:
            raise ValueError('Floating-point representative is not an exact decimal-cost minimizer')
        phi[y] = cost + values_det_exact[y]
    best_phi = min(phi.values())
    y_D = min((y for y in y_f if phi[y] == best_phi), key=lambda y: state_sort_key(y, case.n))
    y_hat, action = map_recovered_successor(case, state, values_kl, model, values_det_exact)
    gap = strict_successor_gap(phi)
    mismatch = max((abs(values_kl[y] - float(values_det_exact[y])) for y in y_f))
    exact_loss = phi[y_hat] - best_phi
    return RecoveryDiagnostic(case=case.name, state=state, state_delta=bits_to_delta_index(state), deterministic_successor=y_D, map_successor=y_hat, representative_action=action, Delta=float(gap), epsilon=mismatch, two_epsilon=2.0 * mismatch, local_loss=float(exact_loss), exact_transfer_condition_met=numerical_gap_test(gap, mismatch, tol), map_is_deterministic_optimal=exact_loss == 0, compression_factor_state=len(case.actions) / len(model.ysets[state]))

def recovery_evaluations(case: BNCase, states: Sequence[State] | None=None) -> List[RecoveryDiagnostic]:
    model = build_successor_model(case)
    values_det = deterministic_values(case)
    exact = rational_deterministic_values(case)
    _, values_kl = kl_values(case)
    if states is None:
        states = sorted((s for s in case.transient_states if isinstance(exact[s], Fraction)), key=bits_to_delta_index)
    return [recovery_evaluation(case, s, values_det, values_kl, model, values_det_exact=exact) for s in states]

def exactness_summary_row(case: BNCase) -> Dict[str, object]:
    model = build_successor_model(case)
    values_comp = deterministic_values(case)
    values_raw = raw_deterministic_values(case)
    basin = controllable_basin(case, values_comp)
    max_err = 0.0
    for s in case.transient_states:
        a = values_comp[s]
        b = values_raw[s]
        if math.isfinite(a) or math.isfinite(b):
            max_err = max(max_err, abs(a - b))
    evaluations = recovery_evaluations(case)
    exact_map = sum((1 for d in evaluations if d.map_is_deterministic_optimal))
    condition_met = sum((1 for d in evaluations if d.exact_transfer_condition_met))
    n_eval = len(evaluations)
    return {'case': case.name, 'N_B': len(basin), 'N_transient': len(case.transient_states), 'A': model.A, 'E_B': model.E_B, 'NB_A_over_EB': f'{model.raw_edge_count / model.E_B:.6g}' if model.E_B else 'undefined', 'max_value_error': f'{max_err:.3g}', 'MAP_exact_states': f'{exact_map}/{n_eval}', 'gap_condition_met_states': f'{condition_met}/{n_eval}'}

def cumulative_cost_rows(case: BNCase, max_sweeps: int=50) -> List[Dict[str, object]]:
    model = build_successor_model(case)
    rows: List[Dict[str, object]] = []
    raw_once = model.raw_edge_count
    comp_once = model.E_B
    for k in range(1, max_sweeps + 1):
        rows.append({'case': case.name, 'K': k, 'raw_total': k * raw_once, 'compressed_total': raw_once + k * comp_once, 'raw_sweep': raw_once, 'compressed_sweep': comp_once})
    return rows

def storage_summary_row(case: BNCase) -> Dict[str, object]:
    model = build_successor_model(case)
    return {'case': case.name, 'N_B': len(controllable_basin(case)), 'N_transient': model.N_transient, 'A': model.A, 'E_B': model.E_B, 'raw_storage': model.raw_edge_count, 'model_storage': model.N_transient + model.E_B, 'desirability_storage': model.N_transient, 'raw_sweep': model.raw_edge_count, 'compressed_sweep': model.E_B, 'compression_factor': f'{model.compression_factor:.12g}'}

def make_cases():

    def v1_update(s: State) -> State:
        if s == '110':
            return '110'
        x1, x2, x3 = map(int, s)
        return f'{1 - x2}{x1 & 1 - x3}{x2}'

    def v2_update(s: State) -> State:
        return '0' + s[2] + '0'

    def v3_update(s: State) -> State:
        x1, x2, x3 = map(int, s)
        return f'{x1}{1 - x2}{x1 & x3}'

    def v4_update(s: State) -> State:
        x1, x2, x3 = map(int, s)
        return f'0{x1}{x1 | x3}'

    def r1_update(s: State) -> State:
        x1, x2 = map(int, s)
        return f'{x1 & x2}0'

    def r2_update(s: State) -> State:
        x1, x2, x3 = map(int, s)
        return f'0{x1}{x2 | x3}'
    return {'V1': BNCase('V1', 3, (1, 2), ('110',), v1_update, {}), 'V2': BNCase('V2', 3, (1, 2), ('000',), v2_update, {}), 'V3': BNCase('V3', 3, (1, 2), ('100', '110'), v3_update, {}), 'V4': BNCase('V4', 3, (1, 2), ('001',), v4_update, {}), 'R1': BNCase('R1', 2, (1, 2), ('00',), r1_update, {1: 0.4, 2: 1.0}), 'R2': BNCase('R2', 3, (1, 2), ('001',), r2_update, {1: 0.4, 2: 1.0})}
PHENOTYPE9_NODE_NAMES = ('MBF', 'SBF', 'YOX1', 'HCM1', 'YHP1', 'SFF', 'ACE2', 'SWI5', 'CLN3')

def phenotype9_update(s: State) -> State:
    """Apply the nine-node Boolean rule in the node order declared by PHENOTYPE9_NODE_NAMES."""
    if len(s) != 9 or any((c not in '01' for c in s)):
        raise ValueError(f'Expected a 9-bit Boolean state, got {s!r}')
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = map(int, s)
    eta = 1 - (x3 & x5)
    nxt = (x9, (x9 | x1) & eta, x1 & x2, x1 & x2, x1 | x2, x2 & x4, x6, x6, x8 & x7 & eta)
    return ''.join((str(b) for b in nxt))

def phenotype9_case(flip_nodes: Sequence[int]=(9,), weights: Dict[int, float] | None=None, max_flip_size: int | None=None) -> BNCase:
    """Create a nine-node instance targeting the all-zero state. max_flip_size restricts cardinality; omission uses the full powerset of eligible nodes."""
    actions = None
    if max_flip_size is not None:
        actions = tuple(actions_up_to(flip_nodes, max_flip_size))
    return BNCase(name='P9' if max_flip_size is None else f'P9-r{max_flip_size}', n=9, flip_nodes=tuple(flip_nodes), target=(delta_index_to_bits(512, 9),), update=phenotype9_update, weights={} if weights is None else dict(weights), actions_override=actions)

def uncontrolled_attractors(update: Callable[[State], State], n: int) -> List[Tuple[State, ...]]:
    seen_global: set[State] = set()
    attractors: List[Tuple[State, ...]] = []
    for s0 in all_states(n):
        if s0 in seen_global:
            continue
        path: List[State] = []
        pos: Dict[State, int] = {}
        s = s0
        while s not in pos and s not in seen_global:
            pos[s] = len(path)
            path.append(s)
            s = update(s)
        if s in pos:
            attractors.append(tuple(path[pos[s]:]))
        seen_global.update(path)
    return attractors

def normalize_cycle_indices(cycle: Sequence[State]) -> Tuple[int, ...]:
    ids = tuple((bits_to_delta_index(s) for s in cycle))
    if len(ids) <= 1:
        return ids
    rotations = [ids[i:] + ids[:i] for i in range(len(ids))]
    return min(rotations)

def attractor_reached(update: Callable[[State], State], s0: State) -> Tuple[State, ...]:
    path: List[State] = []
    pos: Dict[State, int] = {}
    s = s0
    while s not in pos:
        pos[s] = len(path)
        path.append(s)
        s = update(s)
    return tuple(path[pos[s]:])

def hitting_time_to_set(update: Callable[[State], State], s0: State, target: set[State], max_steps: int=1000) -> int | None:
    s = s0
    seen: set[State] = set()
    for t in range(max_steps + 1):
        if s in target:
            return t
        if s in seen:
            return None
        seen.add(s)
        s = update(s)
    return None
