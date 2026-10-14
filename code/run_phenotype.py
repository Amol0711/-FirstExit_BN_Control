#!/usr/bin/env python3
"""Generate nine-node Boolean-network basin, intervention, policy and trajectory data. Node positions and state indices follow MODEL_NOTES.txt."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence
from boolean_networks import PHENOTYPE9_NODE_NAMES, all_states, attractor_reached, bits_to_delta_index, build_successor_model, controllable_basin, deterministic_values, hitting_time_to_set, kl_values, map_recovered_rollout, normalize_cycle_indices, phenotype9_case, phenotype9_update, recovery_evaluation, rational_deterministic_values, write_tsv
UNIT_INTERVENTION_WEIGHT = 1.0
MAX_ROLLOUT_STEPS = 20
MAX_TRAJECTORY_DISPLAY_STEPS = 10
ENSEMBLE_TIME_HORIZON = 4
ENSEMBLE_ACTION_HORIZON = 3

def copy_rows(rows: Sequence[Mapping[str, object]], destinations: Iterable[Path], fieldnames: Sequence[str] | None=None) -> None:
    for path in destinations:
        write_tsv(path, rows, fieldnames=fieldnames)

def action_nodes(action_text: str) -> list[int]:
    if action_text in {'empty', 'target', 'none'}:
        return []
    cleaned = action_text.strip().strip('{}')
    if not cleaned:
        return []
    return [int(part) for part in cleaned.split(',')]

def node_label(node: int | str) -> str:
    if node == 'none':
        return 'none'
    return PHENOTYPE9_NODE_NAMES[int(node) - 1]

def first_node_code(label: object) -> int:
    if str(label) == 'none':
        return 10
    return int(label)

def atlas_sort_key(row: Mapping[str, object]) -> tuple[int, int, int, int, int]:
    fate_rank = 0 if str(row['uncontrolled_fate']) == 'target' else 1
    return (fate_rank, int(row['controlled_tau']), int(row['N_flip']), first_node_code(row['first_nonempty_node']), int(row['state_delta']))

def action_label_from_nodes(nodes: Sequence[int]) -> str:
    if not nodes:
        return 'empty'
    if len(nodes) == 1:
        return node_label(nodes[0])
    return '+'.join((node_label(node) for node in nodes))

def state_to_gene_rows(*, case_id: str, mode: str, time: int, state_bits: str) -> list[dict[str, object]]:
    return [{'case_id': case_id, 'mode': mode, 'time': time, 'gene_idx': gene_idx, 'gene_name': gene_name, 'value': int(state_bits[gene_idx - 1])} for gene_idx, gene_name in enumerate(PHENOTYPE9_NODE_NAMES, start=1)]

def controlled_action_rows(*, case_id: str, rollout_rows: Sequence[Mapping[str, object]], display_horizon: int | None=None) -> list[dict[str, object]]:
    tau = int(rollout_rows[-1]['t'])
    horizon = tau if display_horizon is None else max(tau, display_horizon)
    rows: list[dict[str, object]] = []
    cumulative = 0
    action_by_time = {int(row['t']): str(row['action']) for row in rollout_rows[:-1]}
    for time in range(horizon + 1):
        if time < tau:
            action_text = action_by_time.get(time, 'empty')
            nodes = action_nodes(action_text)
            if nodes:
                cumulative += len(nodes)
            action_label = action_label_from_nodes(nodes)
            action_code = nodes[0] if len(nodes) == 1 else 0 if not nodes else -1
            hit_after = 'True' if time == tau - 1 else 'False'
        elif time == tau:
            action_text = 'target'
            nodes = []
            action_label = 'target'
            action_code = 10
            hit_after = 'False'
        else:
            action_text = 'target_hold'
            nodes = []
            action_label = 'target'
            action_code = 10
            hit_after = 'False'
        rows.append({'case_id': case_id, 'time': time, 'action': action_text, 'action_label': action_label, 'flipped_node_idx': action_code, 'flipped_node_name': action_label if action_code > 0 else action_label, 'nonempty': str(bool(nodes)), 'cumulative_flips': cumulative, 'target_hit_after_action': hit_after})
    return rows

def controlled_trajectory_rows(*, case_id: str, rollout_rows: Sequence[Mapping[str, object]], display_horizon: int | None=None) -> list[dict[str, object]]:
    tau = int(rollout_rows[-1]['t'])
    horizon = tau if display_horizon is None else max(tau, display_horizon)
    state_by_time = {int(row['t']): str(row['state_bits']) for row in rollout_rows}
    target_bits = str(rollout_rows[-1]['state_bits'])
    rows: list[dict[str, object]] = []
    for time in range(horizon + 1):
        bits = state_by_time.get(time, target_bits)
        rows.extend(state_to_gene_rows(case_id=case_id, mode='controlled', time=time, state_bits=bits))
    return rows

def uncontrolled_rollout_rows(*, case_id: str, initial: str, target_set: set[str], max_steps: int=MAX_TRAJECTORY_DISPLAY_STEPS) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    visited: dict[str, int] = {}
    states: list[str] = []
    s = initial
    stop_reason = 'max_steps'
    repeat_time = 'none'
    for time in range(max_steps + 1):
        states.append(s)
        if s in target_set:
            stop_reason = 'target'
            break
        if s in visited:
            stop_reason = 'cycle_repeat'
            repeat_time = visited[s]
            break
        visited[s] = time
        s = phenotype9_update(s)
    traj_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for time, bits in enumerate(states):
        traj_rows.extend(state_to_gene_rows(case_id=case_id, mode='uncontrolled', time=time, state_bits=bits))
        action_rows.append({'case_id': case_id, 'time': time, 'action': 'empty' if time < len(states) - 1 else stop_reason, 'action_label': 'empty' if time < len(states) - 1 else stop_reason, 'flipped_node_idx': 0, 'flipped_node_name': 'none', 'nonempty': 'False', 'cumulative_flips': 0, 'target_hit_after_action': 'False'})
    meta = {'uncontrolled_stop_reason': stop_reason, 'uncontrolled_repeat_time': repeat_time, 'uncontrolled_display_horizon': len(states) - 1}
    return (traj_rows, action_rows, meta)

def ensemble_gene_activity_rows(*, initial_states: Sequence[str], controlled_rollouts_by_state: Mapping[str, Sequence[Mapping[str, object]]], horizon: int, target_bits: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in ('uncontrolled', 'recovered'):
        for time in range(horizon + 1):
            gene_totals = [0] * len(PHENOTYPE9_NODE_NAMES)
            for initial in initial_states:
                if mode == 'uncontrolled':
                    s = initial
                    for _ in range(time):
                        s = phenotype9_update(s)
                    bits = s
                else:
                    rollout = controlled_rollouts_by_state[initial]
                    state_by_time = {int(row['t']): str(row['state_bits']) for row in rollout}
                    bits = state_by_time.get(time, target_bits)
                for idx, bit in enumerate(bits):
                    gene_totals[idx] += int(bit)
            cohort_size = len(initial_states)
            for gene_idx, gene_name in enumerate(PHENOTYPE9_NODE_NAMES, start=1):
                frac = gene_totals[gene_idx - 1] / cohort_size
                rows.append({'cohort': 'cycle_basin', 'mode': mode, 'time': time, 'gene_idx': gene_idx, 'gene_name': gene_name, 'active_fraction': f'{frac:.8f}'})
    return rows

def intervention_time_rows(*, initial_states: Sequence[str], controlled_rollouts_by_state: Mapping[str, Sequence[Mapping[str, object]]], horizon: int) -> list[dict[str, object]]:
    action_counter: dict[tuple[int, str], int] = defaultdict(int)
    row_order = ['empty', *PHENOTYPE9_NODE_NAMES]
    for initial in initial_states:
        rollout = controlled_rollouts_by_state[initial]
        action_by_time = {int(row['t']): str(row['action']) for row in rollout[:-1]}
        tau = int(rollout[-1]['t'])
        for time in range(horizon + 1):
            label = 'empty'
            if time < tau:
                nodes = action_nodes(action_by_time.get(time, 'empty'))
                label = 'empty' if not nodes else node_label(nodes[0])
            action_counter[time, label] += 1
    cohort_size = len(initial_states)
    rows: list[dict[str, object]] = []
    for row_idx, label in enumerate(row_order, start=1):
        for time in range(horizon + 1):
            count = action_counter.get((time, label), 0)
            rows.append({'time': time, 'row_idx': row_idx, 'node_name': label, 'count': count, 'fraction': f'{count / cohort_size:.8f}'})
    return rows

def choose_trajectory_cases(state_rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    cycle_rows = [row for row in state_rows if str(row['uncontrolled_fate']) == 'cycle']
    if not cycle_rows:
        raise RuntimeError('No cycle-basin states available for trajectory cases')
    tau_values = sorted((int(row['controlled_tau']) for row in state_rows))
    flip_values = sorted((int(row['N_flip']) for row in state_rows))
    median_tau = tau_values[len(tau_values) // 2]
    median_flips = flip_values[len(flip_values) // 2]
    representative = min(cycle_rows, key=lambda row: (abs(int(row['controlled_tau']) - median_tau), abs(int(row['N_flip']) - median_flips), int(row['state_delta'])))
    harder = min(cycle_rows, key=lambda row: (-int(row['N_flip']), -int(row['controlled_tau']), int(row['state_delta'])))
    return (representative, harder)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output_destination = args.output
    weights = {i: UNIT_INTERVENTION_WEIGHT for i in range(1, 10)}
    case = phenotype9_case(flip_nodes=tuple(range(1, 10)), weights=weights, max_flip_size=1)
    model = build_successor_model(case)
    values_det = deterministic_values(case)
    _, values_kl = kl_values(case)
    basin = controllable_basin(case, values_det)
    if len(basin) != len(case.transient_states):
        raise RuntimeError(f'Expected all non-target states to be controllable; got {len(basin)}')
    target_cycle = (512,)
    basin_counts: dict[tuple[int, ...], int] = {}
    for s in all_states(9):
        att = normalize_cycle_indices(attractor_reached(phenotype9_update, s))
        basin_counts[att] = basin_counts.get(att, 0) + 1
    cycles = sorted(basin_counts, key=lambda c: (len(c), c))
    fixed = [c for c in cycles if len(c) == 1]
    nontrivial = [c for c in cycles if len(c) > 1]
    if fixed != [target_cycle]:
        raise RuntimeError(f'Unexpected fixed attractors: {fixed}')
    if len(nontrivial) != 1:
        raise RuntimeError(f'Expected one competing cycle; got {nontrivial}')
    competing_cycle = nontrivial[0]
    attractor_rows = [{'fate_idx': 1, 'fate': 'target', 'fate_label': 'target fixed point', 'count': basin_counts[target_cycle]}, {'fate_idx': 2, 'fate': 'cycle', 'fate_label': 'competing cycle', 'count': basin_counts[competing_cycle]}]
    target_set = set(case.target)
    uncontrolled_tau = Counter()
    uncontrolled_fate: dict[str, str] = {}
    for s in case.transient_states:
        ht = hitting_time_to_set(phenotype9_update, s, target_set)
        if ht is None:
            uncontrolled_tau['inf'] += 1
        else:
            uncontrolled_tau[int(ht)] += 1
        att = normalize_cycle_indices(attractor_reached(phenotype9_update, s))
        uncontrolled_fate[s] = 'target' if att == target_cycle else 'cycle'
    controlled_tau = Counter()
    flip_count_hist = Counter()
    first_node_hist = Counter()
    total_node_hist = Counter()
    by_uncontrolled_fate_tau: dict[str, Counter] = defaultdict(Counter)
    state_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    controlled_rollouts_by_state: dict[str, list[dict[str, object]]] = {}
    controlled_flip_count_by_state: dict[str, int] = {}
    tau_values: list[int] = []
    flip_values: list[int] = []
    max_local_loss = 0.0
    map_exact = 0
    gap_condition_met = 0
    values_det_exact = rational_deterministic_values(case)
    for s in sorted(case.transient_states, key=bits_to_delta_index):
        rows, n_flips = map_recovered_rollout(case, s, values_kl=values_kl, model=model, values_det=values_det, max_steps=MAX_ROLLOUT_STEPS)
        controlled_rollouts_by_state[s] = rows
        controlled_flip_count_by_state[s] = n_flips
        tau = int(rows[-1]['t'])
        if int(rows[-1]['state_delta']) != 512:
            raise RuntimeError(f'Recovered policy from delta512^{bits_to_delta_index(s)} did not reach target')
        tau_values.append(tau)
        flip_values.append(n_flips)
        controlled_tau[tau] += 1
        flip_count_hist[n_flips] += 1
        by_uncontrolled_fate_tau[uncontrolled_fate[s]][tau] += 1
        first_node = 'none'
        node_sequence: list[int] = []
        for row in rows[:-1]:
            nodes = action_nodes(str(row['action']))
            if nodes and first_node == 'none':
                first_node = str(nodes[0]) if len(nodes) == 1 else '+'.join((str(v) for v in nodes))
            node_sequence.extend(nodes)
        first_node_hist[first_node] += 1
        for node in node_sequence:
            total_node_hist[node] += 1
        diag = recovery_evaluation(case, s, values_det, values_kl, model, values_det_exact=values_det_exact)
        max_local_loss = max(max_local_loss, diag.local_loss)
        if diag.map_is_deterministic_optimal:
            map_exact += 1
        if diag.exact_transfer_condition_met:
            gap_condition_met += 1
        ht = hitting_time_to_set(phenotype9_update, s, target_set)
        node_order = list(node_sequence[:3])
        while len(node_order) < 3:
            node_order.append('none')
        state_rows.append({'state_delta': bits_to_delta_index(s), 'state_bits': s, 'uncontrolled_fate': uncontrolled_fate[s], 'uncontrolled_tau': 'inf' if ht is None else ht, 'controlled_tau': tau, 'N_flip': n_flips, 'first_nonempty_node': first_node, 'first_node_name': node_label(first_node), 'second_nonempty_node': node_order[1], 'second_node_name': node_label(node_order[1]), 'third_nonempty_node': node_order[2], 'third_node_name': node_label(node_order[2]), 'intervention_sequence': ';'.join((str(v) for v in node_sequence)) if node_sequence else 'none', 'MAP_exact': str(diag.map_is_deterministic_optimal), 'gap_condition_met': str(diag.exact_transfer_condition_met), 'local_loss': f'{diag.local_loss:.12g}'})
        for row in rows:
            rollout_rows.append({'initial_delta': bits_to_delta_index(s), **row})
    if map_exact != len(case.transient_states):
        raise RuntimeError(f'Expected MAP exact on all finite-value P9-r1 states; got {map_exact}/{len(case.transient_states)}')
    outcome_rows = [{'outcome_idx': 1, 'outcome': 'target', 'uncontrolled_non_target': uncontrolled_tau[1] + uncontrolled_tau[2] + uncontrolled_tau[3] + uncontrolled_tau[4], 'controlled_non_target': len(case.transient_states)}, {'outcome_idx': 2, 'outcome': 'cycle', 'uncontrolled_non_target': uncontrolled_tau['inf'], 'controlled_non_target': 0}]
    hitting_bins = [(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, 'inf')]
    hitting_rows = []
    for idx, label in hitting_bins:
        key = 'inf' if idx == 5 else idx
        hitting_rows.append({'bin_idx': idx, 'tau_label': label, 'uncontrolled': uncontrolled_tau.get(key, 0), 'controlled': controlled_tau.get(key, 0)})
    intervention_rows = []
    for node in range(1, 10):
        intervention_rows.append({'node_idx': node, 'node_name': PHENOTYPE9_NODE_NAMES[node - 1], 'first_frequency': first_node_hist.get(str(node), 0) + first_node_hist.get(node, 0), 'total_frequency': total_node_hist.get(node, 0)})
    intervention_rows.append({'node_idx': 10, 'node_name': 'none', 'first_frequency': first_node_hist.get('none', 0), 'total_frequency': 0})
    flip_hist_rows = [{'N_flip': k, 'count': flip_count_hist[k]} for k in sorted(flip_count_hist)]
    fate_tau_rows = []
    for fate in ['target', 'cycle']:
        for tau in sorted(controlled_tau):
            fate_tau_rows.append({'uncontrolled_fate': fate, 'tau': tau, 'count': by_uncontrolled_fate_tau[fate].get(tau, 0)})
    sorted_state_rows = sorted(state_rows, key=atlas_sort_key)
    policy_atlas_rows: list[dict[str, object]] = []
    atlas_cell_rows: list[dict[str, object]] = []
    band_specs = [(1, 'uncontrolled fate', 'uncontrolled_fate_code', 'uncontrolled_fate'), (2, 'controlled tau', 'controlled_tau', 'controlled_tau'), (3, 'N_flip', 'N_flip', 'N_flip'), (4, 'first intervention', 'first_node_code', 'first_node_name')]
    for pos, row in enumerate(sorted_state_rows, start=1):
        fate_code = 1 if str(row['uncontrolled_fate']) == 'target' else 2
        fnode_code = first_node_code(row['first_nonempty_node'])
        atlas_row = {'sorted_position': pos, 'state_delta': row['state_delta'], 'state_bits': row['state_bits'], 'uncontrolled_fate': row['uncontrolled_fate'], 'uncontrolled_fate_code': fate_code, 'controlled_tau': row['controlled_tau'], 'N_flip': row['N_flip'], 'first_nonempty_node': row['first_nonempty_node'], 'first_node_code': fnode_code, 'first_node_name': row['first_node_name'], 'second_nonempty_node': row['second_nonempty_node'], 'second_node_name': row['second_node_name'], 'third_nonempty_node': row['third_nonempty_node'], 'third_node_name': row['third_node_name'], 'sort_fate_rank': 0 if str(row['uncontrolled_fate']) == 'target' else 1, 'sort_tau': row['controlled_tau'], 'sort_N_flip': row['N_flip'], 'sort_first_node_code': fnode_code}
        policy_atlas_rows.append(atlas_row)
        for band_idx, band_name, code_key, label_key in band_specs:
            atlas_cell_rows.append({'sorted_position': pos, 'band_idx': band_idx, 'band_name': band_name, 'state_delta': row['state_delta'], 'code': atlas_row[code_key], 'label': atlas_row[label_key]})
    bubble_counter: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for row in state_rows:
        bubble_counter[int(row['controlled_tau']), int(row['N_flip'])][str(row['uncontrolled_fate'])] += 1
    tau_flip_bubble_rows = []
    for (tau, n_flips), fate_counter in sorted(bubble_counter.items()):
        count = sum(fate_counter.values())
        tau_flip_bubble_rows.append({'tau': tau, 'N_flip': n_flips, 'count': count, 'target_fate_count': fate_counter.get('target', 0), 'cycle_fate_count': fate_counter.get('cycle', 0), 'marker_size_pt': f'{1.5 + 0.18 * count ** 0.5:.4g}', 'count_label': count})
    order_names = {1: 'first', 2: 'second', 3: 'third'}
    intervention_order_counter: dict[tuple[int, int], int] = defaultdict(int)
    for row in state_rows:
        for order_idx, key in enumerate(['first_nonempty_node', 'second_nonempty_node', 'third_nonempty_node'], start=1):
            node_value = row[key]
            if str(node_value) != 'none':
                intervention_order_counter[int(node_value), order_idx] += 1
    intervention_order_rows = []
    for node in range(1, 10):
        for order_idx in range(1, 4):
            count = intervention_order_counter.get((node, order_idx), 0)
            intervention_order_rows.append({'node_idx': node, 'node_name': PHENOTYPE9_NODE_NAMES[node - 1], 'order_idx': order_idx, 'order_name': order_names[order_idx], 'count': count})
    intervention_order_wide_rows = []
    for node in range(1, 10):
        intervention_order_wide_rows.append({'node_idx': node, 'node_name': PHENOTYPE9_NODE_NAMES[node - 1], 'first_count': intervention_order_counter.get((node, 1), 0), 'second_count': intervention_order_counter.get((node, 2), 0), 'third_count': intervention_order_counter.get((node, 3), 0)})
    if len(policy_atlas_rows) != len(case.transient_states):
        raise RuntimeError('Policy-summary row count mismatch')
    if sum((row['count'] for row in tau_flip_bubble_rows)) != len(case.transient_states):
        raise RuntimeError('Tau/flip joint counts do not sum to the transient state count')
    total_order_count = sum((row['count'] for row in intervention_order_rows))
    if total_order_count != sum(flip_values):
        raise RuntimeError('Intervention-order heatmap counts do not match total intervention count')
    representative_case, harder_case = choose_trajectory_cases(state_rows)
    trajectory_case_specs = [('case1', representative_case, 'representative rescued cycle-basin state: median tau/flip burden among cycle-basin states'), ('case2', harder_case, 'harder rescued cycle-basin state: maximal N_flip, then maximal tau_T')]
    trajectory_case_rows: list[dict[str, object]] = []
    trajectory_heatmap_rows: list[dict[str, object]] = []
    trajectory_action_rows: list[dict[str, object]] = []
    case1_uncontrolled_traj_rows: list[dict[str, object]] = []
    case1_uncontrolled_action_rows: list[dict[str, object]] = []
    case1_controlled_traj_rows: list[dict[str, object]] = []
    case1_controlled_action_rows: list[dict[str, object]] = []
    case2_controlled_traj_rows: list[dict[str, object]] = []
    case2_controlled_action_rows: list[dict[str, object]] = []
    for case_id, selected, reason in trajectory_case_specs:
        initial_bits = str(selected['state_bits'])
        controlled_rows = controlled_rollouts_by_state[initial_bits]
        selected_tau = int(selected['controlled_tau'])
        selected_n_flips = int(selected['N_flip'])
        uncontrolled_traj, uncontrolled_actions, uncontrolled_meta = uncontrolled_rollout_rows(case_id=case_id, initial=initial_bits, target_set=target_set, max_steps=MAX_TRAJECTORY_DISPLAY_STEPS)
        display_horizon = max(selected_tau, int(uncontrolled_meta['uncontrolled_display_horizon']) if case_id == 'case1' else selected_tau)
        controlled_traj = controlled_trajectory_rows(case_id=case_id, rollout_rows=controlled_rows, display_horizon=display_horizon)
        controlled_actions = controlled_action_rows(case_id=case_id, rollout_rows=controlled_rows, display_horizon=display_horizon)
        trajectory_case_rows.append({'case_id': case_id, 'state_delta': selected['state_delta'], 'state_bits': initial_bits, 'uncontrolled_fate': selected['uncontrolled_fate'], 'uncontrolled_tau': selected['uncontrolled_tau'], 'controlled_tau': selected_tau, 'N_flip': selected_n_flips, 'first_intervention': selected['first_node_name'], 'intervention_sequence': selected['intervention_sequence'], 'selection_reason': reason, 'display_horizon': display_horizon, **uncontrolled_meta})
        if case_id == 'case1':
            case1_uncontrolled_traj_rows = uncontrolled_traj
            case1_uncontrolled_action_rows = uncontrolled_actions
            case1_controlled_traj_rows = controlled_traj
            case1_controlled_action_rows = controlled_actions
            trajectory_heatmap_rows.extend(uncontrolled_traj)
            trajectory_action_rows.extend(uncontrolled_actions)
        else:
            case2_controlled_traj_rows = controlled_traj
            case2_controlled_action_rows = controlled_actions
        trajectory_heatmap_rows.extend(controlled_traj)
        trajectory_action_rows.extend(controlled_actions)
    if len(trajectory_case_rows) != 2:
        raise RuntimeError('Expected exactly two selected trajectory cases')
    if not case1_uncontrolled_traj_rows or not case1_controlled_traj_rows or (not case2_controlled_traj_rows):
        raise RuntimeError('Trajectory case data generation failed')
    if int(trajectory_case_rows[0]['controlled_tau']) > MAX_TRAJECTORY_DISPLAY_STEPS:
        raise RuntimeError('Representative controlled trajectory exceeds display horizon')
    if int(trajectory_case_rows[1]['controlled_tau']) > MAX_TRAJECTORY_DISPLAY_STEPS:
        raise RuntimeError('Harder controlled trajectory exceeds display horizon')
    cycle_basin_initials = [str(row['state_bits']) for row in state_rows if str(row['uncontrolled_fate']) == 'cycle']
    ensemble_gene_rows = ensemble_gene_activity_rows(initial_states=cycle_basin_initials, controlled_rollouts_by_state=controlled_rollouts_by_state, horizon=ENSEMBLE_TIME_HORIZON, target_bits=case.target[0])
    ensemble_gene_uncontrolled_rows = [row for row in ensemble_gene_rows if str(row['mode']) == 'uncontrolled']
    ensemble_gene_recovered_rows = [row for row in ensemble_gene_rows if str(row['mode']) == 'recovered']
    intervention_time_heatmap_rows = intervention_time_rows(initial_states=[str(row['state_bits']) for row in state_rows], controlled_rollouts_by_state=controlled_rollouts_by_state, horizon=ENSEMBLE_ACTION_HORIZON)
    if len(cycle_basin_initials) != uncontrolled_tau['inf']:
        raise RuntimeError('Cycle-basin cohort size mismatch for ensemble gene activity')
    mean_tau = sum(tau_values) / len(tau_values)
    mean_flips = sum(flip_values) / len(flip_values)
    summary_rows = [{'target': 'delta512^512', 'competing_cycle': '->'.join((f'delta512^{i}' for i in list(competing_cycle) + [competing_cycle[0]])), 'target_uncontrolled_basin_size': basin_counts[target_cycle], 'cycle_uncontrolled_basin_size': basin_counts[competing_cycle], 'controlled_non_target_states': len(case.transient_states), 'N_B': len(basin), 'A': len(case.actions), 'E_B': model.E_B, 'N_B_times_A': model.raw_edge_count, 'compression_factor': f'{model.compression_factor:.12g}', 'unit_intervention_weight': UNIT_INTERVENTION_WEIGHT, 'max_tau_T': max(tau_values), 'median_tau_T': median(tau_values), 'mean_tau_T': f'{mean_tau:.6g}', 'max_N_flip': max(flip_values), 'median_N_flip': median(flip_values), 'mean_N_flip': f'{mean_flips:.6g}', 'uncontrolled_not_hit_count': uncontrolled_tau['inf'], 'MAP_exact_states': f'{map_exact}/{len(case.transient_states)}', 'gap_condition_met_states': f'{gap_condition_met}/{len(case.transient_states)}', 'max_local_loss': f'{max_local_loss:.12g}'}]
    files = {'phenotype_attractors.tsv': (attractor_rows, ['fate_idx', 'fate', 'fate_label', 'count']), 'phenotype_outcome_counts.tsv': (outcome_rows, ['outcome_idx', 'outcome', 'uncontrolled_non_target', 'controlled_non_target']), 'phenotype_hitting_times.tsv': (hitting_rows, ['bin_idx', 'tau_label', 'uncontrolled', 'controlled']), 'phenotype_intervention_frequency.tsv': (intervention_rows, ['node_idx', 'node_name', 'first_frequency', 'total_frequency']), 'phenotype_flip_histogram.tsv': (flip_hist_rows, ['N_flip', 'count']), 'phenotype_fate_tau.tsv': (fate_tau_rows, ['uncontrolled_fate', 'tau', 'count']), 'phenotype_basin_summary.tsv': (summary_rows, list(summary_rows[0].keys())), 'phenotype_state_policy.tsv': (state_rows, list(state_rows[0].keys())), 'phenotype_policy_atlas.tsv': (policy_atlas_rows, list(policy_atlas_rows[0].keys())), 'phenotype_policy_atlas_cells.tsv': (atlas_cell_rows, list(atlas_cell_rows[0].keys())), 'phenotype_tau_flip_bubbles.tsv': (tau_flip_bubble_rows, list(tau_flip_bubble_rows[0].keys())), 'phenotype_intervention_order.tsv': (intervention_order_rows, list(intervention_order_rows[0].keys())), 'phenotype_intervention_order_wide.tsv': (intervention_order_wide_rows, list(intervention_order_wide_rows[0].keys())), 'phenotype_rollouts.tsv': (rollout_rows, list(rollout_rows[0].keys())), 'phenotype_trajectory_cases.tsv': (trajectory_case_rows, list(trajectory_case_rows[0].keys())), 'phenotype_trajectory_heatmap.tsv': (trajectory_heatmap_rows, list(trajectory_heatmap_rows[0].keys())), 'phenotype_trajectory_actions.tsv': (trajectory_action_rows, list(trajectory_action_rows[0].keys())), 'phenotype_traj_case1_uncontrolled.tsv': (case1_uncontrolled_traj_rows, list(case1_uncontrolled_traj_rows[0].keys())), 'phenotype_traj_case1_uncontrolled_actions.tsv': (case1_uncontrolled_action_rows, list(case1_uncontrolled_action_rows[0].keys())), 'phenotype_traj_case1_controlled.tsv': (case1_controlled_traj_rows, list(case1_controlled_traj_rows[0].keys())), 'phenotype_traj_case1_actions.tsv': (case1_controlled_action_rows, list(case1_controlled_action_rows[0].keys())), 'phenotype_traj_case2_controlled.tsv': (case2_controlled_traj_rows, list(case2_controlled_traj_rows[0].keys())), 'phenotype_traj_case2_actions.tsv': (case2_controlled_action_rows, list(case2_controlled_action_rows[0].keys())), 'phenotype_ensemble_gene_activity.tsv': (ensemble_gene_rows, list(ensemble_gene_rows[0].keys())), 'phenotype_ensemble_gene_uncontrolled.tsv': (ensemble_gene_uncontrolled_rows, list(ensemble_gene_uncontrolled_rows[0].keys())), 'phenotype_ensemble_gene_recovered.tsv': (ensemble_gene_recovered_rows, list(ensemble_gene_recovered_rows[0].keys())), 'phenotype_intervention_time_heatmap.tsv': (intervention_time_heatmap_rows, list(intervention_time_heatmap_rows[0].keys()))}
    figure_input_names = {'phenotype_outcome_counts.tsv', 'phenotype_hitting_times.tsv', 'phenotype_intervention_frequency.tsv', 'phenotype_policy_atlas.tsv', 'phenotype_policy_atlas_cells.tsv', 'phenotype_tau_flip_bubbles.tsv', 'phenotype_intervention_order.tsv', 'phenotype_intervention_order_wide.tsv', 'phenotype_trajectory_cases.tsv', 'phenotype_trajectory_heatmap.tsv', 'phenotype_trajectory_actions.tsv', 'phenotype_traj_case1_uncontrolled.tsv', 'phenotype_traj_case1_uncontrolled_actions.tsv', 'phenotype_traj_case1_controlled.tsv', 'phenotype_traj_case1_actions.tsv', 'phenotype_traj_case2_controlled.tsv', 'phenotype_traj_case2_actions.tsv', 'phenotype_ensemble_gene_activity.tsv', 'phenotype_ensemble_gene_uncontrolled.tsv', 'phenotype_ensemble_gene_recovered.tsv', 'phenotype_intervention_time_heatmap.tsv'}
    for filename, (rows, fieldnames) in files.items():
        copy_rows(rows, [output_destination / filename], fieldnames=fieldnames)
    report_lines = ['Basin-wide 9-node phenotype-steering evaluations', 'State convention: delta_512^1=111111111, delta_512^512=000000000.', 'Intervention family: F_s^{(1)}={empty,{1},...,{9}} with unit singleton weights.', f'Uncontrolled target basin size over all 512 states: {basin_counts[target_cycle]}', f'Uncontrolled competing-cycle basin size over all 512 states: {basin_counts[competing_cycle]}', 'Competing cycle: ' + ' -> '.join((f'delta512^{i}' for i in list(competing_cycle) + [competing_cycle[0]])), f'Controlled finite-value non-target states: {len(case.transient_states)}', f'N_B={len(basin)}, A={len(case.actions)}, E_B={model.E_B}, N_B*A={model.raw_edge_count}, compression={model.compression_factor:.6g}', f'Controlled tau histogram: {dict(sorted(controlled_tau.items()))}', f'Uncontrolled tau histogram: {dict(uncontrolled_tau)}', f'Flip-count histogram: {dict(sorted(flip_count_hist.items()))}', f'First nonempty intervention histogram: {dict(first_node_hist)}', f'Total intervention-node histogram: {dict(total_node_hist)}', f'Policy-policy-summary states: {len(policy_atlas_rows)}', f'Tau/flip joint classes: {len(tau_flip_bubble_rows)}', f'Intervention-order total count: {total_order_count}', 'Trajectory cases: ' + '; '.join((f"{row['case_id']}=delta512^{row['state_delta']} tau={row['controlled_tau']} flips={row['N_flip']} reason={row['selection_reason']}" for row in trajectory_case_rows)), f'Ensemble trajectory cohort size (cycle basin): {len(cycle_basin_initials)}', f'Intervention-time heatmap horizon: 0..{ENSEMBLE_ACTION_HORIZON}', f'MAP deterministic-optimal states: {map_exact}/{len(case.transient_states)}', f'Gap-condition states: {gap_condition_met}/{len(case.transient_states)}', f'Max local suboptimality: {max_local_loss:.12g}']
    (output_destination / 'phenotype_basin_report.txt').write_text('\n'.join(report_lines) + '\n')
    print(f'Phenotype basin data generated: target basin {basin_counts[target_cycle]}, cycle basin {basin_counts[competing_cycle]}, controlled finite-value {len(case.transient_states)}/{len(case.transient_states)} states, max tau {max(tau_values)}, max flips {max(flip_values)}, policy-summary states {len(policy_atlas_rows)}, joint classes {len(tau_flip_bubble_rows)}, trajectory cases {len(trajectory_case_rows)}, MAP exact {map_exact}/{len(case.transient_states)}')
if __name__ == '__main__':
    main()
