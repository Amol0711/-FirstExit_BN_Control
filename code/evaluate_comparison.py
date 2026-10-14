#!/usr/bin/env python3
"""Evaluate numerical policies, high-precision references and recorded comparison trials. Diagnostic timings are separate from controller measurements."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns
import mpmath as mp
from comparison_solvers import TERMINAL, SCALE, METHODS, get_case, configurations, build_graph, basin, reverse_dijkstra, solve_dp, recover, verify_rollouts

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def qstats(values):
    vals = sorted(values)
    q = statistics.quantiles(vals, n=4, method='inclusive') if len(vals) > 1 else [vals[0]] * 3
    return {'median': statistics.median(vals), 'q1': q[0], 'q3': q[2]}

def independent_raw(case):
    """Enumerate raw transitions by integer XOR, independently of the flip helper used by the compared solvers."""
    actions = tuple(sorted(case.actions))
    targets = {int(t, 2) for t in case.target}
    states = [x for x in range(2 ** case.n) if x not in targets]
    costs = [int(sum((Fraction(str(case.weights.get(i, 0))) for i in a), Fraction()) * 10) for a in actions]
    rows = {}
    grouped = {}
    for x in states:
        raw = []
        by_y = {}
        for j, a in enumerate(actions):
            flipped = x
            for node in a:
                flipped ^= 1 << case.n - node
            actual = int(case.update(format(flipped, f'0{case.n}b')), 2)
            y = -1 if actual in targets else actual
            raw.append((y, j, costs[j]))
            if y not in by_y or (costs[j], a) < (by_y[y][1], actions[by_y[y][0]]):
                by_y[y] = (j, costs[j])
        rows[x] = raw
        grouped[x] = by_y
    dist = {x: None for x in states}
    dist[-1] = 0
    for _ in range(len(states)):
        nxt = {-1: 0}
        for x in states:
            vals = [10 + c + dist[y] for y, a, c in rows[x] if dist[y] is not None]
            nxt[x] = min(vals) if vals else None
        if nxt == dist:
            break
        dist = nxt
    return (rows, grouped, dist, actions, costs)

def high_precision_kl(grouped, domain):
    """Compute a separate 80-decimal-digit numerical reference; this is not an interval certificate."""
    with mp.workdps(80):
        beta = mp.exp(-1)
        one = mp.mpf(1)
        zero = mp.mpf(0)
        refs = {}
        for x in domain:
            raw = [(y, mp.exp(-mp.mpf(c) / 10)) for y, (a, c) in grouped[x].items()]
            gamma = mp.fsum((w for y, w in raw))
            refs[x] = [(y, w / gamma) for y, w in raw]

        def apply(z):
            result = {-1: one}
            for x in domain:
                result[x] = beta * mp.fsum((p * z.get(y, zero) for y, p in refs[x]))
            return result
        z = {x: one for x in domain}
        z[-1] = one
        residual = mp.inf
        for sweeps in range(1, 10001):
            z = apply(z)
            check = apply(z)
            residual = max([abs(z[x] - check[x]) for x in domain] or [zero])
            if residual / (1 - beta) <= mp.mpf('1e-60'):
                break
        else:
            raise RuntimeError('High-precision KL audit exhausted its sweep budget')
        return ({x: mp.nstr(v, 82) for x, v in z.items()}, mp.nstr(residual / (1 - beta), 82), sweeps)

def audit_one(config, blocks):
    start = perf_counter_ns()
    case = get_case(config)
    t = perf_counter_ns()
    raw, groups, values, actions, action_costs = independent_raw(case)
    oracle_ns = perf_counter_ns() - t
    domain = [x for x in raw if values[x] is not None]
    t = perf_counter_ns()
    cg = build_graph(case, True)
    b = basin(cg)
    full = solve_dp(cg, b, early_stop=False)
    full_policy = recover(cg, b, 'TW-DP-H', full)
    full_rollouts = verify_rollouts(cg, b, full_policy)
    fullH_ns = perf_counter_ns() - t
    if any((full['values'][x] != values[x] for x in domain)):
        raise AssertionError('Full-H DP disagrees with independent original-action oracle')
    if any((o['cost10'] != values[o['initial']] for o in full_rollouts)):
        raise AssertionError('Full-H policy does not realize its optimal costs')
    t = perf_counter_ns()
    z, err, hpsweeps = high_precision_kl(groups, domain)
    hp_ns = perf_counter_ns() - t
    quality = []
    case_summaries = []
    audit_modes = []
    for method in METHODS:
        record = blocks[config['id'], method]['first_result']
        assert set(record['domain']) == set(domain)
        proper = optimal = firstopt = succopt = 0
        gaps = []
        hits = []
        flips = []
        all_costs = []
        for outcome in record['outcomes']:
            x = outcome['initial']
            y = outcome['first_successor']
            a = outcome['first_action']
            d = values[x]
            cost = outcome['cost10']
            isproper = outcome['status'] == 'PROPER'
            gap = (cost - d) / 10 if isproper else math.inf
            first_ok = a is not None and values[y] is not None and (10 + action_costs[a] + values[y] == d)
            succ_ok = y is not None and values[y] is not None and (10 + groups[x][y][1] + values[y] == d)
            isoptimal = isproper and cost == d
            proper += int(isproper)
            optimal += int(isoptimal)
            firstopt += int(first_ok)
            succopt += int(succ_ok)
            gaps.append(gap)
            all_costs.append(cost / 10 if isproper else math.inf)
            if isproper:
                hits.append(outcome['hitting_time'])
                flips.append(outcome['nonempty_interventions'])
            row = {'config': config['id'], 'model': config['model'], 'method': method, 'state_bits': outcome['initial_bits'], 'delta_index': outcome['delta_index'], 'proper': int(isproper), 'Jstar': d / 10, 'Jpolicy': cost / 10 if isproper else 'inf', 'cost_gap': gap, 'optimal_first_action': int(first_ok), 'optimal_successor': int(succ_ok), 'optimal_complete_policy': int(isoptimal), 'hitting_time': outcome['hitting_time'], 'nonempty_interventions': outcome['nonempty_interventions'], 'flipped_nodes': outcome['flipped_nodes'], 'action': str(actions[a]) if a is not None else '', 'successor': y}
            quality.append(row)
            if method in ('U-SP', 'C-SP', 'TW-DP-H'):
                if not isoptimal or record['values'][str(x)] != d:
                    raise AssertionError(f"{method} failed exact original-cost check at {config['id']}:{x}")
        summary = {'config': config['id'], 'model': config['model'], 'method': method, 'N_B': len(domain), 'proper_states': proper, 'optimal_first_actions': firstopt, 'optimal_successors': succopt, 'optimal_complete_policies': optimal, 'proper_fraction': proper / len(domain), 'optimal_complete_fraction': optimal / len(domain), 'mean_original_cost': statistics.mean(all_costs), 'mean_cost_gap': statistics.mean(gaps), 'max_cost_gap': max(gaps), 'mean_hitting_time_proper_only': statistics.mean(hits) if hits else '', 'max_hitting_time_proper_only': max(hits) if hits else '', 'mean_interventions_proper_only': statistics.mean(flips) if flips else '', 'proper_only_denominator': proper, 'near_tie_count': record['near_tie_count'], 'solver_sweeps': record['solver']['sweeps'], 'solver_edge_visits': record['solver']['edge_visits'], 'fullH_edge_visits': full['edge_visits'] if method == 'TW-DP-H' else 0}
        case_summaries.append(summary)
    primary = blocks[config['id'], 'C-KL-MODE']['first_result']
    with mp.workdps(80):
        zm = {int(k): mp.mpf(v) for k, v in z.items()}
        maxerr = mp.mpf(0)
        maxslack = mp.mpf(0)
        for out in primary['outcomes']:
            x = out['initial']
            chosen = out['first_successor']
            opts = []
            maxerr = max(maxerr, abs(mp.mpf(primary['values'][str(x)]) - zm[x]))
            for y, (a, c) in groups[x].items():
                if y in zm:
                    opts.append((mp.mpf(c) / 10 - mp.log(zm[y]), y))
            best = min((v for v, y in opts))
            slack = next((v - best for v, y in opts if y == chosen))
            maxslack = max(maxslack, slack)
            tied = [y for v, y in opts if abs(v - best) <= mp.mpf('1e-50')]
            status = 'NUMERICAL_TIE_UNRESOLVED_EXACT_EQUALITY' if len(tied) > 1 else 'RESOLVED_UNIQUE_NUMERICAL_MODE'
            if slack > mp.mpf('1e-50'):
                status = 'PRIMARY_MODE_DISAGREEMENT'
            audit_modes.append({'config': config['id'], 'state_bits': out['initial_bits'], 'selected_successor': chosen, 'high_precision_score_slack': mp.nstr(slack, 40), 'audit_status': status, 'audit_near_optimal_count': len(tied)})
    audit_ns = perf_counter_ns() - start
    t = perf_counter_ns()
    rg = build_graph(case, False)
    rb = basin(rg)
    optimal = reverse_dijkstra(rg, rb)['values']
    dg = build_graph(case, True)
    refsolve_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    diagnostics = []
    jkl = {int(k): -math.log(v) for k, v in primary['values'].items() if v > 0}
    for x in rb:
        row = dg.rows[x]
        continuations = []
        for j in range(0, len(row), 3):
            y = row[j]
            if y in optimal and y in jkl:
                continuations.append((row[j + 2] + optimal[y], y))
        best = min((c for c, y in continuations))
        worse = [c - best for c, y in continuations if c > best]
        delta = min(worse) / 10 if worse else math.inf
        eps = max((abs(jkl[y] - optimal[y] / 10) for c, y in continuations))
        band = 1e-10 * max(1.0, abs(delta), 2 * eps) if math.isfinite(delta) else 0.0
        cls = 'PASS_INFINITE' if math.isinf(delta) else 'UNRESOLVED_BOUNDARY' if abs(delta - 2 * eps) <= band else 'PASS' if delta > 2 * eps else 'NOT_PASSED'
        selected = next((o for o in primary['outcomes'] if o['initial'] == x))
        a = selected['first_action']
        y = selected['first_successor']
        loss = (10 + action_costs[a] + optimal[y] - optimal[x]) / 10
        diagnostics.append({'config': config['id'], 'model': config['model'], 'state_bits': f'{x:0{case.n}b}', 'gap': delta, 'mismatch': eps, 'gap_test_class': cls, 'local_loss': loss})
    scan_ns = perf_counter_ns() - t
    audit = {'config': config['id'], 'oracle_ns': oracle_ns, 'fullH_validation_ns': fullH_ns, 'high_precision_ns': hp_ns, 'independent_audit_ns': audit_ns, 'gap_reference_rebuild_and_solve_ns': refsolve_ns, 'gap_scan_ns': scan_ns, 'gap_ns': refsolve_ns + scan_ns, 'high_precision_sweeps': hpsweeps, 'high_precision_error_estimate': err, 'max_primary_desirability_error': mp.nstr(maxerr, 40), 'max_audited_mode_score_slack': mp.nstr(maxslack, 40), 'resolved_disagreements': sum((r['audit_status'] == 'PRIMARY_MODE_DISAGREEMENT' for r in audit_modes)), 'numerical_ties_unresolved_exact_equality': sum((r['audit_status'].startswith('NUMERICAL_TIE') for r in audit_modes)), 'fullH_values_and_policy_checked': True, 'fullH_sweeps': full['sweeps'], 'N0': len(raw), 'N_B': len(domain), 'A': len(actions), 'E_all_distinct': sum((len(row) for row in groups.values())), 'E_B_distinct': sum((len(groups[x]) for x in domain))}
    return (quality, case_summaries, audit_modes, diagnostics, audit, z)

def main(run: Path):
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise RuntimeError('Primary campaign is not complete')
    out = run / 'evaluation'
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing audit')
    out.mkdir()
    (out / 'high_precision').mkdir()
    blocks = {}
    memory = {}
    timing = []
    for phase, store in [('timing', blocks), ('memory', memory)]:
        for f in sorted((run / phase).glob('*.json')):
            r = json.loads(f.read_text())
            if r['status'] != 'OK':
                raise RuntimeError(f'Failed campaign block {f}')
            key = (r['config']['id'], r['method'])
            store[key] = r
            if phase == 'timing':
                for t in r['trials']:
                    timing.append({'config': key[0], 'method': key[1], 'phase': t['phase'], 'repetition': t['index'], **t['phases'], 'trial_wall_ns': t['trial_wall_ns'], 'status': t['status'], 'result_signature': t['result_signature']})
    if len(blocks) != 532 or len(memory) != 532:
        raise AssertionError('Incomplete block coverage')
    for k in blocks:
        if blocks[k]['first_result']['result_signature'] != memory[k]['first_result']['result_signature']:
            raise AssertionError(f'Memory/timing numerical outcome differs {k}')
    allq = []
    alls = []
    allm = []
    alld = []
    alla = []
    for index, config in enumerate(configurations(manifest['settings'])):
        q, s, m, d, a, z = audit_one(config, blocks)
        allq += q
        alls += s
        allm += m
        alld += d
        alla.append(a)
        (out / 'high_precision' / f"{config['id']}.json").write_text(json.dumps({'config': config, 'z': z, 'audit': a}, indent=2) + '\n')
        if index % 20 == 0 or index == 132:
            print('audited', index + 1, '/133', flush=True)
    summ = []
    audits = {a['config']: a for a in alla}
    quality_summary = {(s['config'], s['method']): s for s in alls}
    phases = ['model_validation_ns', 'graph_build_ns', 'basin_ns', 'reference_ns', 'solve_ns', 'recovery_ns', 'properness_ns', 'total_no_diagnostic_ns']
    for key, b in blocks.items():
        r = b['first_result']
        a = audits[key[0]]
        mem = memory[key]['first_result']['memory']
        row = {**quality_summary[key], 'N_total': r['N_total'], 'N0': r['N0'], 'A': r['A'], 'E_all_stored': r['E_all'], 'E_B_stored': r['E_B'], 'E_all_distinct': a['E_all_distinct'], 'E_B_distinct': a['E_B_distinct'], 'successor_calls_per_trial': r['C_successor_calls']}
        trials = [t for t in b['trials'] if t['phase'] == 'measured']
        if len(trials) != 10:
            raise AssertionError('Repetition count differs')
        for phase in phases:
            for label, value in qstats([t['phases'][phase] for t in trials]).items():
                row[f'{phase}_{label}'] = value
        row.update({k: v for k, v in mem.items() if k != 'measurement_qualification'})
        row['single_case_independent_audit_ns'] = a['independent_audit_ns']
        row['single_case_optional_gap_ns'] = a['gap_ns'] if key[1] == 'C-KL-MODE' else 0
        row['composite_total_with_gap_ns'] = row['total_no_diagnostic_ns_median'] + row['single_case_optional_gap_ns']
        row['composite_total_with_audit_and_gap_ns'] = row['composite_total_with_gap_ns'] + a['independent_audit_ns']
        summ.append(row)
    write_csv(out / 'timing_trials.csv', timing)
    write_csv(out / 'per_state_quality.csv', allq)
    write_csv(out / 'per_configuration_summary.csv', summ)
    write_csv(out / 'reference_checks.csv', alla)
    write_csv(out / 'mode_precision_checks.csv', allm)
    write_csv(out / 'gap_diagnostics.csv', alld)
    aggregate = {}
    for method in METHODS:
        q = [r for r in allq if r['method'] == method]
        aggregate[method] = {'states': len(q), 'proper': sum((r['proper'] for r in q)), 'optimal_first_action': sum((r['optimal_first_action'] for r in q)), 'optimal_complete_policy': sum((r['optimal_complete_policy'] for r in q)), 'max_cost_gap': max((r['cost_gap'] for r in q)), 'mean_cost_gap': statistics.mean((r['cost_gap'] for r in q))}
    closure = {'status': 'PASS', 'configurations': 133, 'methods': 4, 'timing_blocks': 532, 'memory_blocks': 532, 'warmup_trials': 1596, 'measured_trials': 5320, 'per_state_rows': len(allq), 'per_configuration_rows': len(summ), 'quality_aggregate': aggregate, 'high_precision_mode_disagreements': sum((a['resolved_disagreements'] for a in alla)), 'unresolved_exact_mode_ties': sum((a['numerical_ties_unresolved_exact_equality'] for a in alla)), 'R1_R2_gap_counts': dict(Counter((r['gap_test_class'] for r in alld if r['model'] in ('R1', 'R2')))), 'max_primary_desirability_error': max((float(a['max_primary_desirability_error']) for a in alla)), 'timing_qualification': 'Sub-ms times are resolution-sensitive. Audits/gap replays measured once per configuration after primary workers; composite totals are sums, not repeated audit measurements.', 'memory_qualification': 'Fresh-process RSS includes interpreter and workspace. Baseline-subtracted high-water differences can be zero despite allocations. Payload and deep-size columns are distinct.'}
    if closure['high_precision_mode_disagreements']:
        closure['status'] = 'PASS_WITH_PRIMARY_MODE_DISAGREEMENTS_REPORTED'
    (out / 'evaluation_summary.json').write_text(json.dumps(closure, indent=2, allow_nan=False) + '\n')
    print(json.dumps(closure, indent=2), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    main(a.run.resolve())
