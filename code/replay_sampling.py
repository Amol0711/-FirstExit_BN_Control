#!/usr/bin/env python3
"""Independently replay complete finite sampling histories from integer Boolean updates. The sampler and reference solver under test are not imported. High-precision checks require mpmath."""
from __future__ import annotations
import argparse
import ast
import bisect
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
import mpmath as mp
T = 'dagger'
checks = Counter()

def check(test, group):
    checks[group] += 1
    if not test:
        raise AssertionError(group)

def model(name):
    n = 2 if name == 'R1' else 3
    target = 0 if name == 'R1' else 1

    def update(i):
        b = [i >> k & 1 for k in range(n - 1, -1, -1)]
        return (b[0] & b[1]) * 2 if n == 2 else 2 * b[0] + (b[1] | b[2])
    states = [i for i in range(1 << n) if i != target]
    rows = {}
    for i in states:
        reps = {}
        for a in [(), (1,), (2,), (1, 2)]:
            j = i
            for k in a:
                j ^= 1 << n - k
            real = update(j)
            y = T if real == target else real
            cost = sum(({1: 4, 2: 10}[k] for k in a))
            candidate = (cost, len(a), a, real)
            if y not in reps or candidate[:3] < reps[y][:3]:
                reps[y] = candidate
        ys = sorted(reps, key=lambda y: 2 ** n if y == T else y)
        w = [math.exp(-reps[y][0] / 10) for y in ys]
        den = sum(w)
        rows[i] = [(y, v / den, reps[y][0], reps[y][2], reps[y][3]) for y, v in zip(ys, w)]
    return (n, target, states, rows, update)

def high_precision(states, rows, dps):
    with mp.workdps(dps):
        idx = {x: i for i, x in enumerate(states)}
        a = mp.eye(len(states))
        b = mp.zeros(len(states), 1)
        d = mp.exp(-1)
        for x in states:
            weights = [mp.exp(-mp.mpf(e[2]) / 10) for e in rows[x]]
            den = mp.fsum(weights)
            for e, w in zip(rows[x], weights):
                if e[0] == T:
                    b[idx[x]] += d * w / den
                else:
                    a[idx[x], idx[e[0]]] -= d * w / den
        z = mp.lu_solve(a, b)
        return {x: mp.nstr(z[idx[x]], dps) for x in states}

def audit(output: Path, settings: dict):
    global checks
    beta = math.exp(-1)
    total_updates = 0
    total_checkpoints = 0
    max_reference_delta = 0.0
    max_hp_delta = mp.mpf('0')
    summaries = list(csv.DictReader((output / 'sampling_summary.csv').open()))
    saved_summary = {(r['model'], int(r['seed'])): r for r in summaries}
    expected = {(name, s) for name in settings['models'] for s in settings['seeds']}
    check(set(saved_summary) == expected, 'complete_runs')
    for name in settings['models']:
        n, target, states, rows, update = model(name)
        bits = lambda x: T if x == T else format(x, f'0{n}b')
        ref = json.loads((output / f'{name}_reference.json').read_text())
        hp80 = high_precision(states, rows, 80)
        hp140 = high_precision(states, rows, 140)
        with mp.workdps(150):
            delta = max((abs(mp.mpf(hp80[x]) - mp.mpf(hp140[x])) for x in states))
            max_hp_delta = max(max_hp_delta, delta)
            check(delta < mp.mpf('1e-78'), 'high_precision_agreement')
        hpf = {x: float(hp140[x]) for x in states}
        for x in states:
            for e, actual in zip(rows[x], ref['rows'][bits(x)]):
                check(actual['successor'] == bits(e[0]) and actual['action'] == list(e[3]) and (actual['physical_successor'] == bits(e[4])), 'independent_row_structure')
                check(actual['probability'] == e[1], 'independent_row_weights')
            err = abs(ref['z'][bits(x)] - hpf[x])
            max_reference_delta = max(max_reference_delta, err)
            check(err < 1e-15, 'dense_reference_accuracy')
        for seed in settings['seeds']:
            final = json.loads((output / f'{name}_{seed}_result.json').read_text())
            history = [json.loads(s) for s in (output / f'{name}_{seed}_history.jsonl').read_text().splitlines()]
            z = {x: 1.0 for x in states}
            v = dict.fromkeys(states, 0)
            starts = dict(v)
            cumulative = {x: list(itertools.accumulate((e[1] for e in rows[x]))) for x in states}
            rng = random.Random(seed)
            w = dict.fromkeys(['sample_updates', 'episodes', 'sampler_edge_visits', 'residual_scans', 'residual_edge_visits', 'target_events', 'zero_boundary_events', 'cap_events', 'interrupted_episodes'], 0)
            pos = 0
            status = 'BUDGET_EXHAUSTED'
            cap = len(states)
            limit = settings['max_updates']
            scan_every = settings['scan_every']
            while w['sample_updates'] < limit:
                x = states[rng.randrange(len(states))]
                starts[x] += 1
                w['episodes'] += 1
                for j in range(cap):
                    ind = min(bisect.bisect_right(cumulative[x], rng.random()), len(rows[x]) - 1)
                    y = rows[x][ind][0]
                    v[x] += 1
                    alpha = v[x] ** (-settings['omega'])
                    previous = z[x]
                    continuation = 1.0 if y == T else z[y]
                    z[x] = (1.0 - alpha) * previous + alpha * beta * continuation
                    w['sample_updates'] += 1
                    w['sampler_edge_visits'] += ind + 1
                    ended = y == T or j + 1 == cap
                    if y == T:
                        w['target_events'] += 1
                    elif j + 1 == cap:
                        w['cap_events'] += 1
                    check(0 <= z[x] <= 1, 'sampled_value_bounds')
                    stop = False
                    if w['sample_updates'] % scan_every == 0 or w['sample_updates'] == limit:
                        r = max((abs(z[i] - beta * sum((e[1] * (1 if e[0] == T else z[e[0]]) for e in rows[i]))) for i in states))
                        success = min(v.values()) > 0 and r / (1 - beta) <= settings['tolerance']
                        stop = success or w['sample_updates'] == limit
                        if stop and (not ended):
                            w['interrupted_episodes'] += 1
                        w['residual_scans'] += 1
                        w['residual_edge_visits'] += sum(map(len, rows.values()))
                        check(pos < len(history), 'checkpoint_not_missing')
                        record = history[pos]
                        pos += 1
                        check(record['updates'] == w['sample_updates'], 'checkpoint_index')
                        check(record['residual'] == r, 'checkpoint_residual')
                        check(record['error_bound_estimate'] == r / (1 - beta), 'checkpoint_bound')
                        check(record['z'] == {**{bits(i): z[i] for i in states}, T: 1.0}, 'checkpoint_all_values')
                        check(record['visits'] == {bits(i): v[i] for i in states}, 'checkpoint_all_visits')
                        check(record['restarts'] == {bits(i): starts[i] for i in states}, 'checkpoint_all_restarts')
                        check(record['work'] == w, 'checkpoint_all_work')
                        check(record['rng_state_sha256'] == hashlib.sha256(repr(rng.getstate()).encode()).hexdigest(), 'checkpoint_rng_binding')
                        check(max((abs(z[i] - hpf[i]) for i in states)) <= r / (1 - beta) + 1e-15, 'observed_error_below_estimate')
                        if success:
                            status = 'RESIDUAL_TOLERANCE_MET'
                    if ended or stop:
                        break
                    x = y
                if stop:
                    break
            check(pos == len(history), 'checkpoint_exact_coverage')
            check(final['status'] == status, 'termination_status')
            check(final['work'] == w, 'final_work')
            check(final['z'] == {**{bits(i): z[i] for i in states}, T: 1.0}, 'final_values')
            check(final['visits'] == {bits(i): v[i] for i in states}, 'final_visits')
            check(final['restarts'] == {bits(i): starts[i] for i in states}, 'final_restarts')
            check(w['episodes'] == sum((w[k] for k in ['target_events', 'zero_boundary_events', 'cap_events', 'interrupted_episodes'])), 'episode_accounting')
            s = saved_summary[name, seed]
            check(s['status'] == status, 'summary_status')
            for k, val in w.items():
                check(int(s[k]) == val, 'summary_work')
            check(abs(float(s['residual']) - r) < 1e-17, 'summary_residual')
            check(float(s['observed_desirability_error']) == max((abs(z[i] - ref['z'][bits(i)]) for i in states)), 'summary_observed_error')
            matched = 0
            proper = 0
            for i in states:

                def mode(values):
                    return min(rows[i], key=lambda e: (-math.log(e[1]) - math.log(1.0 if e[0] == T else values[e[0]]), 2 ** n + 1 if e[0] == T else 2 ** n - e[0], len(e[3]), e[3]))
                a, b = (mode(z), mode(hpf))
                actual = final['modes'][bits(i)]
                check(actual['successor'] == bits(a[0]) and actual['action'] == list(a[3]), 'mode_independent')
                matched += a[0] == b[0] and a[3] == b[3]
                x = i
                seen = set()
                steps = 0
                while x != target and x not in seen:
                    seen.add(x)
                    pair = final['modes'][bits(x)]
                    j = x
                    for k in pair['action']:
                        j ^= 1 << n - k
                    real = update(j)
                    check(pair['successor'] == (T if real == target else bits(real)), 'policy_physical_realization')
                    x = real
                    steps += 1
                proper += x == target
                check(final['policy_paths'][bits(i)]['proper'] == (x == target), 'properness_independent')
                check(final['policy_paths'][bits(i)]['hitting_time'] == steps, 'hitting_time_independent')
            check(final['matching_mode_action_pairs'] == matched, 'mode_count')
            check(final['proper_states'] == proper, 'proper_count')
            total_updates += w['sample_updates']
            total_checkpoints += pos
            print(name, seed, 'independent replay', w['sample_updates'], status, flush=True)
    return {'status': 'PASS', 'implementation': 'independent_sampling_replay', 'assertions': sum(checks.values()), 'groups': dict(checks), 'total_updates_replayed': total_updates, 'checkpoints_replayed': total_checkpoints, 'reference_dps': [80, 140], 'maximum_dense_reference_error': max_reference_delta, 'maximum_80_140_difference': str(max_hp_delta)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    settings = json.loads(Path(__file__).with_name('sampling_settings.json').read_text())
    result = audit(a.output, settings)
    a.report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(result['status'], result['assertions'], 'checks', flush=True)
if __name__ == '__main__':
    main()
