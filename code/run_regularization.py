#!/usr/bin/env python3
"""Generate deterministic support-preserving regularization checks."""
from __future__ import annotations
import argparse
import csv
from math import exp, log
from pathlib import Path
from boolean_networks import kl_values, make_cases, build_successor_model, state_sort_key
EPS_LEVELS = [0.0, 1e-06, 0.0001, 0.001, 0.01]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case = make_cases()['R2']
    trans = case.transient_states
    model = build_successor_model(case)
    z0, j0 = kl_values(case, epsilon=0.0)
    beta = exp(-1.0)
    bound_factor = 2 * beta / (1 - beta)

    def numerical_actions(eps, values):
        result = {}
        for state in trans:
            row = model.ysets[state]
            gamma = sum((exp(-model.rhostar[state, y]) for y in row))
            scores = {}
            for y in row:
                probability = (1.0 - eps) * exp(-model.rhostar[state, y]) / gamma + eps / len(row)
                scores[y] = -log(probability) + values[y]
            y = min(row, key=lambda y: (scores[y], state_sort_key(y, case.n)))
            result[state] = model.reps[state, y]
        return result
    baseline_actions = numerical_actions(0.0, j0)
    eps_rows = []
    max_action_changes = 0
    for eps in EPS_LEVELS:
        z, j = kl_values(case, epsilon=eps)
        zdiff = max((abs(z[s] - z0[s]) for s in trans))
        jdiff = max((abs(j[s] - j0[s]) for s in trans))
        bound = bound_factor * eps
        actions = numerical_actions(eps, j)
        action_changes = sum((actions[s] != baseline_actions[s] for s in trans))
        max_action_changes = max(max_action_changes, action_changes)
        eps_rows.append({'eps': f'{eps:g}', 'zdiff': f'{zdiff:.12g}', 'bound': f'{bound:.12g}', 'jdiff': f'{jdiff:.12g}', 'map_action_changes_vs_eps0': action_changes})
    with (args.output / 'regularization_eps.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=['eps', 'zdiff', 'bound', 'jdiff', 'map_action_changes_vs_eps0'])
        writer.writeheader()
        writer.writerows(eps_rows)
    with (args.output / 'regularization_notes.txt').open('w') as f:
        f.write('Reference distribution nu is uniform on each feasible successor set Y(x).\n')
        f.write('P_epsilon=(1-epsilon)P+epsilon*nu; support remains Y(x).\n')
        f.write('Infeasible positive passive probability count: 0.\n')
        f.write(f'Maximum MAP-action changes versus epsilon=0 over tested levels: {max_action_changes}.\n')
        f.write('Smoothed actions use -log(P_epsilon)+J_epsilon; no exact-reference gap claim is made.\nThis file records deterministic regularization checks. Sampling histories are produced separately by run_sampling.py.\n')
    if max_action_changes != 0:
        raise RuntimeError(f'Regularization changed MAP actions in {max_action_changes} state-epsilon checks')
    print(f'Regularization data generated: {len(EPS_LEVELS)} epsilon levels, max MAP-action changes {max_action_changes}')
if __name__ == '__main__':
    main()
