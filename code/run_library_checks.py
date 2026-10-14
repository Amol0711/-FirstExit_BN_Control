#!/usr/bin/env python3
"""Run numerical consistency checks for state indices, compressed values and rollouts."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from boolean_networks import actions_up_to, all_states, bits_to_delta_index, build_successor_model, delta_index_to_bits, deterministic_rollout, deterministic_values, kl_values, map_recovered_rollout, exactness_summary_row, make_cases, normalize_cycle_indices, phenotype9_case, phenotype9_update, raw_deterministic_values, recovery_evaluation, rational_deterministic_values, recovery_evaluations, uncontrolled_attractors

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=False)
    args = parser.parse_args()
    lines: list[str] = []
    for n in range(1, 10):
        indices = range(1, 2 ** n + 1) if n <= 5 else [1, 2, 2 ** (n - 1), 2 ** n - 1, 2 ** n]
        for i in indices:
            s = delta_index_to_bits(i, n)
            assert bits_to_delta_index(s) == i, (n, i, s)
    lines.append('delta-index round trips: OK')
    for case in make_cases().values():
        raw = raw_deterministic_values(case)
        comp = deterministic_values(case)
        max_err = 0.0
        for s in case.transient_states:
            if math.isfinite(raw[s]) or math.isfinite(comp[s]):
                max_err = max(max_err, abs(raw[s] - comp[s]))
        assert max_err <= 1e-10, (case.name, max_err)
        row = exactness_summary_row(case)
        lines.append(f"{case.name}: uncompressed/compressed value error {max_err:.3g}, A={row['A']}, E_B={row['E_B']}, MAP_exact={row['MAP_exact_states']}")
    for case in make_cases().values():
        for diag in recovery_evaluations(case):
            assert diag.local_loss <= diag.two_epsilon + 1e-08, diag
    lines.append('MAP-recovery local suboptimality checks: OK')
    attractors = {normalize_cycle_indices(cycle) for cycle in uncontrolled_attractors(phenotype9_update, 9)}
    expected = {(512,), (128, 272, 488, 506, 511)}
    assert attractors == expected, attractors
    p9 = phenotype9_case(flip_nodes=(9,))
    values = deterministic_values(p9)
    model = build_successor_model(p9)
    rows_250, flips_250 = deterministic_rollout(p9, delta_index_to_bits(250, 9), values=values, successor_cache=(model.ysets, model.reps))
    rows_511, flips_511 = deterministic_rollout(p9, delta_index_to_bits(511, 9), values=values, successor_cache=(model.ysets, model.reps))
    assert [r['state_delta'] for r in rows_250] == [250, 367, 496, 512]
    assert flips_250 == 1
    assert [r['state_delta'] for r in rows_511] == [511, 512]
    assert flips_511 == 1
    lines.append('9-node attractor and rollout checks: OK')
    p9_r1 = phenotype9_case(flip_nodes=tuple(range(1, 10)), weights={i: 1.0 for i in range(1, 10)}, max_flip_size=1)
    p9_r2_actions = actions_up_to(tuple(range(1, 10)), 2)
    assert len(p9_r1.actions) == 10
    assert len(p9_r2_actions) == 46
    assert len(all_states(9)) == 512
    p9_model = build_successor_model(p9_r1)
    p9_det = deterministic_values(p9_r1)
    _, p9_kl = kl_values(p9_r1)
    p9_exact = rational_deterministic_values(p9_r1)
    p9_map_exact = 0
    p9_max_tau = 0
    for s in p9_r1.transient_states:
        diag = recovery_evaluation(p9_r1, s, p9_det, p9_kl, p9_model, values_det_exact=p9_exact)
        p9_map_exact += int(diag.map_is_deterministic_optimal)
        rows, _ = map_recovered_rollout(p9_r1, s, values_kl=p9_kl, model=p9_model, values_det=p9_det, max_steps=20)
        assert int(rows[-1]['state_delta']) == 512
        p9_max_tau = max(p9_max_tau, int(rows[-1]['t']))
    assert p9_map_exact == len(p9_r1.transient_states)
    assert p9_max_tau == 4
    lines.append('9-node action-budget/MAP helpers: OK (A_r1=10, A_r2=46, P9-r1 MAP exact=511/511)')
    text = '\n'.join(lines) + '\n'
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'shared_library_checks.txt').write_text(text)
    print(text, end='')
if __name__ == '__main__':
    main()
