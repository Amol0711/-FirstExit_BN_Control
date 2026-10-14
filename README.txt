FIRST-EXIT BOOLEAN-NETWORK SIMULATIONS
====================================

This package contains executable numerical models, solvers, experiment settings,
reference results, regression tests and numerical plotting utilities. It is
self-contained for the commands below after its dependencies are installed.

Repository identity: https://github.com/Amol0711/-FirstExit_BN_Control
This archive does not assert the existence of a published commit or release.
No project software license is included. Consult LICENSE_STATUS.txt before reuse
or redistribution. Dependencies are installed separately, not bundled.
MODEL_NOTES.txt identifies the numerical models and their source attributions.

EXPERIMENT_PROTOCOL.txt is the consolidated record of execution environments,
random seeds, stopping settings, benchmark timing and memory procedures, and
backend conventions. Machine-readable settings and complete numerical records
remain the executable and numerical authority.

1. Environment and installation
-------------------------------
The exact-reference reproduction environment is Linux x86-64, CPython 3.13.5,
NumPy 2.3.5 and mpmath 1.3.0. Optional plots use Matplotlib 3.10.8. The standard
library supplies the fixed-update-budget sampler and the primary comparison
solvers. NumPy supplies the dense large-system backend. mpmath supplies the
independent high-precision numerical checks.

From this directory, create an isolated environment OUTSIDE the package, install
the pinned numerical dependencies, then run the reproduction command:

    python3.13 -m venv ../first_exit_environment
    . ../first_exit_environment/bin/activate
    python -m pip install -r requirements.txt
    python -B reproduce.py --output ../first_exit_run

The output directory must be absent or empty and outside the package. Run from
an extracted directory, not inside a ZIP viewer. No network access is used by
any experiment once dependencies are installed. The fresh performance-campaign
entry point uses Linux resource limits, affinity and RSS conventions; it is not
claimed portable to other operating systems.

FILE_HASHES.json binds every distributed file except itself. reproduce.py checks
both coverage and hashes and verifies that the package remains unchanged. Do not
add caches, virtual environments or generated figures beneath the package root.
The sampling-specific source lock also checks its five declared source/input
files before running. These are integrity checks, not scientific certificates.

2. One-command numerical reproduction
------------------------------------

    python -B reproduce.py --output ../first_exit_run

This command runs the regression suite, six deterministic producers, all ten
fixed-seed sampling runs, an independent sampling replay, and numerical checks
of the recorded 133-configuration, four-method comparison. It writes a final
REPRODUCTION.json and individual logs in the new output directory.

The comparison checks replay numerical values, policies and work accounting
with the timing clock disabled. They independently recompute original costs
and high-precision reference values, and reaggregate the existing timing and
memory observations. They do not create a new performance campaign, replace
any measured trial, select a different action, or overwrite recorded outputs.

There are 52 unique deterministic reference files and 24 sampling numerical
files compared exactly, plus two sampling metadata files checked separately.
Every history line, final policy, termination status and reference value is
included in those comparisons. Exact mismatches are errors. Another Python,
math-library or numerical-backend environment can differ in floating arithmetic
or random primitives; the command does not silently accept such differences by
loosening a tolerance.

3. Individual commands
----------------------
All examples use new output directories outside the package.

    python -B -m unittest discover -s code -v
    python -B code/run_deterministic.py --output ../deterministic_run
    python -B code/run_sampling.py --output ../sampling_run
    python -B code/replay_sampling.py --output ../sampling_run --report ../sampling_replay.json
    python -B code/verify_comparison.py --run results/comparison --output ../comparison_checks

The deterministic producer runner executes run_exactness.py, run_recovery.py,
run_regularization.py, run_scaling.py, run_phenotype.py and run_library_checks.py.
Each also accepts --output for targeted numerical generation.

To collect a genuinely NEW performance campaign, use a nonexistent directory:

    python -B code/run_comparison.py --output ../new_comparison_run
    python -B code/evaluate_comparison.py ../new_comparison_run

These two optional commands collect new, environment-dependent measurements.
They are not part of the default reproduction command. The second command runs
separate numerical reference checks and records their own diagnostic durations.
Do not substitute a new campaign for the fixed measurements in results/comparison.
The comparison driver preserves unsuccessful trials and explicit error statuses.

4. Inputs and numerical conventions
----------------------------------
code/boolean_networks.py contains V1--V4, R1/R2 and nine-node model definitions,
action families, costs and stable state/action ordering. See MODEL_NOTES.txt for
model equations, node order, targets, scientific scope and attributions.

code/sampling_models.json contains complete Boolean truth tables for R1 and R2.
code/sampling_settings.json fixes the full sampling protocol. The settings are:

    models                         R1, R2
    seeds                          1101, 1102, 1103, 1104, 1105
    transient state cost          1
    node weights                  0.4, 1.0
    reference smoothing epsilon   0
    initial desirability          1 at every transient state
    stepsize exponent omega       0.75
    maximum sampled updates       1,000,000 per run
    full residual scan interval   1,000 sampled updates
    episode path cap              transient basin size (3 for R1, 7 for R2)
    stopping tolerance            0.001 on residual/(1-exp(-1))

Uniform independent restarts use random.Random(seed).choice on ascending
Boolean strings. Successor sampling uses one random() draw, ordered cumulative
probabilities, and a strict draw < cumulative comparison, with the terminal
symbol last. Stepsizes use the update count for the visited state. A successful
scan can interrupt an episode; it cannot declare success before all basin states
have received updates. At the hard cap, a successful scan takes precedence;
otherwise the outcome remains BUDGET_EXHAUSTED.

Representative ties in the model/sampling interfaces use exact decimal costs,
then action cardinality and lexicographic node order. Modal ties use the declared
score followed by the semi-tensor-product successor index, terminal last. The
matched comparison uses its declared lexicographic action convention and integer
tenths for deterministic costs. No numerical tolerance merges distinct scores.

The recovery sweep uses the Cartesian grid

    0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0

for both eligible-node weights on R1/R2, covering 640 state-weight records.
The diagnostic gap excludes every exact deterministic minimum and is infinite
when no worse successor exists. Its positive test uses Delta > 2*epsilon + 1e-9.
There are 23 infinite gaps; these are retained in dedicated files and also in
the complete data. Smoothed-reference checks use the actual smoothed probabilities.
The regularization levels are exactly 0, 1e-6, 1e-4, 1e-3 and 1e-2.

The general solve_desirability interface defaults to tolerance 1e-14 and a
100,000-sweep cap. The automatic backend uses Python dense elimination through
128 basin states and NumPy above that size; an unavailable NumPy import triggers
a sparse-iteration fallback. This changes the backend and is not the pinned
reference environment. The separate episode-budgeted z_learning interface is
not the fixed-update-budget sampling experiment entry point.

code/comparison_settings.json declares all 133 configurations: V2--V4, 128 R1/R2
weight configurations and P9-r1/P9-r2. V1 is an exactness-only validation case;
S70/S100/S200 are analytic counts, not timed Boolean-network instances. The four
methods are U-SP, C-SP, C-KL-MODE and TW-DP-H. The primary KL iteration tolerance
is 1e-12 with a 10,000-sweep cap. Independent comparison references use 80 decimal
digits and a 1e-60 error-estimate target; sampling replay uses 80 and 140 digits.
Comparison mode checks use a 1e-50 numerical tie threshold, without asserting
exact equality. The comparison gap classification uses a 1e-10 scale-dependent
boundary band; it is distinct from the recovery-sweep guard.

5. Reference outputs and schemas
-------------------------------
results/deterministic contains numerical TSV tables and four numerical TXT logs.
The principal groups are exactness, recovery, regularization, scaling and
phenotype. TSV column headers are the schemas. State indices, action strings,
costs, visit/intervention counts and infinite/unreached outcomes are retained.
Scaling files count records or operations, not measured allocated bytes or time.

results/sampling contains two model reference JSON files, ten *_history.jsonl
files, ten *_result.json files, sampling_summary.csv, state_visits.csv,
execution.json and output_hashes.json. Each history line contains every state
value, update/restart counter, sampled/scan work count, residual, error estimate
and an RNG-state digest at one full scan. Each result file records the final
state values, policy laws, deterministic modes, physical successors, properness,
termination status and observed error against a separately computed reference.

The two metadata files bind a recorded or generated execution. Environment and
source digests need not match across distinct executions. The numerical fields,
all histories and final result records are checked independently of that metadata.

results/comparison/timing contains 532 fixed method/configuration blocks, each
with three warmups and ten measured repetitions. results/comparison/memory
contains 532 separate no-warmup worker records. Every physical rollout, solver
status and numerical result signature is retained. The run manifest records
the environment, fixed block order, one orchestration interruption, resumption
without outcome-based selection, and complete coverage.

results/comparison/evaluation contains every state-policy quality record,
configuration summary, raw timing-trial row, gap/mode diagnostic, independent
numerical reference check and 133 high-precision reference files. These are
machine-readable numerical outputs. Per-state fields distinguish complete
policy optimality from the first action, properness from accuracy, and a failed
or unresolved diagnostic from an established result.

6. Qualifications and retained adverse results
---------------------------------------------
The fixed sampling study has seven RESIDUAL_TOLERANCE_MET outcomes and three
BUDGET_EXHAUSTED outcomes. All three exhausted runs belong to R2 and retain their
complete one-million-update histories. Across the ten runs, there are 4,536,000
sampled updates and 4,536 residual scans. Budget exhaustion is neither a proof
of unreachability nor a convergence conclusion.

The 640-record recovery sweep retains 376 positive-gap/zero-loss classifications,
247 zero-loss records not passing that test, and 17 positive-local-loss records.
The matched comparison retains 22 of 1,682 state-policy cases with a positive
original-cost loss for C-KL-MODE, including 17 R2 cases and five P9-r2 states.
The largest recorded original-cost loss is 1.0. Four exact-mode equalities remain
numerically unresolved. No adverse case is replaced by an optimal action.

Residuals, high-precision agreement and positive floating-point diagnostics are
numerical evidence, not outward-rounded interval certificates. Properness and
mode agreement are checked separately from the residual stopping criterion.
The numerical nine-node example does not establish biological effectiveness or
clinical applicability; its interventions are abstract bit flips.

Recorded timings are fixed observations from their reported execution. Small
sub-millisecond timings are resolution-sensitive. Construction, reference
formation, solve, recovery and properness phases are separate. Startup/import,
serialization and independent numerical-reference work are excluded from the
controller-phase total. Additional gap/reference-check durations are individual
post-campaign measurements; sums with medians are composites, not medians of
repeated combined trials. The comparison order seed is 20261011, thread count is
one, the limit is 600 seconds per trial and address space is capped at 2 GiB.

Memory records distinguish actual array payloads, reachable Python allocations,
nominal numeric-equivalent entries and fresh-process peak RSS. A zero increase
above an after-import high-water RSS baseline does not mean zero allocations.
SciPy is listed in the recorded environment but is not imported by the numerical
algorithms and is not a required dependency.

7. Optional numerical plots
---------------------------

    python -m pip install -r requirements-plot.txt
    python -B plot_results.py --output ../numerical_plots

For regenerated numerical data:

    python -B plot_results.py --data ../first_exit_run --output ../regenerated_plots

The plotting script reads only numerical TSV/JSON/JSONL inputs and produces four
separate PNG files: sampling_error_estimates.png, recovery_local_loss.png,
phenotype_hitting_times.png and storage_counts.png. Every sampling trace is shown,
including budget exhaustion. Infinite hitting times remain explicitly labeled;
synthetic storage counts are identified as analytic, not memory measurements.
No external document or typesetting installation is needed.
