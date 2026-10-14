#!/usr/bin/env python3
"""Run a new comparison campaign in isolated Linux workers. Timing and memory campaigns are separate. Existing records are never overwritten."""
from __future__ import annotations
from time import perf_counter_ns
_IMPORT_START = perf_counter_ns()
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from comparison_solvers import configurations, run_trial, METHODS, digest
_IMPORT_NS = perf_counter_ns() - _IMPORT_START
HERE = Path(__file__).resolve().parent
SOURCE_FILES = ['comparison_solvers.py', 'run_comparison.py', 'comparison_settings.json', 'boolean_networks.py']

def now():
    return datetime.now(timezone.utc).isoformat()

def hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES}

def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

def alarm_handler(signum, frame):
    raise TimeoutError('Per-trial time limit exceeded')

def worker(config, method, memory):
    settings = json.loads((HERE / 'comparison_settings.json').read_text())
    timing = settings['timing']
    resource.setrlimit(resource.RLIMIT_AS, (timing['memory_limit_bytes'], timing['memory_limit_bytes']))
    signal.signal(signal.SIGALRM, alarm_handler)
    baseline = rss_bytes()
    trials = []
    first = None
    expected = None
    count = 1 if memory else timing['warmups'] + timing['measured_repetitions']
    for i in range(count):
        gc.collect()
        phase = 'memory' if memory else 'warmup' if i < timing['warmups'] else 'measured'
        signal.setitimer(signal.ITIMER_REAL, timing['time_limit_seconds_per_trial'])
        started = perf_counter_ns()
        try:
            result = run_trial(config, method, memory)
            elapsed = perf_counter_ns() - started
            peak = rss_bytes()
            if expected is None:
                expected = result['result_signature']
            if expected != result['result_signature']:
                raise AssertionError('Repeated numerical result changed within a block')
            if first is None:
                first = result
            trials.append({'phase': phase, 'index': i, 'status': result['failure_status'], 'phases': result['phases'], 'trial_wall_ns': elapsed, 'result_signature': result['result_signature']})
            if memory:
                first['memory'].update(peak_RSS_bytes=peak, empty_worker_RSS_bytes=baseline, RSS_above_empty_worker_bytes=max(0, peak - baseline))
            if result is not first:
                del result
        except (TimeoutError, MemoryError, Exception) as e:
            trials.append({'phase': phase, 'index': i, 'status': type(e).__name__, 'message': str(e), 'trial_wall_ns': perf_counter_ns() - started, 'traceback': traceback.format_exc()})
            signal.setitimer(signal.ITIMER_REAL, 0)
            return {'status': 'FAILED', 'config': config, 'method': method, 'memory': memory, 'trials': trials, 'first_result': first, 'import_ns': _IMPORT_NS, 'source_hashes': hashes(), 'timestamp': now()}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return {'status': 'OK', 'config': config, 'method': method, 'memory': memory, 'trials': trials, 'first_result': first, 'import_ns': _IMPORT_NS, 'source_hashes': hashes(), 'timestamp': now()}

def environment():
    versions = {}
    for name in ['mpmath', 'numpy', 'scipy']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = 'unknown'
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu = line.split(':', 1)[1].strip()
            break
    return {'python': sys.version, 'platform': platform.platform(), 'cpu': cpu, 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'versions': versions, 'clock': vars(__import__('time').get_clock_info('perf_counter')), 'thread_environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}}

def campaign(out: Path):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    settings = json.loads((HERE / 'comparison_settings.json').read_text())
    configs = configurations(settings)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '1'
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if hasattr(os, 'sched_getaffinity'):
        affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(affinity)})
    rng = random.Random(settings['timing']['order_seed'])
    order = []
    for config in configs:
        methods = list(METHODS)
        rng.shuffle(methods)
        order.extend(((config, m) for m in methods))
    stamp = {'status': 'RUNNING', 'started': now(), 'settings': settings, 'source_hashes': hashes(), 'environment': environment(), 'block_order': [(c['id'], m) for c, m in order], 'realization': 'Fixed seeded block order; 3 warmups/10 measured repetitions within each isolated block. Separate no-warmup memory blocks. No cross-method cached model graph.', 'per_trial_limits_enforced': True}
    (out / 'run_manifest.json').write_text(json.dumps(stamp, indent=2) + '\n')
    failures = []
    for memory in [False, True]:
        phase = 'memory' if memory else 'timing'
        folder = out / phase
        folder.mkdir()
        for idx, (config, method) in enumerate(order):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', json.dumps(config), '--method', method] + (['--memory'] if memory else [])
            t = perf_counter_ns()
            process = subprocess.run(command, capture_output=True, text=True, env=os.environ, timeout=(1 if memory else 13) * settings['timing']['time_limit_seconds_per_trial'] + 60)
            elapsed = perf_counter_ns() - t
            name = f"{idx:04d}_{config['id']}_{method}.json"
            try:
                record = json.loads(process.stdout)
            except Exception:
                record = {'status': 'WORKER_FAILURE', 'config': config, 'method': method, 'stdout': process.stdout, 'stderr': process.stderr, 'returncode': process.returncode}
            record['worker_process_elapsed_ns'] = elapsed
            record['worker_returncode'] = process.returncode
            if process.stderr:
                record['stderr'] = process.stderr
            (folder / name).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            if record['status'] != 'OK':
                failures.append((phase, config['id'], method, record['status']))
            if idx % 50 == 0 or idx == len(order) - 1:
                print(phase, idx + 1, '/', len(order), 'failures', len(failures), flush=True)
    stamp['ended'] = now()
    stamp['failures'] = failures
    stamp['sources_unchanged'] = stamp['source_hashes'] == hashes()
    stamp['status'] = 'COMPLETE' if not failures and stamp['sources_unchanged'] else 'COMPLETE_WITH_FAILURES'
    (out / 'run_manifest.json').write_text(json.dumps(stamp, indent=2) + '\n')
    print(stamp['status'], flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    p.add_argument('--worker')
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--memory', action='store_true')
    a = p.parse_args()
    if a.worker:
        print(json.dumps(worker(json.loads(a.worker), a.method, a.memory), allow_nan=False))
    elif a.output:
        campaign(a.output.resolve())
    else:
        p.error('--output or --worker is required')
if __name__ == '__main__':
    main()
