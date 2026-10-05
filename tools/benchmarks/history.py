#!/usr/bin/env python3
"""Measure tracked benchmarks at every commit of a range and chart each case's history.

    python3 history.py <start> <end> --filter '<regex>' [--rounds 6] [--workers N] [--outdir DIR]

Both ends are included; the commits are the first-parent chain from start to end. Each commit is built from
its own tree with its own benchmark_tracked preset (a commit without one is skipped with a warning, one that
fails to build is a gap); the binaries are cached per commit in /tmp/nda-history-benchmarks. Each case is
measured on one pinned core: a warm-up launch per commit, then --rounds rounds in which every commit is
launched once, in alternating order. Per commit the median CPU time and the paired comparisons with the
start commit and with the previous commit go to history.json; history-charts.py draws them (needs matplotlib).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import analysis
import common
import metadata
import placement
import rules

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent.parent
# Built binaries per commit, reused across runs; the key does not name the compiler, so delete it when switching.
CACHE = Path('/tmp/nda-history-benchmarks')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit_range(start, end):
    """The first-parent commits from start to end, both included, oldest first."""
    start, end = git(REPO, 'rev-parse', f'{start}^{{commit}}'), git(REPO, 'rev-parse', f'{end}^{{commit}}')
    chain = git(REPO, 'rev-list', '--reverse', '--first-parent', end).splitlines()
    if start not in chain:
        raise ValueError(f'{start[:12]} is not in the first-parent history of {end[:12]}')
    commits = []
    for sha in chain[chain.index(start):]:
        short, date, subject = git(REPO, 'log', '-1', '--format=%h%x00%cs%x00%s', sha).split('\0')  # committer date: follows the chain
        commits.append({'sha': sha, 'short': short, 'date': date, 'subject': subject})
    return commits


def missing_preset(commit):
    """Why the commit cannot be configured with the benchmark_tracked preset, or None when its CMakePresets.json defines it."""
    listed = subprocess.run(['git', '-C', str(REPO), 'cat-file', '-p', f'{commit["sha"]}:CMakePresets.json'],
                            capture_output=True, text=True)
    if listed.returncode:
        return 'no CMakePresets.json'
    try:
        presets = json.loads(listed.stdout).get('configurePresets', [])
    except ValueError as exc:
        return f'CMakePresets.json is not valid JSON ({exc})'
    if not any(p.get('name') == metadata.PRESET for p in presets):
        return f'no {metadata.PRESET} preset in CMakePresets.json'
    return None


def build(commit, cache_dir, workdir, log_path):
    """The tracked benchmarks of commit, built from its own tree with its own preset, from the cache or built now: the
    build's description (metadata.describe_build) with 'bindir' pointing at the cached binaries. Raises RuntimeError on
    a failed build."""
    description_path = cache_dir / 'build.json'
    if not description_path.exists():
        build_into_cache(commit, cache_dir, workdir, log_path)
    description = json.loads(description_path.read_text())
    description['bindir'] = str(cache_dir / 'bin')  # wherever the cache lives now, not where it was built
    return description


def build_into_cache(commit, cache_dir, workdir, log_path):
    """Check out and build; then keep the binaries and build.json under cache_dir and delete the rest."""
    source, build_dir = workdir / f'{commit["short"]}-source', workdir / f'{commit["short"]}-build'
    for path in (source, build_dir):
        shutil.rmtree(path, ignore_errors=True)
    # --shared: the checkout reads the repository's objects instead of copying them
    subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(source)], check=True)
    git(source, 'checkout', '--quiet', '--detach', commit['sha'])
    with log_path.open('w') as log:
        status = subprocess.run(['sh', str(TOOLS / 'build-benchmarks.sh'), str(source), str(build_dir)],
                                stdout=log, stderr=subprocess.STDOUT).returncode
    binaries = common.find_binaries(build_dir / 'benchmarks/tracked') if not status else []
    if status or not binaries:
        raise RuntimeError(f'exit code {status}' if status else 'no benchmark binaries were built')
    description = metadata.describe_build(build_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)  # a half-written entry left by an interrupted run
    bindir = cache_dir / 'bin'
    bindir.mkdir(parents=True)
    for binary in binaries:
        shutil.copy2(binary, bindir / binary.name)
    description['build'] = None  # nda_c is static: the binaries stand alone
    common.write_json(cache_dir / 'build.json', description)
    shutil.rmtree(source)
    shutil.rmtree(build_dir)


def measure_case(key, binaries, worker, args, log_output):
    """One case on one core: a warm-up launch per commit, then rounds in which every commit is launched once in
    rotated order; per commit the median time and the paired comparisons with the first and the previous commit."""
    suite, name = key
    commits = [sha for sha, cases in binaries.items() if key in cases]  # in range order
    result = {'suite': suite, 'name': name, 'commits': commits, 'warmups': {}, 'rounds': []}

    def launch(sha, label):
        return common.measure(binaries[sha][key], name, worker['pin_command'], f'{sha[:10]}-{label}',
                              args.min_time, args.min_warmup_time, 1, log_output)

    for sha in commits:
        result['warmups'][sha] = launch(sha, 'warmup')
    launched = [sha for sha in commits if not result['warmups'][sha].get('error')]
    for index in range(args.rounds if launched else 0):
        order = launched if index % 2 == 0 else launched[::-1]  # alternate the direction: no commit always precedes its neighbour
        result['rounds'].append({'round': index + 1, 'order': order,
                                 'samples': {sha: launch(sha, f'round-{index + 1}') for sha in order}})

    def compare(baseline, candidate):  # the two-sided view analysis.analyze expects, paired within each round
        pairs = [{'samples': {'baseline': r['samples'][baseline], 'candidate': r['samples'][candidate]}} for r in result['rounds']]
        return analysis.analyze(pairs, args.rounds, args.alpha, args.min_improvement, args.max_noise)

    times = {sha: [r['samples'][sha].get('cpu_time_ns') for r in result['rounds']] for sha in launched}
    measured = [sha for sha in launched if None not in times[sha]]
    result['reference'] = measured[0] if measured else None  # the speedups are relative to the first measured commit
    stats, previous = {}, None
    for sha in commits:
        if sha not in launched:
            stats[sha] = {'status': 'launch_failed'}
            continue
        if sha in measured:
            entry = {'status': 'measured', 'median_ns': statistics.median(times[sha]),
                     'se_ns': statistics.stdev(times[sha]) / math.sqrt(len(times[sha])), 'noise_percent': rules.noise_percent(times[sha])}
        else:
            entry = {'status': 'incomplete'}
        if result['reference'] not in (None, sha):
            entry['vs_reference'] = compare(result['reference'], sha)
        if previous is not None:
            entry['step'] = compare(previous, sha)
        stats[sha] = entry
        previous = sha
    result['per_commit'] = stats

    steps = [sha[:7] for sha in measured if stats[sha].get('step', {}).get('status', '').endswith('signal')]
    if measured:
        first, last = stats[measured[0]]['median_ns'] / 1000, stats[measured[-1]]['median_ns'] / 1000
        print(f'{suite}: {name}: {first:.3g} us at {measured[0][:7]} .. {last:.3g} us at {measured[-1][:7]}'
              f' ({first / last:.2f}x); steps at {", ".join(steps) or "none"}', flush=True)
    else:
        print(f'{suite}: {name}: no commit could be measured', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('start', help='first commit of the range (included)')
    parser.add_argument('end', help='last commit of the range (included)')
    parser.add_argument('--filter', help='Google Benchmark regex selecting the cases (default: all)')
    parser.add_argument('--rounds', type=int, default=6, help='launches per commit and case, even (default: 6)')
    parser.add_argument('--min-time', default='0.2s')
    parser.add_argument('--min-warmup-time', type=float, default=0.1, help='seconds of discarded warm-up inside each launch')
    parser.add_argument('--workers', type=int, help='cases measured in parallel on distinct cores (default: all physical cores)')
    parser.add_argument('--no-pin', action='store_true', help='disable CPU/NUMA pinning')
    parser.add_argument('--alpha', type=float, default=rules.DEFAULT_ALPHA, help='one-sided false-positive rate of the paired tests')
    parser.add_argument('--min-improvement', type=float, default=rules.DEFAULT_MIN_IMPROVEMENT, help='percent change a signal needs')
    parser.add_argument('--max-noise', type=float, default=rules.DEFAULT_MAX_NOISE, help='cv percent above which a commit is too noisy')
    parser.add_argument('--outdir', type=Path, help='results directory (default: history-results/<start>..<end>)')
    args = parser.parse_args(argv)
    if args.rounds < 2 or args.rounds % 2:
        parser.error('--rounds must be even and at least 2')
    rules.check_parameters(args.alpha, args.min_improvement)

    commits = commit_range(args.start, args.end)
    outdir = (args.outdir or Path('history-results') / f'{commits[0]["short"]}..{commits[-1]["short"]}').resolve()
    if (outdir / 'history.json').exists():
        parser.error(f'{outdir} already holds a history; use a fresh --outdir')
    (outdir / 'builds').mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='nda-history-'))
    print(f'{len(commits)} commits, {commits[0]["short"]} .. {commits[-1]["short"]}; builds in {CACHE}', flush=True)

    settings = {'repetitions': 1, 'min_time': args.min_time, 'min_warmup_time': args.min_warmup_time, 'filter': args.filter,
                'rounds': args.rounds, 'repetition_statistic': 'min', 'warmup_invocations_per_commit': 1,
                'alpha': args.alpha, 'min_improvement_percent': args.min_improvement, 'max_noise_percent': args.max_noise}
    document = metadata.create_metadata(settings)
    history = {'range': {'start': commits[0]['sha'], 'end': commits[-1]['sha']},
               'commits': commits, 'settings': settings, 'cases': []}
    history_path, metadata_path = outdir / 'history.json', outdir / 'metadata.json'

    binaries = {}  # {sha: {(suite, name): binary}} for the built commits, in range order
    for commit in commits:
        cache_dir = CACHE / commit['sha'][:12]
        log_path = outdir / 'builds' / f'{commit["short"]}.log'
        cached = (cache_dir / 'build.json').exists()
        print(f'{commit["short"]} {commit["date"]} {commit["subject"]}: ', end='', flush=True)
        reason = missing_preset(commit)
        if reason:
            commit['build'] = 'skipped'
            commit['error'] = reason
            print(f'warning: skipped, {reason}', flush=True)
            continue
        started = time.perf_counter()
        try:
            description = build(commit, cache_dir, workdir, log_path)
            cases = common.discover_cases(Path(description['bindir']), args.filter)  # fails when a binary cannot run
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            commit['build'] = 'failed'
            commit['error'] = str(exc)
            print(f'failed ({exc})' + (f'; see {log_path}' if log_path.exists() else ''), flush=True)
            continue
        commit['build'] = 'cached' if cached else 'built'
        document['builds'][commit['sha']] = description
        binaries[commit['sha']] = cases
        print(commit['build'] + ('' if cached else f' in {time.perf_counter() - started:.0f} s') + f', {len(binaries[commit["sha"]])} cases', flush=True)
    shutil.rmtree(workdir, ignore_errors=True)
    if not binaries:
        sys.exit('No commit could be built')
    # The same compiler throughout, else the cache belongs to another toolchain; benchmark sources, preset variables
    # and flags a commit changes are recorded against the first built commit.
    first = next(iter(binaries))
    for commit in commits:
        if commit['sha'] in binaries and commit['sha'] != first:
            commit['build_differences'] = analysis.check_comparable(document['builds'][first], document['builds'][commit['sha']])

    cases = sorted({key for found in binaries.values() for key in found})
    if not cases:
        sys.exit('No case matches the filter in any built commit')
    placements = placement.unpinned_workers(args.workers or os.cpu_count()) if args.no_pin else placement.worker_placements(args.workers)
    placements = placements[:len(cases)]
    workers = len(placements)
    metadata.record_placements(document, placements)
    history['cases'] = [{'suite': suite, 'name': name, 'worker_index': index % workers, 'per_commit': {}}
                        for index, (suite, name) in enumerate(cases)]
    common.write_json(metadata_path, document)
    common.write_json(history_path, history)
    print(f'{len(cases)} cases across {len(binaries)} commits on {workers} workers', flush=True)
    log_output = common.output_logger(outdir / 'output.log')
    lock = threading.Lock()

    def checkpoint(index, case):
        with lock:
            context_added = False
            for samples in [case['warmups']] + [r['samples'] for r in case['rounds']]:
                for sample in samples.values():
                    context_added |= metadata.record_context(document, sample)
            if context_added:
                common.write_json(metadata_path, document)
            history['cases'][index] = case
            common.write_json(history_path, history)

    def run_worker(worker):
        for index in range(worker, len(cases), workers):  # a case's whole history on one core
            case = measure_case(cases[index], binaries, placements[worker], args, log_output)
            case['worker_index'] = worker
            checkpoint(index, case)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run_worker, range(workers)))
    document['finished_at'] = common.timestamp()
    document['totals'].update(cases=len(cases), wall_seconds=round(time.perf_counter() - started, 3))
    common.write_json(metadata_path, document)
    common.write_json(history_path, history)
    print(f'Results: {history_path}', flush=True)
    try:
        importlib.import_module('history-charts').main([str(outdir)])
    except ImportError as exc:  # no matplotlib in this Python
        print(f'Charts not drawn ({exc}); run: python3 {TOOLS / "history-charts.py"} {outdir}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
