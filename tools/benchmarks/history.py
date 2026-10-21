#!/usr/bin/env python3
"""Measure tracked benchmarks at every commit of a range and chart each case's history.

    python3 history.py <start> <end> --filter '<regex>' [--rounds 6] [--workers N] [--outdir DIR]

Both ends are included; the commits are the first-parent chain from start to end. Each commit is built from
its own tree with its own benchmark_tracked preset (a commit that fails to build, for instance one older than the
preset, is a gap); the binaries are cached per commit in /tmp/nda-history-benchmarks. Each case is
measured on one pinned core: a warm-up launch per commit, then --rounds rounds in which every commit is
launched once, in alternating order. Per commit the median CPU time and the paired comparison with the
previous measured commit go to history.json; history-charts.py draws them (needs matplotlib).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import common
import metadata
import stats

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent.parent
# Built binaries and their build.json (metadata.describe_build) per commit, reused across runs. The key names neither
# the compiler nor the build.json format, so delete the cache when switching compilers or changing describe_build.
CACHE = Path('/tmp/nda-history-benchmarks')


def commit_range(start, end):
    """The first-parent commits from start to end, both included, oldest first."""
    start = common.run(['git', '-C', str(REPO), 'rev-parse', f'{start}^{{commit}}'])
    # committer date: it follows the chain, unlike the author date
    log = common.run(['git', '-C', str(REPO), 'log', '--reverse', '--first-parent', '--format=%H%x00%h%x00%cs%x00%s', end])
    commits = [dict(zip(('sha', 'short', 'date', 'subject'), line.split('\0'))) for line in log.splitlines()]
    shas = [commit['sha'] for commit in commits]
    if start not in shas:
        raise ValueError(f'{start[:common.SHORT_SHA]} is not in the first-parent history of {end}')
    return commits[shas.index(start):]


def build(commit, cache_dir, workdir, log_path):
    """The description (metadata.describe_build) of commit's tracked benchmarks, built from its own tree with its own
    preset unless cache_dir already holds them; the binaries end up in cache_dir/bin and the rest of the build is deleted.
    Raises RuntimeError on a failed build."""
    if (cache_dir / 'build.json').exists():
        return json.loads((cache_dir / 'build.json').read_text())
    source, build_dir = workdir / f'{commit["short"]}-source', workdir / f'{commit["short"]}-build'
    for path in (source, build_dir):
        shutil.rmtree(path, ignore_errors=True)
    # --shared: the checkout reads the repository's objects instead of copying them
    common.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(source)])
    common.run(['git', '-C', str(source), 'checkout', '--quiet', '--detach', commit['sha']])
    with log_path.open('w') as log:
        status = subprocess.run(['sh', str(TOOLS / 'build-benchmarks.sh'), str(source), str(build_dir)],
                                stdout=log, stderr=subprocess.STDOUT).returncode
    binaries = common.find_binaries(build_dir / 'benchmarks/tracked') if not status else []
    if status or not binaries:
        raise RuntimeError(f'exit code {status}' if status else 'no benchmark binaries were built')
    description = metadata.describe_build(build_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)  # a half-written entry left by an interrupted run
    (cache_dir / 'bin').mkdir(parents=True)
    for binary in binaries:  # nda_c is static: the binaries stand alone
        shutil.copy2(binary, cache_dir / 'bin' / binary.name)
    common.write_json(cache_dir / 'build.json', description)
    shutil.rmtree(source)
    shutil.rmtree(build_dir)
    return description


def measure_case(key, binaries, cpu, rounds, log):
    """One case on one core: a warm-up launch per commit, then rounds in which every commit is launched once in
    alternating order; per commit the median time and the paired comparison (stats.analyze) with the previous measured
    commit."""
    suite, name = key
    commits = [sha for sha, cases in binaries.items() if key in cases]  # in range order

    def launch(sha, label):  # the CPU time in ns, None when the launch failed (see output.log)
        try:
            return common.measure(binaries[sha][key], name, cpu, log, f'{sha[:common.SHORT_SHA]}-{label}')
        except RuntimeError:
            return None

    launched = [sha for sha in commits if launch(sha, 'warmup') is not None]
    times = {sha: [] for sha in launched}
    for index in range(rounds):
        for sha in launched if index % 2 == 0 else launched[::-1]:  # alternate the direction: no commit always precedes its neighbour
            times[sha].append(launch(sha, f'round-{index + 1}'))
    measured = [sha for sha in launched if None not in times[sha]]
    reference = measured[0] if measured else None  # the charts draw their band and speedup axis around it

    per_commit, previous = {}, None
    for sha in commits:
        if sha not in measured:  # the warm-up or a round failed
            per_commit[sha] = {'status': 'launch_failed'}
            continue
        entry = {'status': 'measured', 'median_ns': statistics.median(times[sha]),
                 'se_ns': statistics.stdev(times[sha]) / math.sqrt(rounds), 'noise_percent': stats.noise_percent(times[sha])}
        if previous is not None:
            entry['step'] = stats.analyze(times[previous], times[sha])
        per_commit[sha] = entry
        previous = sha

    steps = [sha[:common.SHORT_SHA] for sha in measured if per_commit[sha].get('step', {}).get('status') in ('improvement', 'regression')]
    if measured:
        first, last = per_commit[measured[0]]['median_ns'] / common.NS_PER_US, per_commit[measured[-1]]['median_ns'] / common.NS_PER_US
        print(f'{suite}: {name}: {first:.3g} us at {measured[0][:common.SHORT_SHA]} .. {last:.3g} us at {measured[-1][:common.SHORT_SHA]}'
              f' ({first / last:.2f}x); steps at {", ".join(steps) or "none"}', flush=True)
    else:
        print(f'{suite}: {name}: no commit could be measured', flush=True)
    return {'suite': suite, 'name': name, 'cpu': cpu, 'commits': commits, 'times_ns': times, 'reference': reference,
            'per_commit': per_commit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('start', help='first commit of the range (included)')
    parser.add_argument('end', help='last commit of the range (included)')
    parser.add_argument('--filter', help='Google Benchmark regex selecting the cases (default: all)')
    parser.add_argument('--rounds', type=int, default=6, help='launches per commit and case, even (default: %(default)s)')
    parser.add_argument('--workers', type=int, help='cases measured in parallel on distinct cores (default: all physical cores)')
    parser.add_argument('--no-pin', action='store_true', help='disable CPU pinning')
    parser.add_argument('--outdir', type=Path, help='results directory (default: history-results/<start>..<end>)')
    args = parser.parse_args(argv)
    if args.rounds < 2 or args.rounds % 2:
        parser.error('--rounds must be even and at least 2')

    commits = commit_range(args.start, args.end)
    outdir = (args.outdir or Path('history-results') / f'{commits[0]["short"]}..{commits[-1]["short"]}').resolve()
    if (outdir / 'history.json').exists():
        parser.error(f'{outdir} already holds a history; use a fresh --outdir')
    (outdir / 'builds').mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='nda-history-'))
    print(f'{len(commits)} commits, {commits[0]["short"]} .. {commits[-1]["short"]}; builds in {CACHE}', flush=True)

    history = metadata.create_metadata(rounds=args.rounds, filter=args.filter)
    history.update(range={'start': commits[0]['sha'], 'end': commits[-1]['sha']}, commits=commits, cases=[])
    history_path = outdir / 'history.json'

    binaries = {}  # {sha: {(suite, name): binary}} for the built commits, in range order
    for commit in commits:
        cache_dir = CACHE / commit['sha'][:12]
        log_path = outdir / 'builds' / f'{commit["short"]}.log'
        cached = (cache_dir / 'build.json').exists()
        print(f'{commit["short"]} {commit["date"]} {commit["subject"]}: ', end='', flush=True)
        started = time.perf_counter()
        try:
            description = build(commit, cache_dir, workdir, log_path)
            cases = common.discover_cases(cache_dir / 'bin', args.filter)  # fails when a binary cannot run
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            commit['build'] = 'failed'
            commit['error'] = str(exc)
            print(f'failed ({exc})' + (f'; see {log_path}' if log_path.exists() else ''), flush=True)
            continue
        commit['build'] = 'cached' if cached else 'built'
        history['builds'][commit['sha']] = description
        binaries[commit['sha']] = cases
        print(commit['build'] + ('' if cached else f' in {time.perf_counter() - started:.0f} s') + f', {len(binaries[commit["sha"]])} cases', flush=True)
    shutil.rmtree(workdir, ignore_errors=True)
    if not binaries:
        sys.exit('No commit could be built')
    # What a commit changes in the compiler, flags or benchmark sources is recorded against the first built commit
    # (a different compiler means the cache holds builds of another toolchain).
    first = next(iter(binaries))
    for commit in commits:
        if commit['sha'] in binaries and commit['sha'] != first:
            commit['build_differences'] = metadata.build_differences(history['builds'][first], history['builds'][commit['sha']])

    cases = sorted({key for found in binaries.values() for key in found})
    if not cases:
        sys.exit('No case matches the filter in any built commit')
    cpus = common.worker_cpus(args.workers, pin=not args.no_pin)[:len(cases)]
    history['settings'].update(workers=len(cpus), cpus=cpus)
    print(f'{len(cases)} cases across {len(binaries)} commits on {len(cpus)} workers', flush=True)
    log = common.output_logger(outdir / 'output.log')
    def run_worker(worker):
        # A case's whole history stays on one core, as in the two-build comparison.
        return [measure_case(key, binaries, cpus[worker], args.rounds, log) for key in cases[worker::len(cpus)]]

    with ThreadPoolExecutor(max_workers=len(cpus)) as executor:
        for measured in executor.map(run_worker, range(len(cpus))):
            history['cases'].extend(measured)
    history['finished_at'] = common.timestamp()
    common.write_json(history_path, history)
    print(f'Results: {history_path}', flush=True)
    try:
        importlib.import_module('history-charts').main([str(outdir)])
    except ImportError as exc:  # no matplotlib in this Python
        print(f'Charts not drawn ({exc}); run: python3 {TOOLS / "history-charts.py"} {outdir}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
