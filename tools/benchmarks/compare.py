#!/usr/bin/env python3
"""Compare two built NDA benchmark suites in alternating paired rounds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import common
import metadata
import reporting
import stats


def compare_case(key, binaries, cpu, rounds, log):
    """One case on one CPU: a warm-up launch per side, then rounds rounds alternating which side goes first."""
    suite, name = key
    case = {'suite': suite, 'name': name, 'cpu': cpu}

    def launch(side, label):
        return common.measure(binaries[side][key], name, cpu, log, label)

    try:
        for side in ('candidate', 'baseline'):
            launch(side, f'warmup-{side}')
        times = {'baseline': [], 'candidate': []}
        for index in range(rounds):
            for side in ('candidate', 'baseline') if index % 2 == 0 else ('baseline', 'candidate'):
                times[side].append(launch(side, f'round-{index + 1}-{side}'))
    except RuntimeError as exc:
        case.update(status='incomplete', error=str(exc))
    else:
        case.update(stats.analyze(times['baseline'], times['candidate']), times_ns=times)
    speedup = f"{case['speedup']:.4f}x" if 'speedup' in case else 'unavailable'
    print(f'{suite}: {name}: {speedup} ({case["status"]})', flush=True)
    return case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-build', type=Path, required=True)
    parser.add_argument('--candidate-build', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, default=Path('benchmark-results'))
    parser.add_argument('--rounds', type=int, default=12, help='rounds per case, even (default: %(default)s)')
    parser.add_argument('--filter', help='Google Benchmark regex selecting the cases (default: all)')
    parser.add_argument('--workers', type=int,
                        help='number of cases measured in parallel on distinct cores (default: all physical cores)')
    parser.add_argument('--no-pin', action='store_true', help='disable CPU pinning for local testing')
    args = parser.parse_args(argv)
    if args.rounds < 2 or args.rounds % 2:
        parser.error('--rounds must be even and at least 2')
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log = common.output_logger(outdir / 'output.log')
    results = metadata.create_metadata(rounds=args.rounds, filter=args.filter)
    results['cases'] = []
    try:
        builds = {'baseline': args.baseline_build.resolve(), 'candidate': args.candidate_build.resolve()}
        for side, build in builds.items():
            results['builds'][side] = metadata.describe_build(build)
        results['build_differences'] = metadata.build_differences(results['builds']['baseline'], results['builds']['candidate'])
        binaries = {side: common.discover_cases(build / 'benchmarks/tracked', args.filter) for side, build in builds.items()}
        for status, side, other in (('added', 'candidate', 'baseline'), ('removed', 'baseline', 'candidate')):
            results['cases'] += [{'suite': suite, 'name': name, 'status': status}
                                 for suite, name in sorted(binaries[side].keys() - binaries[other].keys())]
        matched = sorted(binaries['baseline'].keys() & binaries['candidate'].keys())
        if not matched:
            raise ValueError(f'No benchmark case is present in both builds (baseline build: {len(binaries["baseline"])} cases, '
                             f'candidate build: {len(binaries["candidate"])})')
        cpus = common.worker_cpus(args.workers, pin=not args.no_pin)[:len(matched)]
        results['settings'].update(workers=len(cpus), cpus=cpus)
        print(f'{len(matched)} cases on {len(cpus)} workers', flush=True)

        def run_worker(index):
            # Every len(cpus)-th case, all on this worker's CPU: both sides and all rounds of a case on the same core.
            return [compare_case(key, binaries, cpus[index], args.rounds, log) for key in matched[index::len(cpus)]]

        with ThreadPoolExecutor(max_workers=len(cpus)) as executor:
            for cases in executor.map(run_worker, range(len(cpus))):
                results['cases'] += cases
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        results['error'] = str(exc)
        log('Comparison failed', str(exc))
        print(f'Comparison failed: {exc}', file=sys.stderr)
    results['finished_at'] = common.timestamp()
    common.write_json(outdir / 'comparison.json', results)
    reporting.write_comparison_report(outdir / 'comparison.md', results)
    print(f'Results: {outdir / "comparison.json"}', flush=True)
    failed = 'error' in results or any(case['status'] == 'incomplete' for case in results['cases'])
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
