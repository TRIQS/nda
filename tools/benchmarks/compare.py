#!/usr/bin/env python3
"""Compare two built NDA benchmark suites in alternating paired rounds."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import analysis
import common
import metadata
import placement
import reporting
import rules


def measure(binary, name, prefix, label, min_time, timeout, repetitions, log_output):
    result = common.execute(
        binary, prefix=prefix, pattern=f'^{re.escape(name)}$', repetitions=repetitions,
        min_time=min_time, cases=1, timeout=timeout,
        log_output=log_output, log_label=f'{binary.name}: {name}: {label}')
    if 'benchmark' in result:
        try:
            timings = []
            for row in result['benchmark']['benchmarks']:
                if row.get('run_type') != 'iteration':
                    continue
                if row.get('run_name', row['name']) != name:
                    raise ValueError(f'Expected exactly {name}, got {row["name"]}')
                timings.append(row['cpu_time'] * common.NANOSECONDS[row['time_unit']])
            result['cpu_time_ns'] = min(timings)
        except ValueError as exc:
            result['error'] = str(exc)
    return result


def compare_case(key, binaries, worker, args, log_output):
    suite, name = key
    result = {'suite': suite, 'name': name,
              'warmups': {}, 'rounds': [], 'analysis': {'status': 'incomplete'}}

    def invoke(side, label):
        return measure(binaries[side][key], name, worker['pin_command'], label,
                       args.min_time, args.timeout, args.repetitions, log_output)

    for side in ('candidate', 'baseline'):
        result['warmups'][side] = invoke(side, f'warmup-{side}')
    if not any(sample.get('error') for sample in result['warmups'].values()):
        for index in range(args.rounds):
            order = ['candidate', 'baseline'] if index % 2 == 0 else ['baseline', 'candidate']
            pair = {'round': index + 1, 'order': order, 'samples': {}}
            for side in order:
                pair['samples'][side] = invoke(side, f'round-{index + 1}-{side}')
            result['rounds'].append(pair)
            if any(sample.get('error') for sample in pair['samples'].values()):
                break
    result['analysis'] = analysis.analyze(result['rounds'], args.rounds, args.threshold,
                                          args.min_improvement, args.max_noise)
    case_analysis = result['analysis']
    ratio = f"{case_analysis['median_ratio']:.4f}x" if 'median_ratio' in case_analysis else 'unavailable'
    print(f'{suite}: {name}: {ratio} ({case_analysis["status"]})', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-build', type=Path, required=True)
    parser.add_argument('--candidate-build', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Google Benchmark repetitions per invocation; use their minimum CPU time')
    parser.add_argument('--min-time', default='0.1s')
    parser.add_argument('--filter')
    parser.add_argument('--workers', type=int, required=True, help='number of cases measured in parallel on distinct cores')
    parser.add_argument('--no-pin', action='store_true', help='disable CPU/NUMA pinning for local testing')
    parser.add_argument('--threshold', type=float, default=rules.DEFAULT_THRESHOLD,
                        help='cutoff on the paired t statistic of the per-round differences')
    parser.add_argument('--min-improvement', type=float, default=rules.DEFAULT_MIN_IMPROVEMENT,
                        help='required change as a percentage of the baseline mean before a signal')
    parser.add_argument('--max-noise', type=float, default=rules.DEFAULT_MAX_NOISE,
                        help='a side whose per-round cv exceeds this percentage makes the case too_noisy')
    parser.add_argument('--timeout', type=float, help='optional wall-time limit per invocation, in seconds')
    args = parser.parse_args(argv)
    if args.rounds < 2 or args.rounds % 2:
        parser.error('--rounds must be positive and even (default: 6)')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error('--threshold must be finite and positive')
    if not math.isfinite(args.min_improvement) or args.min_improvement < 0:
        parser.error('--min-improvement must be finite and nonnegative')
    if not math.isfinite(args.max_noise) or args.max_noise <= 0:
        parser.error('--max-noise must be finite and positive')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be finite and positive')
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / 'comparison.json'
    metadata_path = args.outdir / 'metadata.json'
    if any((args.outdir / name).exists() for name in ('comparison.json', 'metadata.json', 'output.log', 'cases')):
        parser.error('Use a fresh output directory for each comparison')
    document = metadata.create_metadata({
        'repetitions': args.repetitions, 'min_time': args.min_time, 'filter': args.filter,
        'workers': args.workers, 'rounds': args.rounds, 'repetition_statistic': 'min',
        'warmup_invocations_per_side': 1, 'rule': 'paired_t', 'threshold_t': args.threshold,
        'min_improvement_percent': args.min_improvement, 'max_noise_percent': args.max_noise,
        'timeout_seconds': args.timeout,
    })
    manifest = {'coverage': {}, 'counts': {}, 'cases': []}
    log_output = common.output_logger(args.outdir / 'output.log')
    measurement_start = None
    outcome = 'failed'
    try:
        builds = {'baseline': args.baseline_build.resolve(), 'candidate': args.candidate_build.resolve()}
        for side, build in builds.items():
            document['builds'][side] = metadata.describe_build(build)
        analysis.check_comparable(document['builds']['baseline'], document['builds']['candidate'])
        binaries = {side: common.discover_cases(build / 'benchmarks/tracked', args.filter) for side, build in builds.items()}
        matched_cases = sorted(binaries['baseline'].keys() & binaries['candidate'].keys())
        missing = []
        for side, status in (('candidate', 'added'), ('baseline', 'removed')):
            other = 'candidate' if side == 'baseline' else 'baseline'
            for suite, name in sorted(binaries[side].keys() - binaries[other].keys()):
                missing.append({'suite': suite, 'name': name, 'analysis': {'status': status}})
        manifest['cases'] = missing
        manifest['coverage'] = {'baseline': len(binaries['baseline']), 'candidate': len(binaries['candidate']),
                                'common': len(matched_cases), 'unmatched': len(missing)}
        if not matched_cases:
            raise ValueError('No matching benchmark cases in the two builds')
        suites = sorted({suite for suite, _ in matched_cases})
        workers = min(args.workers, len(matched_cases))
        placements = placement.unpinned_workers(workers) if args.no_pin else placement.worker_placements(workers)
        placement.validate_placements(placements)
        metadata.record_placements(document, placements)
        for suite in suites:
            names = [name for binary, name in matched_cases if binary == suite]
            cases, families, ops = common.case_counts(names)
            document['binaries'].append({'name': suite, 'cases': cases, 'families': families, 'ops': ops})
        document['totals'].update(cases=len(matched_cases), binaries=len(suites),
                                  families=sum(b['families'] for b in document['binaries']))
        case_indices = {key: len(missing) + i for i, key in enumerate(matched_cases)}
        manifest['cases'].extend({'suite': suite, 'name': name, 'worker_index': index % workers,
                                  'analysis': {'status': 'incomplete'}}
                                 for index, (suite, name) in enumerate(matched_cases))
        manifest['counts'] = dict(Counter(case['analysis']['status'] for case in manifest['cases']))
        common.write_json(metadata_path, document)
        common.write_json(manifest_path, manifest)
        checkpoint_lock = threading.Lock()
        measurement_start = time.perf_counter()

        def checkpoint(case):
            # Only finished cases enter the manifest; serialize shared context and file updates.
            with checkpoint_lock:
                context_added = False
                for samples in [case['warmups']] + [pair['samples'] for pair in case['rounds']]:
                    for side, sample in samples.items():
                        context_added |= metadata.record_context(document['builds'][side], case['suite'], sample)
                if context_added:
                    common.write_json(metadata_path, document)
                manifest['cases'][case_indices[case['suite'], case['name']]] = case
                manifest['counts'] = dict(Counter(case['analysis']['status'] for case in manifest['cases']))
                common.write_json(manifest_path, manifest)

        def run_worker(index):
            # Keep both sides and all rounds of a case on the same worker.
            for key in matched_cases[index::workers]:
                case = compare_case(key, binaries, placements[index], args, log_output)
                case['worker_index'] = index
                checkpoint(case)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(run_worker, range(workers)))
        outcome = 'incomplete' if manifest['counts'].get('incomplete') else 'complete'
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        manifest['error'] = str(exc)
        log_output('Comparison failed', str(exc))
        print(f'Comparison failed: {exc}', file=sys.stderr)
    document['finished_at'] = common.timestamp()
    if measurement_start is not None:
        document['totals']['wall_seconds'] = round(time.perf_counter() - measurement_start, 3)
    common.write_json(metadata_path, document)
    common.write_json(manifest_path, manifest)
    reporting.write_comparison_report(args.outdir / 'comparison.md', manifest, document)
    print(f'Results: {manifest_path}', flush=True)
    return 0 if outcome == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
