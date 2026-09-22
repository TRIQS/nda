#!/usr/bin/env python3
"""Run the tracked nda benchmarks from one existing build.

Runs every benchmark binary in the tracked build directory, collects its results in benchmarks.json, and records
wall time, case counts, build provenance and machine description in metadata.json.

Workers automatically use distinct allowed physical cores with local NUMA memory
binding on Linux with numactl, including when --workers 1 is specified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
import threading

import common
import metadata
import placement
import reporting



def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bindir", type=Path,
                        default=repo_root / "build/benchmarks/tracked",
                        help="directory holding the built tracked benchmark binaries")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="where to write JSON (default: bench-results/<timestamp>)")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="--benchmark_repetitions; >1 is needed for any error bar")
    parser.add_argument("--min-time", default=None,
                        help="--benchmark_min_time, e.g. 0.1s or 100x (default: GB's own)")
    parser.add_argument("--filter", default=None,
                        help="--benchmark_filter regex; a leading - excludes")
    parser.add_argument("--workers", type=int, required=True,
                        help="number of workers on distinct physical cores with NUMA binding")
    args = parser.parse_args()
    if args.workers < 1 or args.repetitions < 1:
        parser.error("workers and repetitions must be positive")
    args.bindir = args.bindir.resolve()

    if args.outdir is None:
        args.outdir = repo_root / "bench-results" / time.strftime("%Y%m%dT%H%M%SZ",
                                                                  time.gmtime())
    if not args.bindir.is_dir():
        sys.exit(f"error: bindir not found: {args.bindir}\n"
                 "enable Build_Benchs and build the tracked benchmark targets first")

    binaries = common.find_binaries(args.bindir)
    if not binaries:
        sys.exit(f"error: no executable benchmark binaries in {args.bindir}")
    selected = []
    for binary in binaries:
        cases, families, ops = common.case_counts(common.list_cases(binary, args.filter))
        if cases:
            selected.append({'name': binary.name, 'cases': cases, 'families': families, 'ops': ops})
    if not selected:
        parser.error('No benchmark cases match the selected filter')

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path = args.outdir / 'benchmarks.json'
    if any((args.outdir / name).exists() for name in ('benchmarks.json', 'metadata.json', 'output.log')):
        parser.error('Use a fresh output directory for each run')

    document = metadata.create_metadata({
        'repetitions': args.repetitions, 'min_time': args.min_time, 'filter': args.filter,
        'workers': args.workers,
    })

    workers = min(args.workers, len(selected))
    try:
        placements = placement.worker_placements(workers)
        placement.validate_placements(placements)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"CPU/NUMA binding failed: {exc}")

    build_dir = args.bindir.parents[1]
    document['builds'] = {'current': metadata.describe_build(build_dir)}
    document['builds']['current']['bindir'] = str(args.bindir)
    metadata.record_placements(document, placements)

    print(f"bindir       {args.bindir}")
    print(f"outdir       {args.outdir}")
    print(f"repetitions  {args.repetitions}")
    print(f"min-time     {args.min_time or '<gb default>'}")
    print(f"filter       {args.filter or '<none>'}")
    print(f"workers      {workers}")
    for i, worker in enumerate(placements):
        print(f"worker {i}     {' '.join(worker['pin_command']) or '<none>'}")
    print(f"binaries     {len(selected)}\n")

    entries, totals = [], {"families": 0, "cases": 0}
    for selected_entry in selected:
        totals['cases'] += selected_entry['cases']
        totals['families'] += selected_entry['families']
        worker = placements[len(entries) % workers]
        entry = selected_entry | worker
        entries.append(entry)

    document['totals'] = totals | {'binaries': len(entries), 'wall_seconds': None}
    document['binaries'] = [placement.public_metadata(entry) for entry in entries]
    common.write_json(args.outdir / 'metadata.json', document)
    results = {'binaries': [
        {'name': entry['name'], 'exit_code': None} for entry in entries]}
    result_indices = {entry['name']: i for i, entry in enumerate(entries)}
    checkpoint_lock = threading.Lock()
    common.write_json(results_path, results)
    log_output = common.output_logger(args.outdir / 'output.log')

    def run_binary(entry):
        binary = args.bindir / entry["name"]
        print(f"{binary.name:<22} {entry['families']:5d} families  {entry['cases']:5d} cases  started", flush=True)

        execution = common.execute(
            binary, prefix=entry['pin_command'], pattern=args.filter,
            repetitions=args.repetitions, min_time=args.min_time,
            cases=entry['cases'], log_output=log_output, log_label=binary.name)
        code = execution.get('exit_code')
        failed = code != 0 or bool(execution.get('error'))
        secs = round(execution["wall_seconds"], 3)

        print(f"{binary.name:<22} {secs:8.1f}s" + ("" if not failed
                                 else f"  FAILED (exit {code}, see output.log)"))
        with checkpoint_lock:
            if metadata.record_context(document['builds']['current'], binary.name, execution):
                common.write_json(args.outdir / 'metadata.json', document)
            results['binaries'][result_indices[binary.name]] = {'name': binary.name, **execution}
            common.write_json(results_path, results)

    def run_worker(entries):
        for entry in entries:
            run_binary(entry)

    suite_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run_worker, [entries[i::workers] for i in range(workers)]))

    total_secs = round(time.perf_counter() - suite_start, 3)

    document['finished_at'] = common.timestamp()
    document['totals'] = totals | {'binaries': len(entries), 'wall_seconds': total_secs}
    common.write_json(args.outdir / 'metadata.json', document)
    reporting.print_run_summary(document, results, args.outdir)

    return 0 if all(e.get('exit_code') == 0 and not e.get('error') for e in results['binaries']) else 1


if __name__ == "__main__":
    sys.exit(main())
