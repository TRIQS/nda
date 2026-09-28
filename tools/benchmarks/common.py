"""Google Benchmark discovery, execution, and timing units."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path

NANOSECONDS = {"ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def find_binaries(bindir):
    return sorted(path for path in bindir.glob("*") if path.is_file() and os.access(path, os.X_OK))


def list_cases(binary, pattern=None):
    command = [str(binary), "--benchmark_list_tests=true"]
    if pattern:
        command.append(f"--benchmark_filter={pattern}")
    output = subprocess.run(command, capture_output=True, text=True, check=True)
    return [name.strip() for name in output.stdout.splitlines() if name.strip()]


def output_logger(path):
    """Append complete invocation sections without interleaving parallel workers."""
    lock = threading.Lock()

    def append(label, text):
        with lock, path.open('a', encoding='utf-8') as stream:
            stream.write(f'=== {label} ===\n{text.rstrip()}\n\n')

    return append


def execute(binary, *, prefix, pattern, repetitions, min_time, min_warmup_time, timeout, log_output, log_label):
    """Keep measurements in JSON and append console diagnostics to output.log."""
    with tempfile.TemporaryDirectory(prefix='nda-benchmark-') as temporary:
        output = Path(temporary) / 'result.json'
        log = Path(temporary) / 'output.log'
        command = [*prefix, str(binary), f'--benchmark_out={output}', '--benchmark_out_format=json',
                   f'--benchmark_repetitions={repetitions}', f'--benchmark_min_time={min_time}',
                   f'--benchmark_min_warmup_time={min_warmup_time:g}', f'--benchmark_filter={pattern}']
        result = {'started_at': timestamp()}
        start = time.monotonic()
        try:
            with log.open('wb') as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            result['exit_code'] = process.returncode
            if process.returncode:
                raise ValueError(f'Process exited with {process.returncode}; see output.log')
            benchmark = json.loads(output.read_text())
            if failed := [row for row in benchmark['benchmarks'] if row.get('error_occurred')]:
                raise ValueError(f'Benchmark reported an error: {failed[0]}')
            result['benchmark'] = benchmark
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            result['error'] = str(exc)
        result.update(finished_at=timestamp(), wall_seconds=time.monotonic() - start)
        diagnostics = log.read_text(errors='replace')
        if result.get('error'):
            diagnostics += '\nError: ' + result['error']
        log_output(log_label, f"Command: {shlex.join(command)}\nStarted: {result['started_at']}\n"
                   f"Finished: {result['finished_at']}\n{diagnostics}")
        return result


def case_counts(names):
    """Collapse names such as f64/A2_C_layout,A2_C_layout/add/64 by removing /<N>."""
    families = {re.sub(r"/\d+$", "", name) for name in names}
    ops = {family.rsplit("/", 1)[-1] for family in families}
    return len(names), len(families), len(ops)


def discover_cases(bindir, pattern=None):
    # Google Benchmark accepts a name registered twice and runs both under one ^name$ filter.
    cases = {}
    for binary in find_binaries(bindir):
        for name in list_cases(binary, pattern):
            key = (binary.name, name)
            if key in cases:
                raise ValueError(f'Duplicate benchmark identity: {key}')
            cases[key] = binary
    return cases
