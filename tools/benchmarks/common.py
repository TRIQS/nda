"""Google Benchmark discovery, execution, JSON validation, and timing units."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
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


def read_result(path: Path, cases: int, repetitions: int) -> dict:
    result = json.loads(path.read_text())
    counts = {}
    for row in result["benchmarks"]:
        if row.get("error_occurred"):
            raise ValueError(f"Benchmark reported an error: {row}")
        if row.get("run_type") != "iteration":
            continue
        if row["time_unit"] not in NANOSECONDS or not all(
                math.isfinite(row[key]) and row[key] > 0 for key in ("cpu_time", "real_time")):
            raise ValueError(f"Invalid benchmark timing: {row}")
        name = row.get("run_name", row["name"])
        repetition = row.get("repetition_index", 0)
        observed = counts.setdefault(name, set())
        if repetition in observed or repetition not in range(repetitions):
            raise ValueError(f"Invalid or duplicate repetition: {row}")
        observed.add(repetition)
    if len(counts) != cases or any(len(indices) != repetitions for indices in counts.values()):
        raise ValueError(f"Expected {cases} nonempty cases with {repetitions} repetitions in {path.name}")

    return result


def measured_seconds(document: dict) -> float:
    """Sum timed-loop totals, excluding Google Benchmark aggregate rows."""
    return sum(row['real_time'] * row['iterations'] * NANOSECONDS[row['time_unit']] / 1e9
               for row in document.get('benchmarks', []) if row.get('run_type') == 'iteration')


def output_logger(path):
    """Append complete invocation sections without interleaving parallel workers."""
    path.touch(exist_ok=False)
    lock = threading.Lock()

    def append(label, text):
        with lock, path.open('a', encoding='utf-8') as stream:
            stream.write(f'=== {label} ===\n{text.rstrip()}\n\n')

    return append


def execute(binary, *, prefix, pattern, repetitions, min_time, cases, log_output, log_label, timeout=None,
            min_warmup_time=None):
    """Keep measurements in JSON and append console diagnostics to output.log."""
    with tempfile.TemporaryDirectory(prefix='nda-benchmark-') as temporary:
        folder = Path(temporary)
        output = folder / 'result.json'
        log = folder / 'output.log'
        command = [*prefix, str(binary), f'--benchmark_out={output}',
                   '--benchmark_out_format=json', f'--benchmark_repetitions={repetitions}']
        if min_time:
            command.append(f'--benchmark_min_time={min_time}')
        if min_warmup_time:
            command.append(f'--benchmark_min_warmup_time={min_warmup_time:g}')
        if pattern:
            command.append(f'--benchmark_filter={pattern}')
        result = {'started_at': timestamp()}
        start = time.monotonic()
        try:
            with log.open('wb') as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            result['exit_code'] = process.returncode
            if process.returncode:
                raise ValueError(f'Process exited with {process.returncode}; see output.log')
            result['benchmark'] = read_result(output, cases, repetitions)
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            result['error'] = str(exc)
        result.update(finished_at=timestamp(), wall_seconds=time.monotonic() - start)
        diagnostics = log.read_text(errors='replace') if log.exists() else ''
        if 'benchmark' not in result and output.exists():
            diagnostics += '\nUnvalidated benchmark JSON:\n' + output.read_text(errors='replace')
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
    cases = {}
    for binary in find_binaries(bindir):
        for name in list_cases(binary, pattern):
            key = (binary.name, name)
            if key in cases:
                raise ValueError(f'Duplicate benchmark identity: {key}')
            cases[key] = binary
    return cases
