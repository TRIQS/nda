"""Google Benchmark discovery, execution, case names, and timing units."""

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


def execute(binary, *, prefix, pattern, repetitions, min_time, min_warmup_time, log_output, log_label):
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
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            result['exit_code'] = process.returncode
            if process.returncode:
                raise ValueError(f'Process exited with {process.returncode}; see output.log')
            benchmark = json.loads(output.read_text())
            if failed := [row for row in benchmark['benchmarks'] if row.get('error_occurred')]:
                raise ValueError(f'Benchmark reported an error: {failed[0]}')
            result['benchmark'] = benchmark
        except (OSError, ValueError) as exc:
            result['error'] = str(exc)
        result.update(finished_at=timestamp(), wall_seconds=time.monotonic() - start)
        diagnostics = log.read_text(errors='replace')
        if result.get('error'):
            diagnostics += '\nError: ' + result['error']
        log_output(log_label, f"Command: {shlex.join(command)}\nStarted: {result['started_at']}\n"
                   f"Finished: {result['finished_at']}\n{diagnostics}")
        return result


def measure(binary, name, prefix, label, min_time, min_warmup_time, repetitions, log_output):
    """One launch of one case: execute() with the name as an anchored filter, plus 'cpu_time_ns', the minimum
    CPU time per iteration over the launch's repetitions, when the launch succeeded."""
    result = execute(
        binary, prefix=prefix, pattern=f'^{re.escape(name)}$', repetitions=repetitions,
        min_time=min_time, min_warmup_time=min_warmup_time,
        log_output=log_output, log_label=f'{binary.name}: {name}: {label}')
    if 'benchmark' in result:
        result['cpu_time_ns'] = min(row['cpu_time'] * NANOSECONDS[row['time_unit']]
                                    for row in result['benchmark']['benchmarks'] if row['run_type'] == 'iteration')
    return result


# The C++ type behind each value-type tag of nda_bench::type_tag (benchmarks/tracked/bench_inputs.hpp). The order is
# the one reports and charts use; a tag missing here still works, shown under its tag after the listed ones.
VALUE_TYPES = {'f64': 'double', 'c128': 'complex<double>', 'f32': 'float', 'c64': 'complex<float>'}


def value_type_label(tag):
    """'double (f64)' for a listed tag, the tag itself otherwise."""
    return f'{VALUE_TYPES[tag]} ({tag})' if tag in VALUE_TYPES else tag


def value_type_key(tag):
    """Sort key: the order of VALUE_TYPES, then unlisted tags alphabetically."""
    return (list(VALUE_TYPES).index(tag), '') if tag in VALUE_TYPES else (len(VALUE_TYPES), tag)


def case_factors(suite, name):
    """Attributes parsed from a case's suite and name, e.g. ('ops_arithmetic', 'c128/A2_C_layout,S/mul/64')."""
    value_type, operands, op, size = name.split('/')
    kinds = operands.split(',')
    return {
        'suite': suite.removeprefix('ops_'),  # arithmetic, mapped, reductions, math: the operation class
        'op': op,
        'value_type': value_type,
        'N': int(size),
        # short form for the chart titles, e.g. 'M2 sliced, M2 sliced'
        'operands': operands.replace('_C_layout', '').replace('.slice(axis=0,start=0,step=2)', ' sliced').replace(',', ', '),
        'matrix': any(k.startswith(('M2', 'V1')) for k in kinds),
        'scalar': 'S' in kinds,
        'strided': 'slice' in operands,
    }


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
