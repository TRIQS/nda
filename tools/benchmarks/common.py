"""Running Google Benchmark binaries pinned to CPUs, and reading case names."""

from datetime import datetime, timezone
import json
import os
import re
import shlex
import subprocess
import threading
from pathlib import Path

MIN_TIME = '0.2s'      # each launch times its case for at least this long
MIN_WARMUP_TIME = 0.1  # seconds of warm-up each launch runs and discards first
NANOSECONDS = {"ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}
NS_PER_US = NANOSECONDS["us"]
SHORT_SHA = 10         # hex digits of a commit hash shown in logs, reports and chart titles


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, **kw):
    """The stripped standard output of command; raises subprocess.CalledProcessError when it fails."""
    return subprocess.run(command, capture_output=True, text=True, check=True, **kw).stdout.strip()


def find_binaries(bindir):
    return sorted(path for path in bindir.glob("*") if path.is_file() and os.access(path, os.X_OK))


def discover_cases(bindir, pattern=None):
    """{(binary name, case name): binary} for every case of every benchmark binary in bindir."""
    cases = {}
    for binary in find_binaries(bindir):
        command = [str(binary), "--benchmark_list_tests=true"]
        if pattern:
            command.append(f"--benchmark_filter={pattern}")
        cases.update({(binary.name, name.strip()): binary for name in run(command).splitlines() if name.strip()})
    return cases


def worker_cpus(count=None, *, pin=True):
    """Select workers on distinct allowed physical cores, or return None entries for unpinned workers."""
    if not pin:
        return [None] * (count or len(os.sched_getaffinity(0)))
    cpus, cores = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        core = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
        if core not in cores:
            cores.add(core)
            cpus.append(cpu)
    return cpus[:count]


def output_logger(path):
    """A function that appends one labelled section to path (emptied first); safe to call from parallel workers."""
    path.write_text('')
    lock = threading.Lock()

    def append(label, text):
        with lock, path.open('a', encoding='utf-8') as stream:
            stream.write(f'=== {label} ===\n{text.rstrip()}\n\n')

    return append


def measure(binary, name, cpu, log, label):
    """Launch binary to run the case name alone, pinned to cpu (None: unpinned), and return its CPU time per
    iteration in ns. The command and its console output go to log under label; a failed launch raises RuntimeError."""
    pin = [] if cpu is None else ['numactl', f'--physcpubind={cpu}', '--localalloc']
    command = [*pin, str(binary), f'--benchmark_filter=^{re.escape(name)}$', f'--benchmark_min_time={MIN_TIME}',
               f'--benchmark_min_warmup_time={MIN_WARMUP_TIME:g}', '--benchmark_format=json']
    process = subprocess.run(command, capture_output=True, text=True, errors='replace')
    log(f'{binary.name}: {name}: {label}', f'Command: {shlex.join(command)}\n{process.stderr}{process.stdout}')
    if process.returncode:
        raise RuntimeError(f'{binary.name} {name} exited with {process.returncode}; see output.log')
    row = json.loads(process.stdout)['benchmarks'][0]
    if row.get('error_occurred'):
        raise RuntimeError(f'{binary.name} {name} reported an error: {row.get("error_message")}')
    return row['cpu_time'] * NANOSECONDS[row['time_unit']]


# The C++ type behind each value-type tag of nda::bench::type_tag (benchmarks/tracked/bench_inputs.hpp), in the order
# reports and charts use.
VALUE_TYPES = {'f64': 'double', 'c128': 'complex<double>', 'f32': 'float', 'c64': 'complex<float>'}


def value_type_label(tag):
    return f'{VALUE_TYPES[tag]} ({tag})'


def value_type_key(tag):
    return list(VALUE_TYPES).index(tag)


def case_factors(suite, name):
    """Attributes parsed from a case's suite and name, e.g. ('ops_arithmetic', 'c128/A2_C_layout,S/mul/64')."""
    value_type, operands, op, size = name.split('/')
    kinds = operands.split(',')
    return {
        'suite': suite.removeprefix('ops_'),  # arithmetic, mapped, math, reductions, tensor: the operation class
        'op': op,
        'value_type': value_type,
        'N': int(size),
        # short form for the chart titles, e.g. 'M2 sliced, M2 sliced'
        'operands': operands.replace('_C_layout', '').replace('.slice(axis=0,start=0,step=2)', ' sliced').replace(',', ', '),
        'matrix': any(k.startswith(('M2', 'V1')) for k in kinds),
        'scalar': 'S' in kinds,
        'strided': 'slice' in operands,
    }


def side_label(results, side, commit_text):
    """'<branch> (<commit_text>)' for the baseline or candidate side of a comparison, or commit_text alone when the
    branch is unknown (a detached checkout outside CI)."""
    branch = results['ci'][f'{side}_branch'] or results['builds'][side]['provenance']['branch']
    return commit_text if branch == 'HEAD' else f'{branch} ({commit_text})'
