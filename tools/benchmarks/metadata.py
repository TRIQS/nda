"""Machine, compiler, source revision, and thread metadata."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path

import common
import placement

_SIZE_SUFFIX = {"K": 2**10, "KI": 2**10, "M": 2**20, "MI": 2**20, "G": 2**30, "GI": 2**30}


def count_cpu_list(text: str) -> int:
    """Count CPUs in a sysfs list such as "0-3" or "0,64"."""
    total = 0
    for part in (text or "").split(","):
        if "-" in part:
            lo, _, hi = part.partition("-")
            total += int(hi) - int(lo) + 1
        elif part.strip():
            total += 1
    return total


def read_sysfs_caches(info: dict) -> bool:
    """Per-core cache sizes from sysfs. lscpu reports the machine-wide total for each
    level, so its L1d on a 64-core host is 64 times the figure macOS reports; sysfs
    gives the per-core size directly and keeps the two platforms comparable."""
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    if not root.is_dir():
        return False
    fields = {("1", "Data"): "l1d_cache_bytes", ("1", "Instruction"): "l1i_cache_bytes",
              ("2", "Unified"): "l2_cache_bytes", ("3", "Unified"): "l3_cache_bytes"}
    found = False
    for index in sorted(root.glob("index*")):
        try:
            level = (index / "level").read_text().strip()
            kind = (index / "type").read_text().strip()
            size = to_bytes((index / "size").read_text().strip())
        except OSError:
            continue
        field = fields.get((level, kind))
        if not field or not size:
            continue
        info[field] = size
        found = True
        if field == "l2_cache_bytes":
            shared = (index / "shared_cpu_list")
            if shared.exists():
                info["cpus_per_l2"] = count_cpu_list(shared.read_text().strip())
        line = index / "coherency_line_size"
        if line.exists() and "cache_line_bytes" not in info:
            info["cache_line_bytes"] = int(line.read_text().strip())
    return found


def to_bytes(text: str) -> int | None:
    """Parse a size such as "512 KiB" or "12 MiB", as lscpu reports caches."""
    match = re.match(r"^\s*([\d.]+)\s*([KMG]i?)?B?\s*$", text or "", re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return int(value * _SIZE_SUFFIX.get((match.group(2) or "").upper(), 1))


def run_text(cmd, **kw) -> str:
    """Capture stdout, returning "" on any failure. For best-effort probes only."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def describe_cpu_ram() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "available_logical_cpus": os.cpu_count(),
    }

    if platform.system() == "Linux":
        info['available_logical_cpus'] = len(os.sched_getaffinity(0))
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_bytes"] = int(line.split()[1]) * 1024
                break
        # lscpu adds topology and cache sizes, which matter for interpreting the sweep.
        wanted = {
            "Socket(s)": "sockets",
            "Core(s) per socket": "cores_per_socket",
            "Thread(s) per core": "threads_per_core",
            "NUMA node(s)": "numa_nodes",
            "CPU max MHz": "cpu_max_mhz",
        }
        if not read_sysfs_caches(info):
            wanted.update({"L1d cache": "l1d_cache_bytes", "L1i cache": "l1i_cache_bytes",
                           "L2 cache": "l2_cache_bytes", "L3 cache": "l3_cache_bytes"})
        for line in run_text(["lscpu"], env=os.environ | {'LC_ALL': 'C'}).splitlines():
            key, _, value = line.partition(":")
            field = wanted.get(key.strip())
            if not field:
                continue
            try:
                info[field] = (to_bytes(value) if field.endswith('_bytes') else
                               float(value) if field == 'cpu_max_mhz' else int(value))
            except ValueError:
                pass
    elif platform.system() == "Darwin":
        for key, field, cast in [
            ("machdep.cpu.brand_string", "cpu_model", str),
            ("hw.memsize", "ram_bytes", int),
            ("hw.physicalcpu", "physical_cpus", int),
            ("hw.cachelinesize", "cache_line_bytes", int),
        ]:
            value = run_text(["sysctl", "-n", key])
            if value:
                try:
                    info[field] = cast(value)
                except ValueError:
                    pass

        # Report the performance cores. On Apple Silicon the bare hw.l1dcachesize and
        # hw.l2cachesize sysctls give the efficiency-core caches, but benchmarks run at
        # default quality of service on the performance cores. hw.perflevel0 is the
        # fastest core type; the fallback covers homogeneous Macs, which have no perflevels.
        for key, field in (("l1dcachesize", "l1d_cache_bytes"),
                           ("l1icachesize", "l1i_cache_bytes"),
                           ("l2cachesize", "l2_cache_bytes"),
                           ("l3cachesize", "l3_cache_bytes"),
                           ("cpusperl2", "cpus_per_l2")):
            value = (run_text(["sysctl", "-n", f"hw.perflevel0.{key}"])
                     or run_text(["sysctl", "-n", f"hw.{key}"]))
            if value:
                info[field] = int(value)

    return info


def read_cmake_cache(build_dir: Path) -> dict:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return {}
    entries = {}
    for line in cache.read_text(errors="replace").splitlines():
        match = re.match(r"^([A-Za-z0-9_\-]+):[A-Z]+=(.*)$", line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def describe_compiler(build_dir: Path, cache: dict) -> dict:
    # CMAKE_CXX_FLAGS is empty in a default nda build: -O3 -DNDEBUG lives in
    # CMAKE_CXX_FLAGS_RELEASE and -g in CMAKE_CXX_FLAGS_DEBUG. Recording only the former
    # would make a Debug and a Release build indistinguishable.
    build_type = cache.get("CMAKE_BUILD_TYPE") or ""
    base = cache.get("CMAKE_CXX_FLAGS") or ""
    config = cache.get(f"CMAKE_CXX_FLAGS_{build_type.upper()}") or "" if build_type else ""
    info = {
        "build_type": build_type or None,
        "cxx_flags": base,
        "cxx_flags_config": config,
        "effective_flags": " ".join(f for f in (base, config) if f),
        "path": cache.get("CMAKE_CXX_COMPILER"),
    }
    # CMake records the vendor and version it detected; trust that over reparsing.
    for path in sorted(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")):
        text = path.read_text(errors="replace")
        for key, field in [("CMAKE_CXX_COMPILER_ID", "vendor"),
                           ("CMAKE_CXX_COMPILER_VERSION", "version")]:
            match = re.search(rf'set\({key} "([^"]*)"\)', text)
            if match:
                info[field] = match.group(1)
        break
    if info["path"] and Path(info["path"]).exists():
        banner = run_text([info["path"], "--version"])
        if banner:
            info["banner"] = banner.splitlines()[0]
    return info


def describe_provenance(repo_root: Path, build_dir: Path) -> dict:
    def git(*args, cwd=repo_root):
        return run_text(["git", *args], cwd=str(cwd)) or None

    info = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    deps = {}
    for src in sorted((build_dir / "deps").glob("*_src")):
        if (sha := git("rev-parse", "HEAD", cwd=src)):
            deps[src.name.removesuffix("_src")] = sha
    info["dependency_shas"] = deps
    return info


def thread_settings() -> dict:
    return {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TBLIS_NUM_THREADS")}


def describe_build(build):
    cache = read_cmake_cache(build)
    if not cache.get('CMAKE_HOME_DIRECTORY'):
        raise ValueError(f'No CMake source directory recorded in {build}')
    source = Path(cache['CMAKE_HOME_DIRECTORY'])
    harness = sorted((source / 'benchmarks/tracked').glob('*.hpp'))
    harness += sorted((source / 'benchmarks/tracked').glob('*.cpp'))
    harness += [source / 'benchmarks/CMakeLists.txt']
    digest = hashlib.sha256()
    for path in harness:
        digest.update(str(path.relative_to(source)).encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return {
        'build': str(build), 'bindir': str(build / 'benchmarks/tracked'),
        'compiler': describe_compiler(build, cache),
        'provenance': describe_provenance(source, build),
        'harness_sha256': digest.hexdigest(),
    }


def create_metadata(settings: dict) -> dict:
    return {
        'started_at': common.timestamp(), 'finished_at': None,
        'settings': dict(settings),
        'ci': {'url': os.environ.get('BUILD_URL'), 'repository': os.environ.get('GIT_URL'),
               'pr': os.environ.get('CHANGE_ID'), 'baseline_branch': os.environ.get('CHANGE_TARGET'),
               'candidate_branch': os.environ.get('CHANGE_BRANCH')},
        'machine': describe_cpu_ram(), 'threads': thread_settings(), 'benchmark_context': None, 'builds': {},
        'totals': {'families': 0, 'cases': 0, 'binaries': 0, 'wall_seconds': None},
        'binaries': [],
    }


def record_placements(document, workers):
    pinned = all(w['pin_command'] for w in workers)
    # Unpinned workers have no placement to describe; listing one null entry per worker says nothing.
    document['settings'].update(workers=len(workers), pinned=pinned,
                               worker_placements=[placement.public_metadata(w) for w in workers] if pinned else [])


def record_context(document, result):
    """Store the Google Benchmark context once per run and drop it from every invocation.

    date and load_avg change every launch and executable names the binary, so they are dropped.
    """
    context = result.get('benchmark', {}).pop('context', None)
    if context is None or document['benchmark_context'] is not None:
        return False
    document['benchmark_context'] = {key: value for key, value in context.items()
                                     if key not in ('date', 'load_avg', 'executable')}
    return True
