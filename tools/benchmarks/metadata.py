"""Machine, compiler, source revision, and thread metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import common
import placement

PRESET = 'benchmark_tracked'  # the configure preset build-benchmarks.sh uses


def run_text(cmd, **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw).stdout.strip()


def describe_cpu_ram() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "available_logical_cpus": len(os.sched_getaffinity(0)),
    }
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            info["cpu_model"] = line.split(":", 1)[1].strip()
            break
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            info["ram_bytes"] = int(line.split()[1]) * 1024
            break
    # lscpu adds topology and cache sizes, which matter for interpreting the results.
    wanted = {"Socket(s)": ("sockets", int), "Core(s) per socket": ("cores_per_socket", int),
              "Thread(s) per core": ("threads_per_core", int), "NUMA node(s)": ("numa_nodes", int),
              "CPU max MHz": ("cpu_max_mhz", float)}
    for line in run_text(["lscpu"], env=os.environ | {'LC_ALL': 'C'}).splitlines():
        key, _, value = line.partition(":")
        if key.strip() in wanted:
            field, cast = wanted[key.strip()]
            info[field] = cast(value)
    # The summary above gives each cache level summed over all cores; --caches gives one instance.
    for line in run_text(["lscpu", "--bytes", "--caches=NAME,ONE-SIZE,COHERENCY-SIZE"]).splitlines()[1:]:
        name, size, line_size = line.split()
        info[f"{name.lower()}_cache_bytes"] = int(size)
        info["cache_line_bytes"] = int(line_size)
    return info


def read_cmake_cache(build_dir: Path) -> dict:
    text = (build_dir / "CMakeCache.txt").read_text(errors="replace")
    return dict(re.findall(r"^([A-Za-z0-9_\-]+):[A-Z]+=(.*)$", text, re.MULTILINE))


def describe_compiler(build_dir: Path, cache: dict) -> dict:
    # CMAKE_CXX_FLAGS holds only the preset's flags: -O3 -DNDEBUG lives in
    # CMAKE_CXX_FLAGS_RELEASE and -g in CMAKE_CXX_FLAGS_DEBUG. Recording only the former
    # would make a Debug and a Release build indistinguishable.
    build_type = cache["CMAKE_BUILD_TYPE"]
    base, config = cache["CMAKE_CXX_FLAGS"], cache[f"CMAKE_CXX_FLAGS_{build_type.upper()}"]
    # CMake records the vendor and version it detected; trust that over reparsing.
    detected = sorted(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))[0].read_text()
    return {
        "build_type": build_type,
        "cxx_flags": base,
        "cxx_flags_config": config,
        "effective_flags": " ".join(f for f in (base, config) if f),
        "path": cache["CMAKE_CXX_COMPILER"],
        "vendor": re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]*)"\)', detected)[1],
        "version": re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]*)"\)', detected)[1],
        "banner": run_text([cache["CMAKE_CXX_COMPILER"], "--version"]).splitlines()[0],
    }


def describe_provenance(repo_root: Path, build_dir: Path) -> dict:
    def git(*args, cwd=repo_root):
        return run_text(["git", *args], cwd=cwd)

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dependency_shas": {src.name.removesuffix("_src"): git("rev-parse", "HEAD", cwd=src)
                            for src in sorted((build_dir / "deps").glob("*_src"))},
    }


def thread_settings() -> dict:
    return {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TBLIS_NUM_THREADS")}


def describe_build(build):
    cache = read_cmake_cache(build)
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
        'preset': read_preset(source),
    }


def read_preset(source: Path) -> dict:
    """Cache variables of the benchmark_tracked preset in source/CMakePresets.json, macros unexpanded."""
    presets = json.loads((source / 'CMakePresets.json').read_text())['configurePresets']
    preset = next(p for p in presets if p['name'] == PRESET)
    if 'inherits' in preset:  # the inherited variables would silently go uncompared
        raise ValueError(f'{PRESET} inherits other presets; read_preset only reads its own cacheVariables')
    return preset['cacheVariables']


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
