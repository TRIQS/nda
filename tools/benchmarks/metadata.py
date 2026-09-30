"""Machine and build metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
from pathlib import Path

import archspec.cpu

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
    # /proc/cpuinfo repeats the same block for every logical CPU; the first block is enough.
    first_cpu = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
    fields = dict(re.findall(r"^([^:\n]+?)\s*:\s*(.*)$", first_cpu, re.MULTILINE))
    info["cpu_model"] = fields.get("model name", "unknown")
    info["microarchitecture"] = archspec_microarchitecture()
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


def archspec_microarchitecture() -> dict:
    host = archspec.cpu.host()
    return {"name": host.name, "vendor": host.vendor, "level": host.generic.name.replace("_", "-")}


# Defines that carry no performance information: build-system housekeeping and the embedded git hashes.
HOUSEKEEPING_DEFINES = re.compile(r"^-D(\w+_GIT_HASH=|BENCHMARK_STATIC_DEFINE|MPICH_SKIP_MPICXX|OMPI_SKIP_MPICXX|_MPICC_H)")


def tracked_compile_flags(build_dir: Path) -> dict:
    """The flags one tracked benchmark was compiled with, from compile_commands.json (needs
    CMAKE_EXPORT_COMPILE_COMMANDS=ON, which build-benchmarks.sh passes): {'flags': [...], 'defines': [...]}.

    Unlike the cache variables this includes what targets add (-std=, -fopenmp, feature defines such as
    NDA_HAVE_XSIMD). Warning switches, include paths, dependency-file and output arguments are dropped.
    """
    entries = json.loads((build_dir / "compile_commands.json").read_text())
    args = shlex.split(next(e["command"] for e in entries if "/benchmarks/tracked/" in e["file"]))
    # Dropped: warnings (-W...), include paths (-I..., -isystem), dependency-file generation (-MD, -MF ...),
    # the output/source file names, coloured diagnostics, and the housekeeping defines above.
    # Kept: everything else, so -O, -march, -std, -f<codegen> flags and the remaining -D defines.
    takes_argument = ("-o", "-c", "-I", "-isystem", "-include", "-MF", "-MT", "-MQ")
    dropped_prefixes = ("-W", "-I", "-isystem", "-M", "-fdiagnostics")
    flags, defines, skip = [], [], False
    for arg in args[1:]:
        if skip:
            skip = False
        elif arg in takes_argument:
            skip = True
        elif arg.startswith(dropped_prefixes) or HOUSEKEEPING_DEFINES.match(arg):
            continue
        elif arg.startswith("-D"):
            defines.append(arg[2:])
        else:
            flags.append(arg)
    return {"flags": flags, "defines": sorted(defines)}


def describe_compiler(build_dir: Path, cache: dict) -> dict:
    # CMake records the vendor and version it detected; trust that over reparsing.
    detected = sorted(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))[0].read_text()
    compiled = tracked_compile_flags(build_dir)
    return {
        "build_type": cache["CMAKE_BUILD_TYPE"],
        "effective_flags": " ".join(compiled["flags"]),  # what the compiler received, from compile_commands.json
        "defines": compiled["defines"],
        "path": cache["CMAKE_CXX_COMPILER"],
        "vendor": re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]*)"\)', detected)[1],
        "version": re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]*)"\)', detected)[1],
        "banner": run_text([cache["CMAKE_CXX_COMPILER"], "--version"]).splitlines()[0],
    }


def describe_provenance(repo_root: Path, build_dir: Path) -> dict:
    def git(*args, cwd=repo_root):
        return run_text(["git", *args], cwd=cwd)

    # Sources fetched by deps/CMakeLists.txt (deps/<name>_src) and by FetchContent (_deps/<name>-src).
    sources = sorted((build_dir / "deps").glob("*_src")) + sorted((build_dir / "_deps").glob("*-src"))
    dependencies = {}
    for src in sources:
        name = src.name.removesuffix("_src").removesuffix("-src")
        dependencies[name] = {"commit": git("rev-parse", "HEAD", cwd=src), "url": git("remote", "get-url", "origin", cwd=src)}
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dependencies": dependencies,
    }


# The language runtime that every C++ program links; not worth listing.
TOOLCHAIN_LIBRARIES = re.compile(r"^(libc|libm|libstdc\+\+|libgcc_s|libgomp|libpthread|libdl|librt|ld-linux.*|linux-vdso)\.so")


def linked_libraries(build_dir: Path) -> dict:
    """Shared libraries a tracked benchmark binary is linked against, {soname: resolved path}, from the
    binary's NEEDED entries (readelf) resolved by the dynamic loader (ldd); the toolchain runtime is left out."""
    binaries = sorted(p for p in (build_dir / "benchmarks/tracked").glob("*") if p.is_file() and os.access(p, os.X_OK))
    if not binaries:
        return {}
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", run_text(["readelf", "-d", str(binaries[0])]))
    resolved = dict(re.findall(r"^\s*(\S+) => (\S+)", run_text(["ldd", str(binaries[0])]), re.MULTILINE))
    return {name: resolved[name] for name in needed if not TOOLCHAIN_LIBRARIES.match(name)}


def describe_build(build):
    cache = read_cmake_cache(build)
    source = Path(cache['CMAKE_HOME_DIRECTORY'])
    harness = sorted((source / 'benchmarks/tracked').glob('*.hpp'))
    harness += sorted((source / 'benchmarks/tracked').glob('*.cpp'))
    harness += [source / 'benchmarks/CMakeLists.txt']
    digest = hashlib.sha256()
    files = {}
    for path in harness:
        digest.update(str(path.relative_to(source)).encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
        files[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return {
        'build': str(build), 'bindir': str(build / 'benchmarks/tracked'),
        'compiler': describe_compiler(build, cache),
        'provenance': describe_provenance(source, build),
        'linked_libraries': linked_libraries(build),
        'harness_sha256': digest.hexdigest(),
        'harness_files': files,  # per-file digests, to name what differs between two builds
        'preset': read_preset(source),
    }


def read_preset(source: Path) -> dict:
    """Cache variables of the benchmark_tracked preset in source/CMakePresets.json, macros unexpanded."""
    presets = json.loads((source / 'CMakePresets.json').read_text())['configurePresets']
    return next(p for p in presets if p['name'] == PRESET)['cacheVariables']


def create_metadata(settings: dict) -> dict:
    return {
        'started_at': common.timestamp(), 'finished_at': None,
        'settings': dict(settings),
        # Jenkins PR build: CHANGE_* ; local compare-benchmarks.sh: BASELINE_REF / CANDIDATE_REF.
        'ci': {'url': os.environ.get('BUILD_URL'), 'pr': os.environ.get('CHANGE_ID'),
               'baseline_branch': os.environ.get('CHANGE_TARGET') or os.environ.get('BASELINE_REF'),
               'candidate_branch': os.environ.get('CHANGE_BRANCH') or os.environ.get('CANDIDATE_REF')},
        'machine': describe_cpu_ram(), 'benchmark_context': None, 'builds': {},
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
