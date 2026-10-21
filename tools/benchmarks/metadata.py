"""Machine and build metadata."""

import hashlib
import json
import os
import platform
import re
import shlex
from pathlib import Path

import archspec.cpu

import common
import stats


def describe_cpu_ram() -> dict:
    host = archspec.cpu.host()
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "hostname": platform.node(),
        "microarchitecture": {"name": host.name, "level": host.generic.name.replace("_", "-")},
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    }
    # lscpu gives the model, topology and cache sizes, which matter for interpreting the results.
    wanted = {"Model name": ("cpu_model", str), "Socket(s)": ("sockets", int), "Core(s) per socket": ("cores_per_socket", int),
              "Thread(s) per core": ("threads_per_core", int), "NUMA node(s)": ("numa_nodes", int)}
    for line in common.run(["lscpu"], env=os.environ | {'LC_ALL': 'C'}).splitlines():
        key, _, value = line.partition(":")
        if key.strip() in wanted:
            field, cast = wanted[key.strip()]
            info[field] = cast(value.strip())
    # The summary above gives each cache level summed over all cores; --caches gives one instance.
    for line in common.run(["lscpu", "--bytes", "--caches=NAME,ONE-SIZE"]).splitlines()[1:]:
        name, size = line.split()
        info[f"{name.lower()}_cache_bytes"] = int(size)
    # 'performance' keeps the clock up; without a cpufreq driver (a virtual machine) there is none to read.
    governors = (Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor") for cpu in os.sched_getaffinity(0))
    info["scaling_governors"] = sorted({path.read_text().strip() for path in governors if path.exists()})
    return info


def read_cmake_cache(build_dir: Path) -> dict:
    text = (build_dir / "CMakeCache.txt").read_text(errors="replace")
    return dict(re.findall(r"^([A-Za-z0-9_\-]+):[A-Z]+=(.*)$", text, re.MULTILINE))


# Defines that carry no performance information: build-system housekeeping and the embedded git hashes.
HOUSEKEEPING_DEFINES = re.compile(r"^-D(\w+_GIT_HASH=|BENCHMARK_STATIC_DEFINE|MPICH_SKIP_MPICXX|OMPI_SKIP_MPICXX|_MPICC_H)")


def tracked_compile_flags(build_dir: Path) -> tuple[list, list]:
    """(flags, defines) one tracked benchmark was compiled with, from compile_commands.json (needs
    CMAKE_EXPORT_COMPILE_COMMANDS=ON, which build-benchmarks.sh passes).

    Unlike the cache variables this includes what targets add (-std=, -fopenmp, feature defines such as
    NDA_HAVE_OPENMP). Warning switches, include paths, dependency-file and output arguments are dropped.
    """
    entries = json.loads((build_dir / "compile_commands.json").read_text())
    command = next((e["command"] for e in entries if "/benchmarks/tracked/" in e["file"]), None)
    if command is None:  # a revision without tracked benchmarks, such as a baseline that predates them
        return [], []
    # Dropped: warnings (-W...), include paths (-I..., -isystem), dependency-file generation (-MD, -MF ...),
    # the output/source file names, coloured diagnostics, and the housekeeping defines above.
    # Kept: everything else, so -O, -march, -std, -f<codegen> flags and the remaining -D defines.
    takes_argument = ("-o", "-c", "-I", "-isystem", "-include", "-MF", "-MT", "-MQ")
    dropped_prefixes = ("-W", "-I", "-isystem", "-M", "-fdiagnostics")
    flags, defines, skip = [], [], False
    for arg in shlex.split(command)[1:]:
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
    return flags, sorted(defines)


def describe_provenance(repo_root: Path, build_dir: Path) -> dict:
    # Sources fetched by deps/CMakeLists.txt (deps/<name>_src) and by FetchContent (_deps/<name>-src).
    sources = sorted((build_dir / "deps").glob("*_src")) + sorted((build_dir / "_deps").glob("*-src"))
    dependencies = {}
    for src in sources:
        name = src.name.removesuffix("_src").removesuffix("-src")
        dependencies[name] = {"commit": common.run(["git", "rev-parse", "HEAD"], cwd=src),
                              "url": common.run(["git", "remote", "get-url", "origin"], cwd=src)}
    return {
        "commit": common.run(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "branch": common.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "dependencies": dependencies,
    }


def harness_files(source: Path) -> dict:
    """Short sha256 of each file of the benchmark harness: benchmarks/tracked/*.hpp, *.cpp and benchmarks/CMakeLists.txt."""
    paths = sorted((source / 'benchmarks/tracked').glob('*.[ch]pp')) + [source / 'benchmarks/CMakeLists.txt']
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()[:12] for path in paths}


def describe_build(build):
    cache = read_cmake_cache(build)
    source = Path(cache['CMAKE_HOME_DIRECTORY'])
    flags, defines = tracked_compile_flags(build)
    return {
        'compiler': common.run([cache["CMAKE_CXX_COMPILER"], "--version"]).splitlines()[0],
        'flags': flags,  # what the compiler received, from compile_commands.json
        'defines': defines,
        'provenance': describe_provenance(source, build),
        'harness_files': harness_files(source),
    }


def build_differences(baseline, candidate):
    """{what: how it differs} between two build descriptions. Each side builds its own benchmarks with its own
    benchmark_tracked preset and dependency pins, so a changed flag, benchmark source or dependency is part of what
    the comparison measures."""
    differences = {}
    if baseline['compiler'] != candidate['compiler']:
        differences['compiler'] = f"baseline {baseline['compiler']}; candidate {candidate['compiler']}"
    for key in ('flags', 'defines'):
        only_baseline = sorted(set(baseline[key]) - set(candidate[key]))
        only_candidate = sorted(set(candidate[key]) - set(baseline[key]))
        if only_baseline or only_candidate:
            differences[key] = (f"only in baseline: {' '.join(only_baseline) or 'none'}; "
                                f"only in candidate: {' '.join(only_candidate) or 'none'}")
    files = baseline['harness_files'], candidate['harness_files']
    if changed := sorted(f for f in files[0].keys() | files[1].keys() if files[0].get(f) != files[1].get(f)):
        differences['benchmark sources'] = 'changed: ' + ', '.join(changed)
    # A changed pin in deps/CMakeLists.txt, or a dependency only one side fetches.
    deps = [{name: dep['commit'] for name, dep in side['provenance']['dependencies'].items()} for side in (baseline, candidate)]
    if changed := sorted(name for name in deps[0].keys() | deps[1].keys() if deps[0].get(name) != deps[1].get(name)):
        short = common.SHORT_SHA
        differences['dependencies'] = ', '.join(f"{name} {deps[0].get(name, 'none')[:short]} -> {deps[1].get(name, 'none')[:short]}"
                                                for name in changed)
    return differences


def create_metadata(**settings) -> dict:
    """The document a run records its results in, with the measurement settings and the machine filled in."""
    return {
        'started_at': common.timestamp(), 'finished_at': None,
        'settings': {'min_time': common.MIN_TIME, 'min_warmup_time': common.MIN_WARMUP_TIME, 'alpha': stats.ALPHA,
                     'min_improvement_percent': stats.MIN_IMPROVEMENT, 'max_noise_percent': stats.MAX_NOISE,
                     'outlier_sigma': stats.OUTLIER_SIGMA, **settings},
        # Jenkins PR build: CHANGE_* ; local compare-benchmarks.sh: BASELINE_REF / CANDIDATE_REF.
        'ci': {'url': os.environ.get('BUILD_URL'), 'pr': os.environ.get('CHANGE_ID'),
               'baseline_branch': os.environ.get('CHANGE_TARGET') or os.environ.get('BASELINE_REF'),
               'candidate_branch': os.environ.get('CHANGE_BRANCH') or os.environ.get('CANDIDATE_REF')},
        'machine': describe_cpu_ram(), 'builds': {},
    }
