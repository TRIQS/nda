#!/usr/bin/env python3
"""Pilot runner for the tracked nda benchmarks.

Runs every ops_* binary, writes one Google Benchmark JSON per binary, and records
wall time, case counts, build provenance and machine description in summary.json.
The purpose is to find out what a full sweep costs before wiring it into Jenkins.

Workers automatically use distinct allowed physical cores with local NUMA memory
binding on Linux with numactl, including when --workers 1 is specified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --benchmark_min_time is the main cost lever: it sets the target time per case, and
# Google Benchmark's iteration-count search scales with it. Unset means the GB default
# of roughly 0.7s per case.

UNIT_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}

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


# machine description


def scaling_governor(cpu: int | None) -> str | None:
    if cpu is None:
        return None
    try:
        return Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor").read_text().strip() or None
    except OSError:
        return None


def describe_cpu_ram() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "logical_cpus": os.cpu_count(),
    }

    if platform.system() == "Linux":
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
        for line in run_text(["lscpu"]).splitlines():
            key, _, value = line.partition(":")
            field = wanted.get(key.strip())
            if not field:
                continue
            # lscpu prints caches as "512 KiB"; store bytes so the column is numeric.
            info[field] = to_bytes(value) if field.endswith("_bytes") else value.strip()
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

    if ram := info.get("ram_bytes"):
        info["ram_gib"] = round(ram / 2**30, 1)
    return info


# build provenance


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
        # Tree hash of the benchmark sources: changes exactly when a definition changes,
        # so comparisons across a boundary can be rejected from the data alone.
        "suite_tree_sha": git("rev-parse", "HEAD:benchmarks/tracked"),
    }
    deps = {}
    for src in sorted((build_dir / "deps").glob("*_src")):
        if (sha := git("rev-parse", "HEAD", cwd=src)):
            deps[src.name.removesuffix("_src")] = sha
    info["dependency_shas"] = deps
    return info


# pinning


def worker_placements(workers: int) -> list[dict]:
    linux = platform.system() == "Linux"
    numa = linux and shutil.which("numactl")
    cpu_only = linux and shutil.which("taskset")
    if not numa and not cpu_only:
        if workers > 1:
            raise ValueError("Parallel measurement requires Linux and numactl or taskset")
        print("warning: CPU pinning is unavailable; running unpinned", file=sys.stderr)
        return [{"cpu": None, "numa_node": None, "pin_command": []}]
    if not numa:
        print("warning: numactl is unavailable; pinning CPUs without memory binding", file=sys.stderr)
    status = Path("/proc/self/status").read_text()
    allowed = re.search(r"^Mems_allowed_list:\s*(.+)$", status, re.MULTILINE)
    if not allowed:
        raise ValueError("Cannot determine allowed NUMA memory nodes")
    memory_nodes = set()
    for part in allowed[1].split(","):
        lo, _, hi = part.strip().partition("-")
        memory_nodes.update(range(int(lo), int(hi or lo) + 1))

    by_node, seen = {}, set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        core = tuple(int((root / "topology" / key).read_text())
                     for key in ("physical_package_id", "core_id"))
        nodes = sorted(root.glob("node[0-9]*"))
        if len(nodes) != 1:
            raise ValueError(f"Cannot determine NUMA node for CPU {cpu}")
        node = int(nodes[0].name[4:])
        if core in seen or node not in memory_nodes:
            continue
        seen.add(core)
        by_node.setdefault(node, []).append(cpu)

    # Round-robin nodes; never assign two SMT siblings to different workers.
    placements = []
    while len(placements) < workers:
        for node in sorted(by_node):
            if by_node[node] and len(placements) < workers:
                cpu = by_node[node].pop(0)
                command = (["numactl", f"--physcpubind={cpu}", f"--membind={node}"]
                           if numa else ["taskset", "-c", str(cpu)])
                placements.append({"cpu": cpu, "numa_node": node if numa else None, "pin_command": command})
        if not any(by_node.values()) and len(placements) < workers:
            raise ValueError(f"Need {workers} distinct allowed physical cores with local memory")
    return placements


# counting


def count_cases(binary: Path, gb_filter: list[str]) -> tuple[int, int, int]:
    """Ask the binary's own registry, so counts cannot drift from what runs.

    A registered name looks like  f64/A2_C_layout,A2_C_layout/add/64
    Stripping the trailing /<N> collapses the size sweep into one family.
    """
    listing = subprocess.run([str(binary), "--benchmark_list_tests=true", *gb_filter],
                             capture_output=True, text=True, check=True).stdout
    names = [line for line in listing.splitlines() if line.strip()]
    families = {re.sub(r"/\d+$", "", name) for name in names}
    ops = {family.rsplit("/", 1)[-1] for family in families}
    return len(names), len(families), len(ops)


def validate_result(path: Path, cases: int, repetitions: int) -> None:
    result = json.loads(path.read_text())
    counts = {}
    for row in result["benchmarks"]:
        if row.get("error_occurred"):
            raise ValueError(f"Benchmark reported an error: {row}")
        if row.get("run_type") != "iteration":
            continue
        if row["time_unit"] not in UNIT_SECONDS or not all(
                math.isfinite(row[key]) and row[key] > 0 for key in ("cpu_time", "real_time")):
            raise ValueError(f"Invalid benchmark timing: {row}")
        name = row.get("run_name", row["name"])
        repetition = row.get("repetition_index", 0)
        observed = counts.setdefault(name, set())
        if repetition in observed or repetition not in range(repetitions):
            raise ValueError(f"Invalid or duplicate repetition: {row}")
        observed.add(repetition)
    if cases == 0 or len(counts) != cases or any(len(indices) != repetitions for indices in counts.values()):
        raise ValueError(f"Expected {cases} nonempty cases with {repetitions} repetitions in {path.name}")


def measured_seconds(path: Path) -> float:
    """Sum GB's own timed-loop totals, skipping _mean/_stddev aggregate rows."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0.0
    total = 0.0
    for bench in doc.get("benchmarks", []):
        if bench.get("run_type") != "iteration":
            continue
        total += bench["real_time"] * bench["iterations"] * UNIT_SECONDS.get(
            bench.get("time_unit", "ns"), 1e-9)
    return total


# main


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bindir", type=Path,
                        default=repo_root / "build/benchmarks/tracked",
                        help="directory holding the built ops_* binaries")
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
    parser.add_argument("--count-only", action="store_true",
                        help="report counts without running anything")
    args = parser.parse_args()
    if args.workers < 1 or args.repetitions < 1:
        parser.error("workers and repetitions must be positive")
    args.bindir = args.bindir.resolve()

    if args.outdir is None:
        args.outdir = repo_root / "bench-results" / time.strftime("%Y%m%dT%H%M%SZ",
                                                                  time.gmtime())
    if not args.bindir.is_dir():
        sys.exit(f"error: bindir not found: {args.bindir}\n"
                 "enable Build_Benchs and build the tracked ops_* targets first")

    binaries = sorted(p for p in args.bindir.glob("ops_*") if p.is_file()
                      and os.access(p, os.X_OK))
    if not binaries:
        sys.exit(f"error: no ops_* binaries in {args.bindir}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    threads = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TBLIS_NUM_THREADS")}

    gb_filter = [f"--benchmark_filter={args.filter}"] if args.filter else []
    gb_min_time = [f"--benchmark_min_time={args.min_time}"] if args.min_time else []
    workers = min(args.workers, len(binaries))
    try:
        placements = worker_placements(workers)
        for placement in placements:
            placement["scaling_governor"] = scaling_governor(placement["cpu"])
        if not args.count_only:
            for placement in placements:
                if placement["pin_command"]:
                    subprocess.run([*placement["pin_command"], sys.executable, "-c",
                                    f"import os; assert os.sched_getaffinity(0) == {{{placement['cpu']}}}"], check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"CPU/NUMA binding failed: {exc}")

    build_dir = args.bindir.parents[1]
    cache = read_cmake_cache(build_dir)

    print(f"bindir       {args.bindir}")
    print(f"outdir       {args.outdir}")
    print(f"repetitions  {args.repetitions}")
    print(f"min-time     {args.min_time or '<gb default>'}")
    print(f"filter       {args.filter or '<none>'}")
    print(f"workers      {workers}")
    for i, placement in enumerate(placements):
        print(f"worker {i}     {' '.join(placement['pin_command']) or '<none>'}")
    print(f"binaries     {len(binaries)}\n")

    entries, totals = [], {"families": 0, "cases": 0}
    for binary in binaries:
        cases, families, ops = count_cases(binary, gb_filter)
        totals["cases"] += cases
        totals["families"] += families

        placement = placements[len(entries) % workers]
        entry = {"name": binary.name, "families": families, "cases": cases, "ops": ops, **placement}

        if args.count_only:
            print(f"{binary.name:<22} {families:5d} families  {cases:5d} cases  (not run)")
            entries.append(entry | {"wall_seconds": None, "exit_code": None, "json": None})
            continue

        entries.append(entry)

    def run_binary(entry):
        binary = args.bindir / entry["name"]
        print(f"{binary.name:<22} {entry['families']:5d} families  {entry['cases']:5d} cases  started", flush=True)

        out_json = args.outdir / f"{binary.name}.json"
        command = [*entry["pin_command"], str(binary),
                   f"--benchmark_out={out_json}",
                   "--benchmark_out_format=json",
                   f"--benchmark_repetitions={args.repetitions}",
                   *gb_min_time, *gb_filter]

        start = time.perf_counter()
        with open(args.outdir / f"{binary.name}.log", "wb") as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                if code == 0:
                    validate_result(out_json, entry["cases"], args.repetitions)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log.write(str(exc).encode())
                code = 1
        secs = round(time.perf_counter() - start, 3)

        print(f"{binary.name:<22} {secs:8.1f}s" + ("" if code == 0
                                 else f"  FAILED (exit {code}, see {binary.name}.log)"))
        entry.update(wall_seconds=secs, exit_code=code, json=out_json.name)

    def run_worker(entries):
        for entry in entries:
            run_binary(entry)

    suite_start = time.perf_counter()
    if not args.count_only:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(run_worker, [entries[i::workers] for i in range(workers)]))

    total_secs = round(time.perf_counter() - suite_start, 3)

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "repetitions": args.repetitions,
            "min_time": args.min_time,
            "filter": args.filter,
            "pinned": all(p["pin_command"] for p in placements),
            "workers": workers,
            "worker_placements": [{k: v for k, v in p.items() if k != "pin_command"} for p in placements],
            "bindir": str(args.bindir),
        },
        "machine": describe_cpu_ram(),
        "compiler": describe_compiler(build_dir, cache),
        "provenance": describe_provenance(repo_root, build_dir),
        "threads": threads,
        "totals": totals | {"binaries": len(binaries), "wall_seconds": total_secs},
        "binaries": [{k: v for k, v in entry.items() if k != "pin_command"} for entry in entries],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nfamilies   {totals['families']}   (distinct benchmarks, size sweep collapsed)")
    print(f"cases      {totals['cases']}   (registered cases, one per size)")
    print(f"wall       {total_secs}s at {args.repetitions} repetition(s)")
    print(f"results    {args.outdir}")

    if args.count_only:
        return 0

    # GB reports only its final timed run per case; the gap to the wall clock is mostly
    # GB's own iteration-count search (it tries 1, 10, 100 ... and discards each trial),
    # plus input generation, which only matters at the largest size.
    measured = sum(measured_seconds(args.outdir / e["json"])
                   for e in entries if e["json"] and e["exit_code"] == 0)
    if workers > 1:
        print(f"measured   {measured:.1f}s summed across parallel workers (not a fraction of wall time)")
    elif measured and total_secs:
        print(f"measured   {measured:.1f}s in timed loops "
              f"({100 * measured / total_secs:.0f}% of wall; the rest is GB's "
              f"iteration search and input generation)")

    return 0 if all(e["exit_code"] in (0, None) for e in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
