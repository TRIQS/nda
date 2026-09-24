"""Automatic worker CPU selection, NUMA binding, and binding validation."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def scaling_governor(cpu: int | None) -> str | None:
    if cpu is None:
        return None
    try:
        return Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor").read_text().strip() or None
    except OSError:
        return None


def unpinned_workers(workers: int) -> list[dict]:
    print('warning: running unpinned; workers may share CPUs and performance comparisons may be noisy', file=sys.stderr)
    return [{"cpu": None, "numa_node": None, "pin_command": [], "scaling_governor": None} for _ in range(workers)]


def worker_placements(workers: int) -> list[dict]:
    if platform.system() != "Linux":
        if workers > 1:
            raise ValueError("Parallel measurement requires Linux and numactl")
        return unpinned_workers(workers)
    if not shutil.which("numactl"):
        raise ValueError("Linux measurement requires numactl for CPU and NUMA memory binding")
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
                command = ["numactl", f"--physcpubind={cpu}", f"--membind={node}"]
                placements.append({"cpu": cpu, "numa_node": node, "pin_command": command,
                                   "scaling_governor": scaling_governor(cpu)})
        if not any(by_node.values()) and len(placements) < workers:
            if not placements:
                raise ValueError("No allowed physical core with local memory to pin a worker to")
            print(f"warning: only {len(placements)} distinct allowed physical cores with local memory; "
                  f"using {len(placements)} pinned workers instead of {workers}", file=sys.stderr)
            break
    return placements


def validate_placements(placements):
    for worker in placements:
        if worker["pin_command"]:
            subprocess.run([*worker["pin_command"], sys.executable, "-c",
                            f"import os; assert os.sched_getaffinity(0) == {{{worker['cpu']}}}"], check=True)


def public_metadata(record):
    return {key: value for key, value in record.items() if key != "pin_command"}
