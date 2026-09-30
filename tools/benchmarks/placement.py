"""Automatic worker CPU selection and NUMA binding (Linux)."""

from __future__ import annotations

import itertools
import os
import re
import sys
from pathlib import Path


def scaling_governor(cpu: int) -> str | None:
    try:
        return Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor").read_text().strip()
    except OSError:  # no cpufreq driver, e.g. in a virtual machine
        return None


def unpinned_workers(workers: int) -> list[dict]:
    print('warning: running unpinned; workers may share CPUs and performance comparisons may be noisy', file=sys.stderr)
    return [{"cpu": None, "numa_node": None, "pin_command": [], "scaling_governor": None} for _ in range(workers)]


def worker_placements(workers: int) -> list[dict]:
    allowed = re.search(r"^Mems_allowed_list:\s*(.+)$", Path("/proc/self/status").read_text(), re.MULTILINE)[1]
    memory_nodes = set()
    for part in allowed.split(","):
        lo, _, hi = part.partition("-")
        memory_nodes.update(range(int(lo), int(hi or lo) + 1))

    by_node, seen = {}, set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        core = tuple(int((root / "topology" / key).read_text())
                     for key in ("physical_package_id", "core_id"))
        node = int(next(root.glob("node[0-9]*")).name[4:])
        if core in seen or node not in memory_nodes:  # one CPU per physical core, with local memory
            continue
        seen.add(core)
        by_node.setdefault(node, []).append(cpu)

    # Round-robin over the nodes.
    queues = []
    for node in sorted(by_node):
        queue = []
        for cpu in by_node[node]:
            queue.append((cpu, node))
        queues.append(queue)
    order = []
    for column in itertools.zip_longest(*queues):  # the first CPU of each node, then the second, ...
        for pair in column:
            if pair:  # zip_longest pads the shorter queues with None
                order.append(pair)
    if len(order) < workers:
        print(f"warning: only {len(order)} distinct allowed physical cores with local memory; "
              f"using {len(order)} pinned workers instead of {workers}", file=sys.stderr)
    return [{"cpu": cpu, "numa_node": node, "pin_command": ["numactl", f"--physcpubind={cpu}", f"--membind={node}"],
             "scaling_governor": scaling_governor(cpu)} for cpu, node in order[:workers]]


def public_metadata(record):
    return {key: value for key, value in record.items() if key != "pin_command"}
