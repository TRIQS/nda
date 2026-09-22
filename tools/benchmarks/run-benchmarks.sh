#!/bin/sh
set -eu

worker_counts=${BENCHMARK_WORKERS:?Set BENCHMARK_WORKERS explicitly}

sh "$WORKSPACE/tools/benchmarks/build-benchmarks.sh" "$WORKSPACE" "$WORKSPACE_TMP/build"

numactl --hardware
for workers in $worker_counts; do
  python3 "$WORKSPACE/tools/benchmarks/run.py" \
    --bindir "$WORKSPACE_TMP/build/benchmarks/tracked" \
    --outdir "$WORKSPACE/benchmark-results/workers-$workers" \
    --workers "$workers" --repetitions 5 --min-time 0.1s
done
