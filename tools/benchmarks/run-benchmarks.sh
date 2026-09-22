#!/bin/sh
set -eu

workers=${BENCHMARK_WORKERS:?Set BENCHMARK_WORKERS explicitly}

cmake -S "$WORKSPACE" -B "$WORKSPACE_TMP/build" \
  -DCMAKE_INSTALL_PREFIX="$WORKSPACE_TMP/install" \
  -DBuild_Benchs=ON -DBuild_Tests=OFF -DPythonSupport=OFF -DCMAKE_BUILD_TYPE=Release
set --
for source in "$WORKSPACE"/benchmarks/tracked/ops_*.cpp; do
  target=$(basename "$source" .cpp)
  set -- "$@" "$target"
done
cmake --build "$WORKSPACE_TMP/build" --parallel "$PARALLEL" --target "$@"

numactl --hardware
python3 "$WORKSPACE/tools/benchmarks/run.py" \
  --bindir "$WORKSPACE_TMP/build/benchmarks/tracked" \
  --outdir "$WORKSPACE/benchmark-results" \
  --workers "$workers" --repetitions 5 --min-time 0.1s
