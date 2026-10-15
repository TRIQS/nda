#!/bin/sh
# Temporary same-revision comparison using two independent builds.
# WORKSPACE defaults to this checkout, WORKSPACE_TMP to /tmp and BENCHMARK_WORKERS to 12.
set -eu

# Single-threaded measurements: the workers already measure one case per core in parallel.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBLIS_NUM_THREADS=1
WORKSPACE=${WORKSPACE:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
WORKSPACE_TMP=${WORKSPACE_TMP:-/tmp}
sha=$(git -C "$WORKSPACE" rev-parse --verify 'HEAD^{commit}')
mkdir -p "$WORKSPACE_TMP"
root=$(mktemp -d "$WORKSPACE_TMP/benchmark-self-comparison.XXXXXX")
for side in baseline candidate; do
  git clone --quiet --no-checkout "$WORKSPACE" "$root/$side-source"
  git -C "$root/$side-source" checkout --quiet --detach "$sha"
done
echo "Temporary comparison: baseline and candidate both use $sha"

for side in candidate baseline; do
  sh "$WORKSPACE/tools/benchmarks/build-benchmarks.sh" "$root/$side-source" "$root/$side-build"
done

numactl --hardware
set --
[ -z "${BENCHMARK_FILTER:-}" ] || set -- --filter "$BENCHMARK_FILTER"
python3 "$WORKSPACE/tools/benchmarks/compare.py" \
  --baseline-build "$root/baseline-build" --candidate-build "$root/candidate-build" \
  --outdir "$WORKSPACE/benchmark-results" \
  --workers "${BENCHMARK_WORKERS:-12}" "$@"
