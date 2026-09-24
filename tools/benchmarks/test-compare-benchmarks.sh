#!/bin/sh
# Temporary same-revision comparison using two independent builds.
set -eu

worker_counts=${BENCHMARK_WORKERS:?Set BENCHMARK_WORKERS explicitly}
min_time=${BENCHMARK_MIN_TIME:?Set BENCHMARK_MIN_TIME explicitly}
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
for workers in $worker_counts; do
  python3 "$WORKSPACE/tools/benchmarks/compare.py" \
    --baseline-build "$root/baseline-build" --candidate-build "$root/candidate-build" \
    --outdir "$WORKSPACE/benchmark-results/workers-$workers" \
    --workers "$workers" --rounds 6 --threshold 3 --min-improvement 5 --max-noise 20 --max-retries 3 --retry-factor 1.5 --min-warmup-time 0.1 \
    --repetitions 1 --min-time "$min_time" "$@"
done
