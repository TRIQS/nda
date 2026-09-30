#!/bin/sh
# Paired PR comparison: clone and build both revisions, then hand the two build
# directories to compare.py, which does all the measuring.
#
# Revisions come from the environment. Jenkins PR build: CHANGE_TARGET (baseline,
# target branch tip) and CHANGE_ID (candidate, the PR head). Local: BASELINE_REF and
# CANDIDATE_REF (default HEAD).
set -eu
# Single-threaded measurements: the workers already measure one case per core in parallel.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBLIS_NUM_THREADS=1
cd "$WORKSPACE"

if [ -n "${CHANGE_ID:-}" ]; then
  git fetch --quiet --no-tags origin "refs/heads/$CHANGE_TARGET"; baseline=$(git rev-parse FETCH_HEAD)
  git fetch --quiet --no-tags origin "refs/pull/$CHANGE_ID/head"; candidate=$(git rev-parse FETCH_HEAD)
else
  baseline=$(git rev-parse --verify "${BASELINE_REF:?Set BASELINE_REF}^{commit}")
  candidate=$(git rev-parse --verify "${CANDIDATE_REF:-HEAD}^{commit}")
fi
echo "baseline  $baseline"
echo "candidate $candidate"

mkdir -p "$WORKSPACE_TMP"
root=$(mktemp -d "$WORKSPACE_TMP/benchmark-comparison.XXXXXX")
checkout() {  # checkout <side> <sha>: a fresh clone of the workspace at that commit
  git clone --quiet --no-checkout "$WORKSPACE" "$root/$1-source"
  git -C "$root/$1-source" checkout --quiet --detach "$2"
}
checkout candidate "$candidate"
checkout baseline "$baseline"
# Each revision builds its own tracked benchmarks with its own benchmark_tracked preset; cases
# present on both sides are compared and comparison.md lists any difference in configuration.
# Nothing is copied between the two trees except the preset, for a baseline that predates it.
[ -f "$root/baseline-source/CMakePresets.json" ] || cp "$root/candidate-source/CMakePresets.json" "$root/baseline-source/"

# Each side resolves its own dependencies from its own deps/CMakeLists.txt, so a PR that
# changes a pin is measured with that change. The two sides' dependency commits are
# recorded in metadata.json and may differ.
for side in candidate baseline; do
  sh "$WORKSPACE/tools/benchmarks/build-benchmarks.sh" "$root/$side-source" "$root/$side-build"
done

set --
[ -z "${BENCHMARK_FILTER:-}" ] || set -- --filter "$BENCHMARK_FILTER"
python3 "$WORKSPACE/tools/benchmarks/compare.py" \
  --baseline-build "$root/baseline-build" --candidate-build "$root/candidate-build" \
  --outdir "$WORKSPACE/benchmark-results" \
  --workers "${BENCHMARK_WORKERS:?Set BENCHMARK_WORKERS}" --rounds 12 --alpha 0.001 --min-improvement 5 --max-noise 20 --max-retries 0 --min-warmup-time 0.1 \
  --min-time 0.2s --repetitions 1 "$@"
