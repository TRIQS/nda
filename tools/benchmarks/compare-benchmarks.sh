#!/bin/sh
# Paired PR comparison: clone and build both revisions, then hand the two build
# directories to compare.py, which does all the measuring.
#
# Revisions come from the environment. Jenkins PR build: CHANGE_TARGET (baseline,
# target branch tip) and CHANGE_ID (candidate, the PR head). Local: BASELINE_REF and
# CANDIDATE_REF (default HEAD).
set -eu
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
# Both revisions compile the candidate's benchmark harness against their own headers.
# Each is configured with its own benchmark_tracked preset; comparison.md lists any difference.
# A baseline that predates CMakePresets.json borrows the candidate's.
rm -rf "$root/baseline-source/benchmarks"
cp -R "$root/candidate-source/benchmarks" "$root/baseline-source/benchmarks"
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
  --min-time "${BENCHMARK_MIN_TIME:?Set BENCHMARK_MIN_TIME}" --repetitions 1 "$@"
