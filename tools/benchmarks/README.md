# Tracked benchmarks

The tracked benchmarks in `benchmarks/tracked/` compare a pull request's performance
with its target branch. Jenkins runs the comparison after the regular build and test
matrix succeeds, using a dedicated benchmark node. Results are posted as two PR
comments and never fail the build.

## How a PR comparison runs

1. **Jenkins** ([jenkins/Jenkinsfile](../../jenkins/Jenkinsfile)) runs
   [compare-benchmarks.sh](compare-benchmarks.sh), then [charts.py](charts.py),
   and archives `benchmark-results/`. It also runs [push-charts.sh](push-charts.sh)
   to upload charts and [post-comment.sh](post-comment.sh) to publish the PR comments.
2. **`compare-benchmarks.sh`** picks the two commits: the PR head
   (**candidate**) and the target branch tip (**baseline**). It clones each one and builds
   it with [build-benchmarks.sh](build-benchmarks.sh), which uses the `benchmark_tracked`
   preset. Each side builds with its own sources, preset, and dependency pins. It then runs
   [compare.py](compare.py) on the two build directories.
3. **`compare.py`** times the benchmarks:

   - It records both builds (compiler, flags, commits, benchmark sources) and lists how
     they differ.
   - It lists the cases in each build. A case found on only one side is reported as
     added or removed and is not timed.
   - It measures each case repeatedly on the same CPU core, alternating which build
     runs first. Measurements use CPU time per iteration.
   - It decides each case's verdict with [stats.py](stats.py) (see below), then writes
     `comparison.json`, `output.log`, and `comparison.md`.
4. **`charts.py`** draws one bar chart per suite and writes `charts.md`.
5. **`push-charts.sh`** uploads the charts to `refs/benchmarks/pr-<N>`.
   **`post-comment.sh`** posts `comparison.md` and `charts.md` as PR
   comments. Later runs of the same PR edit those comments instead of adding new ones.

The support modules handle shared tasks: [common.py](common.py) runs binaries,
[metadata.py](metadata.py) records the environment and builds, and
[reporting.py](reporting.py) writes `comparison.md`.

## How a case gets its verdict

`stats.py` analyzes the paired timings from both builds:

1. **Outliers:** a round is dropped when either side is 10 or more robust standard
   deviations from its median. If that would drop more than half the rounds, none are
   dropped.
2. **Speedup:** the median over rounds of baseline time / candidate time. Above 1 means
   the candidate is faster.
3. **Noise:** if either side varies by more than 20% (coefficient of variation) over
   the rounds, the case is marked `too_noisy` and no test is run.
4. **Paired t-test:** the per-round timing differences determine the verdict:

   - `improvement` or `regression`: the candidate is faster or slower by more than 5% of
     the baseline, at a one-sided false-positive rate of 0.1%.
   - `unchanged`: neither test passed. This means no change was shown, not that the two
     are equal.

The thresholds are constants at the top of `stats.py`. The launch timing (0.2 s minimum,
0.1 s warm-up) is set at the top of `common.py`.

## Results and failures

`benchmark-results/` holds:

- `comparison.json`: settings, machine, both builds, and every case's per-round times
  and verdict.
- `comparison.md` and `charts.md`: the two comments.
- `output.log`: the command and console output of every launch.

A failure marks the benchmark stage unstable and leaves the PR build result unchanged:

- Either side fails to build: nothing is posted.
- No case exists in both builds: a "Comparison failed" comment is posted.
- A launch fails: that case is marked `incomplete` and the comparison comment is posted,
  but the charts are skipped. Look in `output.log` for the failing launch.

## Running locally

Needs `numactl`, `scipy`, `pandas`, and `archspec`. The charts also need `matplotlib`.

Run commands from the repository root. To compare the current commit against `unstable`:

```sh
BASELINE_REF=unstable sh tools/benchmarks/compare-benchmarks.sh
python3 tools/benchmarks/charts.py benchmark-results
```

For existing builds, replace the directory names with your build paths:

```sh
python3 tools/benchmarks/compare.py --baseline-build build-baseline \
    --candidate-build build-candidate --outdir results
```

Use `--help` on the Python scripts for filters, measurement rounds, and worker options.

## Performance history

[history.py](history.py) helps locate performance changes across a commit range,
including both endpoints along the first-parent history. It builds each revision and
caches the binaries in `/tmp/nda-history-benchmarks`; clear the cache after switching
compilers. Timings and metadata go into `history.json`, and
[history-charts.py](history-charts.py) automatically generates charts and `history.md`.

For example, measure the last six commits:

```sh
python3 tools/benchmarks/history.py HEAD~5 HEAD --outdir history-results
```

Charts show CPU time at each commit, with colored segments for detected changes and
gaps for unavailable results.

To regenerate charts from an existing run without repeating the measurements:

```sh
python3 tools/benchmarks/history-charts.py history-results
```
