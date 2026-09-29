"""The Markdown comparison report for the pull-request comment."""

import math
import statistics


# case structure: what a case name says about the case, used to group the report

def case_factors(suite, name):
    """Attributes parsed from a case's suite and name, e.g. ('ops_mapped', 'c128/A2_C_layout,S/max/64')."""
    value_type, operands, op, size = name.split('/')
    kinds = operands.split(',')
    return {
        'suite': suite.removeprefix('ops_'),  # arithmetic, mapped, reductions, math: the operation class
        'op': op,
        'complex': value_type == 'c128',
        'N': int(size),
        'matrix': any(k.startswith(('M2', 'V1')) for k in kinds),
        'scalar': 'S' in kinds,
        'strided': 'slice' in operands,
    }


def _operand(f):
    return ('strided slice' if f['strided'] else 'matrix' if f['matrix'] else 'array') + (' + scalar' if f['scalar'] else '')


def _value_type(f):
    return 'c128' if f['complex'] else 'f64'


GROUPINGS = [  # (title, key(factors)) for the foldable tables of the report
    ('By value type', lambda f: 'complex (c128)' if f['complex'] else 'real (f64)'),
    ('By suite (operation class)', lambda f: f['suite']),
    ('By operand kind', _operand),
    ('By size N', lambda f: f"N={f['N']}"),
    ('By value type and suite', lambda f: f"{_value_type(f)} {f['suite']}"),
    ('By value type and size', lambda f: f"{_value_type(f)} N={f['N']}"),
    ('By suite and size', lambda f: f"{f['suite']} N={f['N']}"),
]


def _median_speedup(logs):
    """Median speedup (candidate speed / baseline speed) from log(candidate time / baseline time) values."""
    x = math.exp(-statistics.median(logs))
    return f'{x:.2f}x faster' if x >= 1.05 else f'{1 / x:.2f}x slower' if x <= 1 / 1.05 else f'{x:.2f}x (no change)'


def _fmt_bytes(n):
    return f'{n / 2**30:.0f} GiB' if n >= 2**30 else f'{n / 2**20:.0f} MiB' if n >= 2**20 else f'{n // 1024} KiB'


def _speedup(case):
    """(median speedup = candidate speed / baseline speed = baseline time / candidate time, its +- from the standard error
    of the per-round log ratio) or None."""
    analysis = case['analysis']
    if 'median_ratio' not in analysis:
        return None
    logs = [math.log(r['samples']['baseline']['cpu_time_ns'] / r['samples']['candidate']['cpu_time_ns']) for r in case['rounds']]
    speedup = analysis['median_ratio']
    pm = speedup * (math.exp(statistics.stdev(logs) / math.sqrt(len(logs))) - 1)
    return speedup, pm


def _fmt_speedup_pm(sp):
    if sp is None:
        return '—'
    speedup, pm = sp
    digits = 2 if speedup < 10 else 1
    return f'{speedup:.{digits}f}x ± {pm:.{digits}f}'


STATUS_LABEL = {'regression_signal': 'regression', 'improvement_signal': 'improvement', 'inconclusive': 'no change',
                'too_noisy': 'too noisy', 'incomplete': 'incomplete', 'added': 'added', 'removed': 'removed'}
CASE_COLS = ['| Suite | Case | Speedup (candidate speed / baseline speed) | Status |', '|:--|:--|--:|:--|']


def _case_row(case):
    analysis = case['analysis']
    name = case['name'].replace('|', '\\|')
    status = STATUS_LABEL[analysis['status']]
    if analysis['status'] == 'too_noisy':
        status += f', cv {max(analysis["noise_percent"].values()):.0f}%'
    return f'| {case["suite"]} | {name} | {_fmt_speedup_pm(_speedup(case))} | {status} |'


def _fmt_setting(value):
    return 'unset' if value in (None, '') else f'`{value}`'


def _details(summary, lines, open_=False):
    return [f'<details{" open" if open_ else ""}>', f'<summary>{summary}</summary>', ''] + lines + ['', '</details>', '']


def _group_section(title, key, cases):
    """A foldable grouping: one table row per group with status counts, then a foldable case list per group."""
    groups = {}
    for c in cases:
        groups.setdefault(key(case_factors(c['suite'], c['name'])), []).append(c)

    def order(g):  # most regressions first, then the slowest median: what needs a look comes to the top
        cs = groups[g]
        sps = [c['analysis']['median_ratio'] for c in cs]
        return (-sum(c['analysis']['status'] == 'regression_signal' for c in cs), statistics.median(sps), str(g))

    def counts(cs):
        n = {k: sum(c['analysis']['status'] == k for c in cs) for k in ('regression_signal', 'improvement_signal', 'inconclusive', 'too_noisy')}
        return n

    table = ['| group | cases | median speedup | range (slowest .. fastest) | regressions | improvements | no change | too noisy |',
             '|:--|--:|:--|:--|--:|--:|--:|--:|']
    for g in sorted(groups, key=order):
        cs = groups[g]
        sps = [c['analysis']['median_ratio'] for c in cs]
        n = counts(cs)
        med = f'{statistics.median(sps):.2f}x'
        rng = f'{min(sps):.2f}x .. {max(sps):.2f}x'
        table.append(f'| {g} | {len(cs)} | {med} | {rng} | {n["regression_signal"]} | {n["improvement_signal"]} | {n["inconclusive"]} | {n["too_noisy"]} |')
    return _details(f'<b>{title}</b>', table)


def write_comparison_report(path, manifest, metadata):
    """comparison.md for the pull-request comment: headline, regressions, foldable groupings, folded detail.
    Kept under GitHub's 65,536-character comment limit by listing each case once, in the folded 'All cases' table."""
    settings = metadata['settings']
    floor = settings['min_improvement_percent']
    cases = sorted(manifest['cases'], key=lambda c: (c['suite'], c['name']))
    measured = [c for c in cases if 'median_ratio' in c['analysis']]
    logs = {(c['suite'], c['name']): -math.log(c['analysis']['median_ratio']) for c in measured}
    by_status = {s: [c for c in cases if c['analysis']['status'] == s]
                 for s in ('regression_signal', 'improvement_signal', 'inconclusive', 'too_noisy', 'incomplete', 'added', 'removed')}
    builds = metadata['builds']
    short = {side: b['provenance']['commit'][:10] for side, b in builds.items()}
    branch = {side: '' if b['provenance']['branch'] == 'HEAD' else f' ({b["provenance"]["branch"]})' for side, b in builds.items()}
    machine = metadata['machine']

    L = [f'## Benchmark comparison: candidate `{short["candidate"]}`{branch["candidate"]} vs baseline `{short["baseline"]}`{branch["baseline"]}', '']
    if manifest.get('error'):
        L += [f'**Comparison failed: {manifest["error"]}**', '']
    overall = _median_speedup(list(logs.values())) if logs else 'n/a'
    L += [f'**{len(by_status["regression_signal"])} regressions, {len(by_status["improvement_signal"])} improvements, '
          f'{len(by_status["inconclusive"])} unchanged, {len(by_status["too_noisy"])} too noisy** out of {len(cases)} cases; '
          f'median case {overall}.', '']
    extra = {k: len(v) for k, v in by_status.items() if k in ('incomplete', 'added', 'removed') and v}
    if extra:
        L += ['Also: ' + ', '.join(f'{n} {k}' for k, n in extra.items()) + '.', '']
    if differences := metadata.get('build_differences'):
        L += ['**The two sides were built differently**, so the speedups include the effect of these changes '
              '(`benchmark_tracked` preset variables, resulting compiler flags, benchmark sources):', '']
        L += [f'- `{d["name"]}`: baseline {_fmt_setting(d["baseline"])}, candidate {_fmt_setting(d["candidate"])}' for d in differences] + ['']
    L += [f'Speedup = candidate speed / baseline speed, median over rounds ± its standard error. A signal needs a change beyond {floor:g}% of the baseline at a '
          f'one-sided false-positive rate of {100 * settings["alpha"]:g}%. Signals do not fail CI.', '']

    # signals first, folded but with the counts in the summary line
    regressions = sorted(by_status['regression_signal'], key=lambda c: c['analysis']['median_ratio'])
    improvements = sorted(by_status['improvement_signal'], key=lambda c: -c['analysis']['median_ratio'])
    if regressions:
        L += _details(f'<b>{len(regressions)} regressions</b> (slowest first)', CASE_COLS + [_case_row(c) for c in regressions])
    if improvements:
        L += _details(f'<b>{len(improvements)} improvements</b> (fastest first)', CASE_COLS + [_case_row(c) for c in improvements])
    if by_status['too_noisy']:
        L += _details(f'<b>{len(by_status["too_noisy"])} too noisy</b> (a side\'s per-round cv exceeded {settings["max_noise_percent"]:g}%; no verdict)',
                      CASE_COLS + [_case_row(c) for c in by_status['too_noisy']])

    # structured summary: one foldable table per grouping
    if measured:
        L += ['### Where the changes are', '',
              'Cases grouped by what their names say (value type, suite, operation class, operand kinds, size). '
              + 'Open a grouping for its table; rows are sorted by regressions, then by median speedup (slowest first). The individual cases are in "All cases" below.', '']
        for title, key in GROUPINGS:
            L += _group_section(title, key, measured)
    # everything else, folded
    order = {'regression_signal': 0, 'too_noisy': 1, 'improvement_signal': 2, 'inconclusive': 3}
    by_verdict = sorted(cases, key=lambda c: (order.get(c['analysis']['status'], 4), c['analysis'].get('median_ratio', 1)))
    L += _details(f'All {len(cases)} cases (regressions, too noisy, improvements, then no change; each by speedup)',
                  CASE_COLS + [_case_row(c) for c in by_verdict])

    method = [
        f'- Each revision builds its own tracked benchmarks with its own `benchmark_tracked` preset (see Environment); each case present on both sides is measured on one pinned physical core, '
        f'with both sides launched as separate processes in **{settings["rounds"]} alternating rounds** (candidate first in odd rounds, baseline first in even rounds).',
        f'- Each launch runs Google Benchmark with a {settings["min_warmup_time"]} s in-process warm-up (discarded) and then times the case for at least '
        f'`{settings["min_time"]}`; with {settings["repetitions"]} repetition(s) per launch the {settings["repetition_statistic"]} CPU time is the launch\'s value. '
        'One warm-up launch per side precedes the rounds.',
        f'- The change of a case is the median over rounds of baseline/candidate. The verdict comes from a paired t-test on the per-round '
        f'differences: a regression or improvement is signalled when the candidate is slower or faster by more than {floor:g}% of the baseline mean beyond '
        f'the noise, tested one-sided in each direction at level {100 * settings["alpha"]:g}% (Student\'s t cutoff for the number of rounds).',
        f'- A case is **too noisy** when either side\'s coefficient of variation over rounds exceeds {settings["max_noise_percent"]:g}%; '
        'it gets no verdict.',
        '- **unchanged** means no change beyond the floor could be shown at this level, not that the two are equal.',
    ]
    L += _details('How this was measured', method)

    ctx = metadata['benchmark_context'] or {}
    env = [f'- **Machine**: {machine["cpu_model"]}, {machine["sockets"]} socket(s) x {machine["cores_per_socket"]} cores x '
           f'{machine["threads_per_core"]} threads, {_fmt_bytes(machine["ram_bytes"])} RAM, {machine["numa_nodes"]} NUMA node(s); '
           f'L1d {_fmt_bytes(machine["l1d_cache_bytes"])}, L2 {_fmt_bytes(machine["l2_cache_bytes"])}, L3 {_fmt_bytes(machine["l3_cache_bytes"])}; '
           f'{machine["system"]} {machine["release"]}.',
           f'- **CPU frequency scaling**: {"enabled" if ctx.get("cpu_scaling_enabled") else "disabled"}'
           + (f'; governor {", ".join(sorted({p["scaling_governor"] for p in settings["worker_placements"] if p["scaling_governor"]}) or ["unknown"])}' if settings.get('pinned') else '')
           + '.',
           f'- **Workers**: {settings["workers"]} {"pinned to distinct physical cores with local NUMA memory" if settings.get("pinned") else "unpinned"}'
           + (f' (CPUs {", ".join(str(p["cpu"]) for p in settings["worker_placements"])})' if settings.get('pinned') else '') + '.']
    for side in ('baseline', 'candidate'):
        b = builds[side]; c = b['compiler']; prov = b['provenance']
        env.append(f'- **{side.title()}**: commit `{prov["commit"]}` ({prov["branch"]})'
                   + f'; {c["banner"]}; `{c["build_type"]}` with flags `{c["effective_flags"]}`'
                   + f'; dependencies ' + ', '.join(f'{k} `{v[:10]}`' for k, v in sorted(prov['dependency_shas'].items())) + '.')
    if builds['baseline']['provenance']['dependency_shas'] != builds['candidate']['provenance']['dependency_shas']:
        env.append('- **Note**: the two sides resolved different dependency commits (see above).')
    ci = metadata['ci']
    if ci['url']:
        env.append(f'- **CI**: [{ci["url"]}]({ci["url"]}), PR {ci["pr"]}, `{ci["baseline_branch"]}` <- `{ci["candidate_branch"]}`.')
    env.append(f'- **Run**: started {metadata["started_at"]}, finished {metadata["finished_at"]}.')
    L += _details('Environment and build', env)

    path.write_text('\n'.join(L) + '\n')
