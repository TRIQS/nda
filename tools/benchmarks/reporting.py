"""The Markdown comparison report for the pull-request comment."""

import math
import re
import statistics

import pandas as pd

import common
import stats


# case structure: what a case name says about the case (common.case_factors), used to group the report

def _operand(f):
    return ('strided slice' if f['strided'] else 'matrix' if f['matrix'] else 'array') + (' + scalar' if f['scalar'] else '')


GROUPINGS = [  # Report title and the column to group by.
    ('By value type', 'value_type'),
    ('By suite (operation class)', 'suite'),
    ('By operand kind', 'operand_kind'),
    ('By size N', 'N'),
]


def _speedup_words(x, floor):
    """A speedup (candidate speed / baseline speed) as 'Nx faster', 'Nx slower' or 'Nx (no change)', no change being
    within floor percent."""
    limit = 1 + floor / 100
    return f'{x:.2f}x faster' if x >= limit else f'{1 / x:.2f}x slower' if x <= 1 / limit else f'{x:.2f}x (no change)'


def _fmt_bytes(n):
    return f'{n / 2**30:.0f} GiB' if n >= 2**30 else f'{n / 2**20:.0f} MiB' if n >= 2**20 else f'{n // 1024} KiB'


def _fmt_speedup(case):
    """'median speedup ± its standard error', the error taken from the standard error of the per-round log speedup."""
    speedup = case['speedup']
    pm = speedup * (math.exp(case['speedup_log_se']) - 1)
    digits = 2 if speedup < 10 else 1
    return f'{speedup:.{digits}f}x ± {pm:.{digits}f}'


CASE_COLS = ['| Suite | Case | Speedup (candidate speed / baseline speed) |', '|:--|:--|--:|']


def _case_cells(case):
    """Suite and case name for the tables, without the ops_ prefix and the layout tags."""
    name = re.sub(r'_[A-Z]_layout', '', case['name']).replace('|', '\\|')
    return f'| {case["suite"].removeprefix("ops_")} | {name} |'


def _case_row(case):
    return f'{_case_cells(case)} {_fmt_speedup(case)} |'


REPOSITORY = 'https://github.com/TRIQS/nda'  # PR commits are shown under the target repository too


def _commit_link(url, sha):
    return f'[`{sha[:common.SHORT_SHA]}`]({url.removesuffix(".git")}/commit/{sha})'


def _details(summary, lines):
    return ['<details>', f'<summary>{summary}</summary>', ''] + lines + ['', '</details>', '']


def _group_section(title, column, cases):
    """Summarize one column: sizes in numeric order, other groups by most regressions first."""
    grouped = cases.groupby(column)
    summary = grouped.agg(cases=('speedup', 'size'), geomean=('speedup', statistics.geometric_mean),
                          slowest=('speedup', 'min'), fastest=('speedup', 'max'))
    counts = grouped['status'].value_counts().unstack(fill_value=0)
    # Include zero counts for verdicts absent from the comparison.
    statuses = ['regression', 'improvement', 'unchanged', 'too_noisy']
    counts = counts.reindex(columns=statuses, fill_value=0)
    summary = summary.join(counts).rename_axis('group')
    if column == 'N':
        summary = summary.sort_index()
        summary.index = 'N=' + summary.index.astype(str)
    else:
        summary = summary.sort_values(['regression', 'geomean', 'group'], ascending=[False, True, True])
    table = ['| group | cases | geo-mean speedup | range (slowest .. fastest) | regressions | improvements | no change | too noisy |',
             '|:--|--:|:--|:--|--:|--:|--:|--:|']
    for row in summary.reset_index().itertuples(index=False):
        table.append(f'| {row.group} | {row.cases} | {row.geomean:.2f}x | {row.slowest:.2f}x .. {row.fastest:.2f}x | '
                     f'{row.regression} | {row.improvement} | {row.unchanged} | {row.too_noisy} |')
    return _details(f'<b>{title}</b>', table)


COMMENT_LIMIT = 65000  # GitHub's limit is 65,536; post-comment.sh adds a marker line


def write_comparison_report(path, results):
    """comparison.md for the PR comment. Above COMMENT_LIMIT the unchanged, then also the improvements list is left out."""
    for omit in ((), ('unchanged',), ('unchanged', 'improvements')):
        text = _render(results, omit)
        if len(text) <= COMMENT_LIMIT:
            break
    path.write_text(text)


def _render(results, omit):
    settings = results['settings']
    floor = settings['min_improvement_percent']
    # suite, then the name without its size, then N as a number: a string sort puts /1024 before /16
    cases = sorted(results['cases'], key=lambda c: (c['suite'], c['name'].rsplit('/', 1)[0], int(c['name'].rsplit('/', 1)[1])))
    measured = [c for c in cases if 'speedup' in c]
    by_status = {s: [c for c in cases if c['status'] == s]
                 for s in ('regression', 'improvement', 'unchanged', 'too_noisy', 'incomplete', 'added', 'removed')}
    builds = results['builds']
    ci = results['ci']

    def label(side):
        link = _commit_link(REPOSITORY, builds[side]['provenance']['commit'])
        return common.side_label(results, side, f'PR #{ci["pr"]}, {link}' if side == 'candidate' and ci['pr'] else link)

    L = [f'## Benchmark comparison: candidate {label("candidate")} vs baseline {label("baseline")}', '']
    only = []
    for status, side in (('added', 'candidate'), ('removed', 'baseline')):
        if by_status[status]:
            only += _details(f'{len(by_status[status])} cases only in the {side} build (not compared)',
                             ['| Suite | Case |', '|:--|:--|'] + [_case_cells(c) for c in by_status[status]])
    if results.get('error'):  # nothing was measured: say why and stop
        return '\n'.join(L + [f'**Comparison failed: {results["error"]}**', ''] + only) + '\n'
    overall = (f'geometric mean {_speedup_words(statistics.geometric_mean(c["speedup"] for c in measured), floor)}'
               if measured else 'no measured cases')
    L += [f'**{len(by_status["regression"])} regressions, {len(by_status["improvement"])} improvements, '
          f'{len(by_status["unchanged"])} unchanged, {len(by_status["too_noisy"])} too noisy** out of {len(cases)} cases; '
          f'{overall}.', '']
    if by_status['incomplete']:
        L += [f'{len(by_status["incomplete"])} cases incomplete (a launch failed; see output.log).', '']
    L += only
    if differences := results['build_differences']:
        L += ['**The two sides were built differently**, so the speedups include the effect of these changes:', '']
        L += [f'- `{name}`: {detail}' for name, detail in differences.items()] + ['']
    L += [f'Speedup = candidate speed / baseline speed, median over rounds ± its standard error. A signal needs a change beyond {floor:g}% of the baseline at a '
          f'one-sided false-positive rate of {100 * settings["alpha"]:g}%. Signals do not fail CI.', '']

    # one folded list per verdict, with the count in the summary line
    lists = [('regressions', sorted(by_status['regression'], key=lambda c: c['speedup']), 'slowest first'),
             ('improvements', sorted(by_status['improvement'], key=lambda c: -c['speedup']), 'fastest first'),
             ('unchanged', sorted(by_status['unchanged'], key=lambda c: c['speedup']), 'slowest first'),
             ('too noisy', by_status['too_noisy'], f'a side\'s per-round cv exceeded {settings["max_noise_percent"]:g}%; no verdict')]
    for name, listed, note in lists:
        if not listed:
            continue
        if name in omit:  # the case lists grow with the suite
            L += [f'The {len(listed)} {name} are left out to fit the comment size limit; see `comparison.json` in the build artifacts.', '']
        else:
            L += _details(f'<b>{len(listed)} {name}</b> ({note})', CASE_COLS + [_case_row(c) for c in listed])

    # structured summary: one foldable table per grouping
    if measured:
        rows = []
        for case in measured:
            factors = common.case_factors(case['suite'], case['name'])
            rows.append({'suite': factors['suite'], 'value_type': common.value_type_label(factors['value_type']),
                         'operand_kind': _operand(factors), 'N': factors['N'],
                         'speedup': case['speedup'], 'status': case['status']})
        case_table = pd.DataFrame(rows)
        L += ['### Where the changes are', '']
        for title, column in GROUPINGS:
            L += _group_section(title, column, case_table)

    method = [
        f'- Each revision builds its own benchmarks with its own `benchmark_tracked` preset; differences are listed above. '
        f'Each case is run in **{settings["rounds"]} rounds**; a round launches '
        f'the baseline and the candidate binary once each, as separate processes on the same pinned core, alternating which goes first.',
        f'- A launch discards {settings["min_warmup_time"]} s of warm-up, then times the case for at least {settings["min_time"]}; '
        f'Google Benchmark\'s CPU time per iteration is the launch\'s value. One extra warm-up launch per side is discarded.',
        f'- **Outlier rounds**: a round where either side is {settings["outlier_sigma"]:g} or more robust standard deviations '
        f'({stats.MAD_TO_SIGMA:g} x median absolute deviation) from its median is left out: '
        f'{sum(len(c["dropped_rounds"]) for c in measured)} of {sum(len(c["times_ns"]["baseline"]) for c in measured)} rounds.',
        f'- **Speedup** is the median over rounds of baseline time / candidate time.',
        f'- **Regression / improvement**: a paired t-test on the per-round differences finds the candidate slower / faster by more than '
        f'{floor:g}% of the baseline, one-sided at a {100 * settings["alpha"]:g}% false-positive rate.',
        f'- **Too noisy**: one side\'s coefficient of variation over rounds exceeds {settings["max_noise_percent"]:g}%; no verdict.',
        '- **Unchanged**: no change beyond the floor was shown; not proof of equality.',
    ]
    L += _details('How this was measured', method)

    machine = results['machine']
    env = [f'- **Machine**: {machine["cpu_model"]}, {machine["sockets"]} socket(s) x {machine["cores_per_socket"]} cores x '
           f'{machine["threads_per_core"]} threads, {_fmt_bytes(machine["ram_bytes"])} RAM, {machine["numa_nodes"]} NUMA node(s); '
           f'L1d {_fmt_bytes(machine["l1d_cache_bytes"])}, L2 {_fmt_bytes(machine["l2_cache_bytes"])}, L3 {_fmt_bytes(machine["l3_cache_bytes"])}; '
           f'{machine["system"]} {machine["release"]}.',
           f'- **Microarchitecture**: `{machine["microarchitecture"]["name"]}`, `{machine["microarchitecture"]["level"]}`.',
           f'- **CPU frequency governor**: {", ".join(machine["scaling_governors"]) or "unknown"}.']
    # Shown once, for the candidate: the baseline differs only as listed under "built differently" above, and the
    # two commits are in the title.
    candidate = builds['candidate']
    dependencies = sorted(candidate['provenance']['dependencies'].items())
    env += [f'- **Compiler**: {candidate["compiler"]}.',
            f'- **Compiler flags**: `{" ".join(candidate["flags"])}`.',
            f'- **Compile definitions**: `{" ".join(candidate["defines"])}`.']
    dependency_links = ', '.join(
        f'{name} {_commit_link(dep["url"], dep["commit"])}'
        for name, dep in dependencies
    )
    env.append(f'- **Dependencies**: {dependency_links}.')
    if ci['url']:
        env.append(f'- **CI**: [{ci["url"]}]({ci["url"]}).')
    L += _details('Environment and build', env)
    return '\n'.join(L) + '\n'
