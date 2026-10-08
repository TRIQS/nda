"""The Markdown comparison report for the pull-request comment."""

import math
import re
import statistics

import common


# case structure: what a case name says about the case (common.case_factors), used to group the report

def _operand(f):
    return ('strided slice' if f['strided'] else 'matrix' if f['matrix'] else 'array') + (' + scalar' if f['scalar'] else '')


GROUPINGS = [  # (title, key(factors), row order(factors) or None for most regressions first) for the foldable tables of the report
    ('By value type', lambda f: common.value_type_label(f['value_type']), None),
    ('By suite (operation class)', lambda f: f['suite'], None),
    ('By operand kind', _operand, None),
    ('By size N', lambda f: f"N={f['N']}", lambda f: f['N']),
]


def _speedup_words(x):
    """A speedup (candidate speed / baseline speed) as 'Nx faster', 'Nx slower' or 'Nx (no change)'."""
    return f'{x:.2f}x faster' if x >= 1.05 else f'{1 / x:.2f}x slower' if x <= 1 / 1.05 else f'{x:.2f}x (no change)'


def _fmt_bytes(n):
    return f'{n / 2**30:.0f} GiB' if n >= 2**30 else f'{n / 2**20:.0f} MiB' if n >= 2**20 else f'{n // 1024} KiB'


def _speedup(case):
    """(median speedup = candidate speed / baseline speed = baseline time / candidate time, its +- from the standard error
    of the per-round log ratio) or None."""
    analysis = case['analysis']
    if 'median_ratio' not in analysis:
        return None
    dropped = set(analysis.get('dropped_rounds', ()))  # outlier rounds
    logs = [math.log(r['samples']['baseline']['cpu_time_ns'] / r['samples']['candidate']['cpu_time_ns'])
            for r in case['rounds'] if r['round'] not in dropped]
    speedup = analysis['median_ratio']
    pm = speedup * (math.exp(statistics.stdev(logs) / math.sqrt(len(logs))) - 1)
    return speedup, pm


def _fmt_speedup_pm(sp):
    if sp is None:
        return '—'
    speedup, pm = sp
    digits = 2 if speedup < 10 else 1
    return f'{speedup:.{digits}f}x ± {pm:.{digits}f}'


CASE_COLS = ['| Suite | Case | Speedup (candidate speed / baseline speed) |', '|:--|:--|--:|']


def _case_cells(case):
    """Suite and case name for the tables, without the ops_ prefix and the layout tags."""
    name = re.sub(r'_[A-Z]_layout', '', case['name']).replace('|', '\\|')
    return f'| {case["suite"].removeprefix("ops_")} | {name} |'


def _case_row(case):
    return f'{_case_cells(case)} {_fmt_speedup_pm(_speedup(case))} |'


REPOSITORY = 'https://github.com/TRIQS/nda'  # PR commits are shown under the target repository too


def _commit_link(url, sha):
    return f'[`{sha[:10]}`]({url.removesuffix(".git")}/commit/{sha})'


def _fmt_libraries(libs):
    """'hdf5-1.12.3, openblas-0.3.29, openmpi-4.1.8' when the paths carry a package version (Nix, Spack), else the sonames."""
    names = []
    for soname, path in libs.items():
        # a package directory like .../<hash>-hdf5-1.12.3/lib or .../openblas-0.3.29/lib names the version
        version = re.search(r'([A-Za-z][A-Za-z0-9_+]*-\d+(?:\.\d+)+)', path.rsplit('/', 1)[0])
        name = version[1] if version else soname
        if name not in names:
            names.append(name)
    return ', '.join(f'`{n}`' for n in sorted(names))


def _fmt_setting(value):
    return 'unset' if value in (None, '') else f'`{value}`'


def _details(summary, lines, open_=False):
    return [f'<details{" open" if open_ else ""}>', f'<summary>{summary}</summary>', ''] + lines + ['', '</details>', '']


def _group_section(title, key, cases, row_order=None):
    """A foldable table, one row per group. Rows follow row_order if given (e.g. by N), else most regressions first."""
    groups, factors = {}, {}
    for c in cases:
        f = common.case_factors(c['suite'], c['name'])
        groups.setdefault(key(f), []).append(c)
        factors.setdefault(key(f), f)

    def order(g):
        if row_order:
            return row_order(factors[g])
        # most regressions first, then the slowest geometric mean: what needs a look comes to the top
        cs = groups[g]
        sps = [c['analysis']['median_ratio'] for c in cs]
        return (-sum(c['analysis']['status'] == 'regression_signal' for c in cs), statistics.geometric_mean(sps), str(g))

    def counts(cs):
        n = {k: sum(c['analysis']['status'] == k for c in cs) for k in ('regression_signal', 'improvement_signal', 'inconclusive', 'too_noisy')}
        return n

    table = ['| group | cases | geo-mean speedup | range (slowest .. fastest) | regressions | improvements | no change | too noisy |',
             '|:--|--:|:--|:--|--:|--:|--:|--:|']
    for g in sorted(groups, key=order):
        cs = groups[g]
        sps = [c['analysis']['median_ratio'] for c in cs]
        n = counts(cs)
        geo = f'{statistics.geometric_mean(sps):.2f}x'
        rng = f'{min(sps):.2f}x .. {max(sps):.2f}x'
        table.append(f'| {g} | {len(cs)} | {geo} | {rng} | {n["regression_signal"]} | {n["improvement_signal"]} | {n["inconclusive"]} | {n["too_noisy"]} |')
    return _details(f'<b>{title}</b>', table)


COMMENT_LIMIT = 65000  # GitHub's limit is 65,536; post-comment.sh adds a marker line


def write_comparison_report(path, manifest, metadata):
    """comparison.md for the PR comment. Above COMMENT_LIMIT the unchanged, then the improvements list is left out."""
    settings = metadata['settings']
    floor = settings['min_improvement_percent']
    # suite, then the name without its size, then N as a number: a string sort puts /1024 before /16
    cases = sorted(manifest['cases'], key=lambda c: (c['suite'], c['name'].rsplit('/', 1)[0], int(c['name'].rsplit('/', 1)[1])))
    measured = [c for c in cases if 'median_ratio' in c['analysis']]
    logs = {(c['suite'], c['name']): -math.log(c['analysis']['median_ratio']) for c in measured}
    by_status = {s: [c for c in cases if c['analysis']['status'] == s]
                 for s in ('regression_signal', 'improvement_signal', 'inconclusive', 'too_noisy', 'incomplete', 'added', 'removed')}
    builds = metadata['builds']
    ci = metadata['ci']

    def label(side):
        prov = builds[side]['provenance']
        name = ci[f'{side}_branch'] or ('' if prov['branch'] == 'HEAD' else prov['branch'])
        pr = f'PR #{ci["pr"]}, ' if side == 'candidate' and ci['pr'] else ''
        link = _commit_link(REPOSITORY, prov['commit'])
        return f'{name} ({pr}{link})' if name else link

    machine = metadata['machine']
    L = [f'## Benchmark comparison: candidate {label("candidate")} vs baseline {label("baseline")}', '']
    only = []
    for status, side in (('added', 'candidate'), ('removed', 'baseline')):
        if by_status[status]:
            only += _details(f'{len(by_status[status])} cases only in the {side} build (not compared)',
                             ['| Suite | Case |', '|:--|:--|'] + [_case_cells(c) for c in by_status[status]])
    if manifest.get('error'):  # nothing was measured: say why and stop
        path.write_text('\n'.join(L + [f'**Comparison failed: {manifest["error"]}**', ''] + only) + '\n')
        return
    overall = f'geometric mean {_speedup_words(math.exp(-statistics.fmean(logs.values())))}' if logs else 'no measured cases'
    L += [f'**{len(by_status["regression_signal"])} regressions, {len(by_status["improvement_signal"])} improvements, '
          f'{len(by_status["inconclusive"])} unchanged, {len(by_status["too_noisy"])} too noisy** out of {len(cases)} cases; '
          f'{overall}.', '']
    if by_status['incomplete']:
        L += [f'{len(by_status["incomplete"])} cases incomplete (a launch failed; see output.log).', '']
    L += only
    if differences := metadata.get('build_differences'):
        L += ['**The two sides were built differently**, so the speedups include the effect of these changes '
              '(`benchmark_tracked` preset variables, compiler flags and definitions, benchmark sources):', '']
        L += [f'- `{d["name"]}`: ' + (d['detail'] if 'detail' in d else f'baseline {_fmt_setting(d["baseline"])}, candidate {_fmt_setting(d["candidate"])}')
              for d in differences] + ['']
    L += [f'Speedup = candidate speed / baseline speed, median over rounds ± its standard error. A signal needs a change beyond {floor:g}% of the baseline at a '
          f'one-sided false-positive rate of {100 * settings["alpha"]:g}%. Signals do not fail CI.', '']

    # one folded list per verdict, with the count in the summary line
    regressions = sorted(by_status['regression_signal'], key=lambda c: c['analysis']['median_ratio'])
    improvements = sorted(by_status['improvement_signal'], key=lambda c: -c['analysis']['median_ratio'])
    unchanged = sorted(by_status['inconclusive'], key=lambda c: c['analysis']['median_ratio'])
    optional = {}  # lists that may be left out for the size limit; L holds a (key,) placeholder for each
    if regressions:
        L += _details(f'<b>{len(regressions)} regressions</b> (slowest first)', CASE_COLS + [_case_row(c) for c in regressions])
    if improvements:
        optional['improvements'] = _details(f'<b>{len(improvements)} improvements</b> (fastest first)',
                                            CASE_COLS + [_case_row(c) for c in improvements])
        L.append(('improvements',))
    if unchanged:
        optional['unchanged'] = _details(f'<b>{len(unchanged)} unchanged</b> (slowest first)', CASE_COLS + [_case_row(c) for c in unchanged])
        L.append(('unchanged',))
    if by_status['too_noisy']:
        L += _details(f'<b>{len(by_status["too_noisy"])} too noisy</b> (a side\'s per-round cv exceeded {settings["max_noise_percent"]:g}%; no verdict)',
                      CASE_COLS + [_case_row(c) for c in by_status['too_noisy']])

    # structured summary: one foldable table per grouping
    if measured:
        L += ['### Where the changes are', '']
        for title, key, row_order in GROUPINGS:
            L += _group_section(title, key, measured, row_order)

    reps = settings['repetitions']
    method = [
        f'- Each revision builds its own benchmarks with its own `benchmark_tracked` preset; differences are listed above. '
        f'Each case is run in **{settings["rounds"]} rounds**; a round launches '
        f'the baseline and the candidate binary once each, as separate processes on the same pinned core, alternating which goes first.',
        f'- A launch discards {settings["min_warmup_time"]} s of warm-up, then times the case for at least {settings["min_time"]}; '
        f'Google Benchmark\'s CPU time per iteration is the launch\'s value'
        + (f' (the minimum over {reps} repetitions)' if reps > 1 else '') + '. One extra warm-up launch per side is discarded.',
    ] + ([f'- **Outlier rounds**: a round where either side is {settings["outlier_sigma"]:g} or more robust standard deviations '
          '(1.4826 x median absolute deviation) from its median is left out: '
          f'{sum(len(c["analysis"].get("dropped_rounds", [])) for c in measured)} of {sum(len(c["rounds"]) for c in measured)} rounds.']
         if settings.get('outlier_sigma') else []) + [
        f'- **Speedup** is the median over rounds of baseline time / candidate time.',
        f'- **Regression / improvement**: a paired t-test on the per-round differences finds the candidate slower / faster by more than '
        f'{floor:g}% of the baseline, one-sided at a {100 * settings["alpha"]:g}% false-positive rate.',
        f'- **Too noisy**: one side\'s coefficient of variation over rounds exceeds {settings["max_noise_percent"]:g}%; no verdict.',
        '- **Unchanged**: no change beyond the floor was shown; not proof of equality.',
    ]
    L += _details('How this was measured', method)

    ctx = metadata['benchmark_context'] or {}
    env = [f'- **Machine**: {machine["cpu_model"]}, {machine["sockets"]} socket(s) x {machine["cores_per_socket"]} cores x '
           f'{machine["threads_per_core"]} threads, {_fmt_bytes(machine["ram_bytes"])} RAM, {machine["numa_nodes"]} NUMA node(s); '
           f'L1d {_fmt_bytes(machine["l1d_cache_bytes"])}, L2 {_fmt_bytes(machine["l2_cache_bytes"])}, L3 {_fmt_bytes(machine["l3_cache_bytes"])}; '
           f'{machine["system"]} {machine["release"]}.',
           f'- **Microarchitecture**: `{machine["microarchitecture"]["name"]}`, `{machine["microarchitecture"]["level"]}`.',
           f'- **CPU frequency scaling**: {"enabled" if ctx.get("cpu_scaling_enabled") else "disabled"}'
           + (f'; governor {", ".join(sorted({p["scaling_governor"] for p in settings["worker_placements"] if p["scaling_governor"]}) or ["unknown"])}' if settings.get('pinned') else '')
           + '.']
    flags = {side: builds[side]['compiler']['effective_flags'] for side in builds}
    defines = {side: builds[side]['compiler']['defines'] for side in builds}
    if flags['baseline'] == flags['candidate']:
        env.append(f'- **Compiler flags**: `{flags["candidate"]}`.')
    else:
        env.append(f'- **Compiler flags**: baseline `{flags["baseline"]}`, candidate `{flags["candidate"]}`.')
    shared = sorted(set(defines['baseline']) & set(defines['candidate']))
    line = f'- **Compile definitions**: `{" ".join(shared)}`'
    for side in ('baseline', 'candidate'):
        if only := sorted(set(defines[side]) - set(shared)):
            line += f'; {side} also `{" ".join(only)}`'
    env.append(line + '.')
    for side in ('baseline', 'candidate'):
        b = builds[side]; prov = b['provenance']
        deps = prov['dependencies']
        env.append(f'- **{side.title()}**: {_commit_link(REPOSITORY, prov["commit"])}'
                   + ('' if prov['branch'] == 'HEAD' else f' ({prov["branch"]})') + f'; {b["compiler"]["banner"]}; '
                   + 'dependencies ' + ', '.join(f'{k} {_commit_link(v["url"], v["commit"])}' for k, v in sorted(deps.items()))
                   + f'; linked against {_fmt_libraries(b["linked_libraries"])}.')
    deps_by_side = [{k: v['commit'] for k, v in builds[s]['provenance']['dependencies'].items()} for s in ('baseline', 'candidate')]
    changed = sorted(k for k in deps_by_side[0].keys() & deps_by_side[1].keys() if deps_by_side[0][k] != deps_by_side[1][k])
    if changed:  # a dependency present on one side only (a PR adding xsimd, say) is visible in the lines above
        env.append(f'- **Note**: the two sides use different commits of {", ".join(changed)} (see above).')
    if ci['url']:
        env.append(f'- **CI**: [{ci["url"]}]({ci["url"]}).')
    L += _details('Environment and build', env)

    def render():
        return '\n'.join(line for x in L for line in (optional[x[0]] if isinstance(x, tuple) else [x])) + '\n'
    for key in ('unchanged', 'improvements'):  # the case lists grow with the suite
        if key in optional and len(render()) > COMMENT_LIMIT:
            optional[key] = [f'The {key} list is left out to fit the comment size limit; '
                             'see `comparison.json` in the build artifacts.', '']
    path.write_text(render())
