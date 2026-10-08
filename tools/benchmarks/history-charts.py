#!/usr/bin/env python3
"""Draw one chart per benchmark family from the history.json of history.py — <suite>-<op>-<operands>.png, one panel per
value type and size N — and write history.md, which shows the charts, lists the flagged steps and tabulates every case.
Needs matplotlib.

    python3 history-charts.py <results-dir>
"""

import argparse
import json
import sys
from pathlib import Path

import charts
import common

STEP = {'regression_signal': ('#c8321e', 'regression'), 'improvement_signal': ('#1f8a4c', 'improvement')}
LABEL = {'regression_signal': 'regression', 'improvement_signal': 'improvement', 'inconclusive': '', 'too_noisy': 'too noisy',
         'incomplete': 'incomplete'}
SPEEDUPS = (0.25, 0.4, 0.5, 0.67, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.25, 1.5, 2, 3, 4, 6, 8)
MAX_WIDTH = 12  # inches (1680 px at 140 dpi): the panels wrap into rows so a chart needs no horizontal scrolling


def families(cases):
    """{(suite, op, operands): {(value_type, N): case}} for the measured cases, operands in file-name form (A2+A2, M2s2)."""
    out = {}
    for case in cases:
        if not case.get('rounds'):  # not measured (yet)
            continue
        f = common.case_factors(case['suite'], case['name'])
        operands = f['operands'].replace(' sliced', 's2').replace(', ', '+')
        out.setdefault((f['suite'], f['op'], operands), {})[(f['value_type'], f['N'])] = case
    return dict(sorted(out.items()))


def draw_panel(ax, case, commits, settings, title, first_column, last_column):
    """Points = median CPU time per iteration per commit, whiskers = ± s.e.; the segment into a commit is coloured when the
    paired test against the previous commit signals; grey band = ± the floor around the reference commit, whose time also
    labels the right axis in speedups; hollow marker = too noisy for a verdict; × = not measured at that commit."""
    position = {c['sha']: i for i, c in enumerate(commits)}
    stats = case['per_commit']
    measured = [sha for sha in case['commits'] if stats[sha]['status'] == 'measured']
    x = [position[sha] for sha in measured]
    y = [stats[sha]['median_ns'] / 1000 for sha in measured]
    e = [stats[sha]['se_ns'] / 1000 for sha in measured]
    floor = settings['min_improvement_percent'] / 100
    reference = case['reference']
    t_ref = stats[reference]['median_ns'] / 1000 if reference in measured else None
    values = [yi - ei for yi, ei in zip(y, e)] + [yi + ei for yi, ei in zip(y, e)]
    if t_ref:
        values += [t_ref / (1 + floor), t_ref * (1 + floor)]
    lo, hi = (min(values) / 1.08, max(values) * 1.08) if values else (0, 1)

    if t_ref:
        ax.axhspan(t_ref / (1 + floor), t_ref * (1 + floor), color=charts.GRID, alpha=0.7, lw=0, zorder=0)
        ax.axhline(t_ref, color=charts.AXIS, lw=1, zorder=1)
    ax.plot(x, y, color=charts.INK2, lw=1, zorder=2)
    for i, sha in enumerate(measured):
        status = stats[sha].get('step', {}).get('status')
        if i and status in STEP:
            ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=STEP[status][0], lw=2.4, zorder=3)
    ax.errorbar(x, y, yerr=e, fmt='o', color=charts.INK, ms=3.5, ecolor=charts.INK2, elinewidth=0.8, capsize=2, zorder=4)
    for sha in measured:
        if stats[sha]['noise_percent'] > settings['max_noise_percent']:
            ax.plot(position[sha], stats[sha]['median_ns'] / 1000, 'o', ms=5.5, mfc=charts.SURFACE, mec=charts.INK, zorder=5)
    for commit in commits:
        if commit['sha'] not in measured:
            ax.plot(position[commit['sha']], lo, marker='x', color=charts.MUTED, ms=6, clip_on=False, zorder=5)
            if commit.get('build') == 'skipped':
                continue
            label = 'build failed' if commit.get('build') == 'failed' else stats.get(commit['sha'], {}).get('status', 'case absent')
            ax.annotate(label.replace('_', ' '), (position[commit['sha']], lo), xytext=(0, 5), textcoords='offset points',
                        rotation=90, ha='center', va='bottom', fontsize=6, color=charts.MUTED)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, len(commits) - 0.4)
    ax.set_xticks(range(len(commits)))
    ax.set_xticklabels([c['short'] for c in commits], fontsize=6, rotation=-45, ha='left', rotation_mode='anchor')
    top = ax.secondary_xaxis('top')
    top.set_xticks(range(len(commits)))
    top.set_xticklabels([c['date'] for c in commits], fontsize=6, rotation=45, ha='left', rotation_mode='anchor')
    top.tick_params(axis='x', length=2, colors=charts.MUTED)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='y', labelsize=6.5)
    if first_column:
        ax.set_ylabel('CPU time per iteration (µs)', fontsize=7.5)
    if t_ref:
        twin = ax.twinx()
        twin.set_ylim(lo, hi)
        twin.grid(False)
        twin.spines['right'].set_visible(True)
        ticks = [(t_ref / s, s) for s in SPEEDUPS if lo <= t_ref / s <= hi]
        twin.set_yticks([t for t, _ in ticks])
        twin.set_yticklabels([f'{s:g}x' for _, s in ticks], fontsize=6.5)
        if last_column:
            twin.set_ylabel(f"speedup vs {reference[:7]}", fontsize=7.5)
    if title:
        ax.set_title(title, fontsize=8.5, loc='left')


def draw_family(path, family, panels, commits, settings):
    """One figure per family, one panel per case's history: by value type, then size N, in as many columns as fit
    MAX_WIDTH. A factor shared by every panel (the only value type, the only N) is named in the title; a panel is titled
    with what varies."""
    plt = charts.pyplot()
    suite, op, _ = family
    value_types = sorted({vt for vt, _ in panels}, key=common.value_type_key)
    sizes = sorted({n for _, n in panels})
    operands = common.case_factors(*[(c['suite'], c['name']) for c in panels.values()][0])['operands']
    order = [(vt, n) for vt in value_types for n in sizes if (vt, n) in panels]
    width = max(3.2, 0.22 * len(commits) + 1.4)  # room for one tilted hash per commit
    ncol = max(1, min(len(order), int((MAX_WIDTH - 0.6) // width)))
    nrow = -(-len(order) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(width * ncol + 0.6, 3.1 * nrow), squeeze=False)
    for ax in axes.flat[len(order):]:
        ax.axis('off')
    for index, (vt, n) in enumerate(order):
        column = index % ncol
        varying = [common.value_type_label(vt)] * (len(value_types) > 1) + [f'N = {n}'] * (len(sizes) > 1)
        draw_panel(axes.flat[index], panels[vt, n], commits, settings, ', '.join(varying),
                   first_column=column == 0, last_column=column == ncol - 1 or index == len(order) - 1)
    floor = settings['min_improvement_percent']
    # The title, the subtitle and the legend are measured and the figure is extended by what they need: a narrow figure
    # (few sizes) wraps the texts and stacks the legend entries, a wide one lays everything out in a line.
    shared = [common.value_type_label(value_types[0])] * (len(value_types) == 1) + [f'N = {sizes[0]}'] * (len(sizes) == 1)
    title = ', '.join(part.replace(' ', '\u00a0') for part in [f'{suite}: {op}({operands})'] + shared)  # breaks only at the commas
    title = wrap_to_width(fig, title, 0.98 * fig.get_figwidth(), fontsize=11, fontweight='bold')
    subtitle = wrap_to_width(fig, f"grey = within {floor:g}% of the first commit", 0.98 * fig.get_figwidth(), fontsize=7.5)
    handles = [plt.Line2D([], [], color=colour, lw=2.4, label=f'{label} vs previous commit') for colour, label in STEP.values()]
    handles.append(plt.Line2D([], [], marker='o', ls='', ms=5.5, mfc=charts.SURFACE, mec=charts.INK, label='too noisy for a verdict'))
    handles.append(plt.Line2D([], [], marker='x', ls='', ms=6, color=charts.MUTED, label='not measured (skipped, build failed or case absent)'))
    legend, legend_height = fit_legend(fig, handles, 7.5)
    subtitle_top = 0.29 + 0.183 * (title.count('\n') + 1)  # inches from the top: below the title's lines
    legend_top = subtitle_top + 0.125 * (subtitle.count('\n') + 1) + 0.05  # below the subtitle's lines
    top = legend_top + legend_height + 0.15  # and a gap above the panels
    height = fig.get_figheight() + top + 0.1
    fig.set_figheight(height)
    fig.suptitle(title, x=0.01, y=1 - 0.3 / height, ha='left', fontsize=11, fontweight='bold', in_layout=False)  # the rect has its room
    fig.text(0.01, 1 - subtitle_top / height, subtitle, va='top', fontsize=7.5, color=charts.INK2)
    legend.set_bbox_to_anchor((0.005, 1 - legend_top / height), transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0.1 / height, 1, 1 - top / height))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def wrap_to_width(fig, text, width, **font):
    """text broken at spaces into lines no wider than width inches in the given font, measured with the figure's renderer."""
    renderer, probe = fig.canvas.get_renderer(), fig.text(0, 0, '', **font)
    words, lines = text.split(' '), []
    while words:
        count = len(words)
        while count > 1:
            probe.set_text(' '.join(words[:count]))
            if probe.get_window_extent(renderer).width <= width * fig.dpi:
                break
            count -= 1
        lines.append(' '.join(words[:count]))
        words = words[count:]
    probe.remove()
    return '\n'.join(lines)


def fit_legend(fig, handles, fontsize):
    renderer = fig.canvas.get_renderer()
    for columns in range(len(handles), 0, -1):
        legend = fig.legend(handles=handles, frameon=False, loc='upper left', ncol=columns, fontsize=fontsize)
        extent = legend.get_window_extent(renderer)
        if extent.width <= fig.get_figwidth() * fig.dpi or columns == 1:
            return legend, extent.height / fig.dpi
        legend.remove()


def _cell(stat, gate):
    """median µs, with the step verdict when it signals and a mark when the commit was too noisy."""
    if stat is None or stat['status'] != 'measured':
        return '—'
    text = f"{stat['median_ns'] / 1000:.3g}"
    step = stat.get('step')
    if step and step['status'] in STEP:
        text += f" ({LABEL[step['status']]} {step['median_ratio']:.2f}x)"
    if stat['noise_percent'] > gate:
        text += ' (noisy)'
    return text


def write_page(path, history, metadata, images):
    commits, settings, machine = history['commits'], history['settings'], metadata['machine']
    short = {c['sha']: c['short'] for c in commits}
    failed = [c for c in commits if c.get('build') == 'failed']
    skipped = [c for c in commits if c.get('build') == 'skipped']
    notes = ([f'{len(failed)} failed to build'] if failed else []) + \
            ([f'{len(skipped)} skipped for lack of a benchmark_tracked preset'] if skipped else [])
    L = [f"# Benchmark history: {commits[0]['short']} .. {commits[-1]['short']}", '',
         f"{len(commits)} commits" + (f' ({", ".join(notes)})' if notes else '') + f", {len(images)} families; "
         f"{machine['cpu_model']}; {settings['rounds']} launches per commit and case, {settings['min_time']} each. "
         f"Speedups are baseline time / candidate time, median over the launches; a signal needs a change beyond "
         f"{settings['min_improvement_percent']:g}% at a one-sided false-positive rate of {100 * settings['alpha']:g}%; "
         f"a commit whose launches vary by more than {settings['max_noise_percent']:g}% (cv) is too noisy for a verdict.", '',
         '| commit | date | subject | build |', '|:--|:--|:--|:--|']
    for c in commits:
        note = c.get('build', '')
        if c.get('build_differences'):
            note += '; ' + '; '.join(d['name'] + (f" ({d['detail']})" if 'detail' in d else f": {d['baseline']} -> {d['candidate']}")
                                    for d in c['build_differences'])
        L.append(f"| {c['short']} | {c['date']} | {c['subject']} | {note}{(': ' + c['error']) if c.get('error') else ''} |")
    L.append('')

    steps = []
    for family, panels, image in images:
        for case in panels.values():
            for sha, stat in case['per_commit'].items():
                step = stat.get('step')
                if step and step['status'] in STEP:
                    verdict = step['verdict']
                    t = verdict['t_statistic'] if step['status'] == 'improvement_signal' else verdict['regression_t_statistic']
                    steps.append((abs(t), case, sha, step))
    L += ['## Steps', '']
    if steps:
        L += ['Commits whose paired test against the previous measured commit signals, largest t first.', '',
              '| case | commit | vs previous | t |', '|:--|:--|--:|--:|']
        for t, case, sha, step in sorted(steps, key=lambda s: -s[0]):
            L.append(f"| {case['suite']} {case['name']} | {short[sha]} | {step['median_ratio']:.2f}x ({LABEL[step['status']]}) | {t:.1f} |")
    else:
        L += ['No step was signalled.']
    L.append('')

    gate = settings['max_noise_percent']
    for (suite, op, _), panels, image in images:
        keys = sorted(panels, key=lambda k: (common.value_type_key(k[0]), k[1]))
        operands = common.case_factors(*[(c['suite'], c['name']) for c in panels.values()][0])['operands']
        L += [f'## {suite}: {op}({operands})', '', f'![{suite} {op}({operands})]({image})', '',
              'Median CPU time per iteration in µs.', '',
              '| commit | ' + ' | '.join(f'{vt} N={n}' for vt, n in keys) + ' |', '|:--|' + '--:|' * len(keys)]
        for c in commits:
            L.append(f"| {c['short']} | " + ' | '.join(_cell(panels[k]['per_commit'].get(c['sha']), gate) for k in keys) + ' |')
        L.append('')
    path.write_text('\n'.join(L))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path, help='directory with the history.json and metadata.json of history.py')
    results = parser.parse_args(argv).results
    history = json.loads((results / 'history.json').read_text())
    metadata = json.loads((results / 'metadata.json').read_text())
    images = []
    for family, panels in families(history['cases']).items():
        path = results / ('-'.join(family) + '.png')
        draw_family(path, family, panels, history['commits'], history['settings'])
        images.append((family, panels, path.name))
        print(path)
    write_page(results / 'history.md', history, metadata, images)
    print(results / 'history.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
