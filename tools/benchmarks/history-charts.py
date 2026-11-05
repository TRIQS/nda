#!/usr/bin/env python3
"""Draw one chart per benchmark family from the history.json of history.py — <suite>-<op>-<operands>.png, one panel per
value type and size N — and write history.md, which shows the charts, lists the flagged steps and tabulates every case.
Needs matplotlib.

    python3 history-charts.py <results-dir>
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import charts
import common

STEP = {'regression': '#c8321e', 'improvement': '#1f8a4c'}  # the colour of a signalled step
SPEEDUPS = (0.25, 0.4, 0.5, 0.67, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.25, 1.5, 2, 3, 4, 6, 8)
Y_MARGIN = 1.08               # factor of room below and above the data

# Layout, in inches
MAX_WIDTH = 12                # 1680 px at charts.DPI: the panels wrap into rows so a chart needs no horizontal scrolling
FIGURE_MARGIN = 0.6           # added to the panels' total width
PANEL_HEIGHT = 3.1
MIN_PANEL_WIDTH = 3.2
WIDTH_PER_COMMIT = 0.22       # room for one tilted hash per commit
PANEL_FRAME_WIDTH = 1.4       # the y labels and ticks on both sides
TITLE_HEIGHT = 0.8
LEGEND_ROW_HEIGHT = 0.25
TWO_COLUMN_LEGEND_WIDTH = 8   # a figure at least this wide gets a two-column legend
MIN_TITLE_CHARS, TITLE_CHARS_PER_INCH = 30, 10  # where the title wraps, at TITLE_FONT

# Font sizes, in points
SMALL_FONT = 6                # commit hashes, dates, gap labels
TICK_FONT = 6.5
LABEL_FONT = 7.5              # axis labels, legend
PANEL_TITLE_FONT = 8.5
TITLE_FONT = 11

# Styles shared by the panels and the legend
STEP_WIDTH = 2.4
NOISY_MARKER = dict(marker='o', ms=5.5, mfc=charts.SURFACE, mec=charts.INK)
NO_MEASUREMENT = dict(marker='x', ms=6, color=charts.MUTED)


def families(cases):
    """{(suite, op, operands): {(value_type, N): case}} for the measured cases, operands in file-name form (A2+A2, M2s2)."""
    out = {}
    for case in cases:
        if not case.get('times_ns'):  # not measured (yet)
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
    y = [stats[sha]['median_ns'] / common.NS_PER_US for sha in measured]
    e = [stats[sha]['se_ns'] / common.NS_PER_US for sha in measured]
    floor = settings['min_improvement_percent'] / 100
    reference = case['reference']
    t_ref = stats[reference]['median_ns'] / common.NS_PER_US if reference in measured else None
    values = [yi - ei for yi, ei in zip(y, e)] + [yi + ei for yi, ei in zip(y, e)]
    if t_ref:
        values += [t_ref / (1 + floor), t_ref * (1 + floor)]
    lo, hi = (min(values) / Y_MARGIN, max(values) * Y_MARGIN) if values else (0, 1)

    if t_ref:
        ax.axhspan(t_ref / (1 + floor), t_ref * (1 + floor), **charts.BAND)
        ax.axhline(t_ref, **charts.REFERENCE_LINE)
    ax.plot(x, y, color=charts.INK2, lw=1, zorder=2)
    for i, sha in enumerate(measured):
        status = stats[sha].get('step', {}).get('status')
        if i and status in STEP:
            ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=STEP[status], lw=STEP_WIDTH, zorder=3)
    ax.errorbar(x, y, yerr=e, fmt='o', color=charts.INK, ms=3.5, zorder=4, **charts.WHISKER)
    for sha in measured:
        if stats[sha]['noise_percent'] > settings['max_noise_percent']:
            ax.plot(position[sha], stats[sha]['median_ns'] / common.NS_PER_US, zorder=5, **NOISY_MARKER)
    for commit in commits:
        if commit['sha'] not in measured:
            ax.plot(position[commit['sha']], lo, clip_on=False, zorder=5, **NO_MEASUREMENT)
            label = 'build failed' if commit.get('build') == 'failed' else stats.get(commit['sha'], {}).get('status', 'case absent')
            ax.annotate(label.replace('_', ' '), (position[commit['sha']], lo), xytext=(0, 5), textcoords='offset points',
                        rotation=90, ha='center', va='bottom', fontsize=SMALL_FONT, color=charts.MUTED)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, len(commits) - 0.4)
    ax.set_xticks(range(len(commits)))
    ax.set_xticklabels([c['short'] for c in commits], fontsize=SMALL_FONT, rotation=-45, ha='left', rotation_mode='anchor')
    top = ax.secondary_xaxis('top')
    top.set_xticks(range(len(commits)))
    top.set_xticklabels([c['date'] for c in commits], fontsize=SMALL_FONT, rotation=45, ha='left', rotation_mode='anchor')
    top.tick_params(axis='x', length=2, colors=charts.MUTED)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    if first_column:
        ax.set_ylabel('CPU time per iteration (µs)', fontsize=LABEL_FONT)
    if t_ref:
        twin = ax.twinx()
        twin.set_ylim(lo, hi)
        twin.grid(False)
        twin.spines['right'].set_visible(True)
        ticks = [(t_ref / s, s) for s in SPEEDUPS if lo <= t_ref / s <= hi]
        twin.set_yticks([t for t, _ in ticks])
        twin.set_yticklabels([f'{s:g}x' for _, s in ticks], fontsize=TICK_FONT)
        if last_column:
            # the reference's tick label: git's short hash, which grows past 7 digits in a large repository
            twin.set_ylabel(f"speedup vs {commits[position[reference]]['short']}", fontsize=LABEL_FONT)
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FONT, loc='left')


def draw_family(path, family, panels, commits, settings):
    """One figure per family, one panel per case's history: by value type, then size N, in as many columns as fit
    MAX_WIDTH. A factor shared by every panel (the only value type, the only N) is named in the title; a panel is titled
    with what varies."""
    plt = charts.plt
    suite, op, _ = family
    value_types = sorted({vt for vt, _ in panels}, key=common.value_type_key)
    sizes = sorted({n for _, n in panels})
    operands = common.case_factors(*[(c['suite'], c['name']) for c in panels.values()][0])['operands']
    order = [(vt, n) for vt in value_types for n in sizes if (vt, n) in panels]
    width = max(MIN_PANEL_WIDTH, WIDTH_PER_COMMIT * len(commits) + PANEL_FRAME_WIDTH)
    ncol = max(1, min(len(order), int((MAX_WIDTH - FIGURE_MARGIN) // width)))
    nrow = -(-len(order) // ncol)
    figure_width = width * ncol + FIGURE_MARGIN
    handles = [plt.Line2D([], [], color=colour, lw=STEP_WIDTH, label=f'{status} vs previous commit') for status, colour in STEP.items()]
    handles.append(plt.Line2D([], [], ls='', label='too noisy for a verdict', **NOISY_MARKER))
    handles.append(plt.Line2D([], [], ls='', label='not measured (build failed, launch failed or case absent)', **NO_MEASUREMENT))
    legend_columns = 2 if figure_width >= TWO_COLUMN_LEGEND_WIDTH else 1
    legend_height = LEGEND_ROW_HEIGHT * -(-len(handles) // legend_columns)
    fig = plt.figure(figsize=(figure_width, PANEL_HEIGHT * nrow + legend_height + TITLE_HEIGHT), constrained_layout=True)
    grid = fig.add_gridspec(nrow + 1, ncol, height_ratios=[PANEL_HEIGHT] * nrow + [legend_height])
    for index, (vt, n) in enumerate(order):
        ax = fig.add_subplot(grid[index // ncol, index % ncol])
        column = index % ncol
        varying = [common.value_type_label(vt)] * (len(value_types) > 1) + [f'N = {n}'] * (len(sizes) > 1)
        draw_panel(ax, panels[vt, n], commits, settings, ', '.join(varying),
                   first_column=column == 0, last_column=column == ncol - 1 or index == len(order) - 1)
    floor = settings['min_improvement_percent']
    shared = [common.value_type_label(value_types[0])] * (len(value_types) == 1) + [f'N = {sizes[0]}'] * (len(sizes) == 1)
    title = ', '.join([f'{suite}: {op}({operands})'] + shared)
    title = textwrap.fill(title, width=max(MIN_TITLE_CHARS, int(figure_width * TITLE_CHARS_PER_INCH)))
    fig.suptitle(f'{title}\ngrey = within {floor:g}% of the first commit', fontsize=TITLE_FONT)
    # A separate grid row lets Matplotlib lay out the legend without measuring text by hand.
    legend_ax = fig.add_subplot(grid[-1, :])
    legend_ax.axis('off')
    legend_ax.legend(handles=handles, loc='center', ncol=legend_columns, frameon=False, fontsize=LABEL_FONT)
    fig.savefig(path, dpi=charts.DPI)
    plt.close(fig)


def _cell(stat, gate):
    """median µs, with the step verdict when it signals and a mark when the commit was too noisy."""
    if stat is None or stat['status'] != 'measured':
        return '—'
    text = f"{stat['median_ns'] / common.NS_PER_US:.3g}"
    step = stat.get('step')
    if step and step['status'] in STEP:
        text += f" ({step['status']} {step['speedup']:.2f}x)"
    if stat['noise_percent'] > gate:
        text += ' (noisy)'
    return text


def write_page(path, history, images):
    commits, settings, machine = history['commits'], history['settings'], history['machine']
    short = {c['sha']: c['short'] for c in commits}
    failed = sum(c.get('build') == 'failed' for c in commits)
    failed_note = f' ({failed} failed to build)' if failed else ''
    summary = (
        f"{len(commits)} commits{failed_note}, {len(images)} families; "
        f"{machine['cpu_model']}; {settings['rounds']} launches per commit and case, {settings['min_time']} each. "
        f"Speedups are baseline time / candidate time, median over the launches; a signal needs a change beyond "
        f"{settings['min_improvement_percent']:g}% at a one-sided false-positive rate of {100 * settings['alpha']:g}%; "
        f"a commit whose launches vary by more than {settings['max_noise_percent']:g}% (cv) is too noisy for a verdict."
    )
    L = [f"# Benchmark history: {commits[0]['short']} .. {commits[-1]['short']}", '', summary, '',
         '| commit | date | subject | build |', '|:--|:--|:--|:--|']
    for c in commits:
        note = c.get('build', '')
        if c.get('build_differences'):
            note += '; ' + '; '.join(f'{name}: {detail}' for name, detail in c['build_differences'].items())
        L.append(f"| {c['short']} | {c['date']} | {c['subject']} | {note}{(': ' + c['error']) if c.get('error') else ''} |")
    L.append('')

    steps = []
    for _, panels, _ in images:
        for case in panels.values():
            for sha, stat in case['per_commit'].items():
                step = stat.get('step')
                if step and step['status'] in STEP:
                    steps.append((step['t'], case, sha, step))
    L += ['## Steps', '']
    if steps:
        L += ['Commits whose paired test against the previous measured commit signals, largest t first.', '',
              '| case | commit | vs previous | t |', '|:--|:--|--:|--:|']
        for t, case, sha, step in sorted(steps, key=lambda s: -s[0]):
            L.append(f"| {case['suite']} {case['name']} | {short[sha]} | {step['speedup']:.2f}x ({step['status']}) | {t:.1f} |")
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
    parser.add_argument('results', type=Path, help='directory with the history.json of history.py')
    results = parser.parse_args(argv).results
    history = json.loads((results / 'history.json').read_text())
    images = []
    for family, panels in families(history['cases']).items():
        path = results / ('-'.join(family) + '.png')
        draw_family(path, family, panels, history['commits'], history['settings'])
        images.append((family, panels, path.name))
        print(path)
    write_page(results / 'history.md', history, images)
    print(results / 'history.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
