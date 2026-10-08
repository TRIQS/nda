#!/usr/bin/env python3
"""Draw one bar chart per benchmark suite (benchmark-<suite>.png) from a comparison, and charts.md showing them.
Needs matplotlib."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import common

# One colour per value type, in the order of common.VALUE_TYPES: a type keeps its colour when others are added.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#e87ba4', '#008300']
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
STYLE = {'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE,
         'axes.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True}


def pyplot():
    """matplotlib.pyplot in the chart style, imported on first use so that --help works without matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt


def side_label(metadata, side):
    """The branch (as in the comment header) and short commit of one side."""
    provenance = metadata['builds'][side]['provenance']
    name = metadata['ci'][f'{side}_branch'] or ('' if provenance['branch'] == 'HEAD' else provenance['branch'])
    return f'{name} ({provenance["commit"][:10]})' if name else provenance['commit'][:10]


def speedups(manifest):
    """{suite: {(op, operands, value_type, N): (median speedup, standard error of the log speedup)}}"""
    suites = {}
    for case in manifest['cases']:
        if 'median_ratio' not in case['analysis']:  # incomplete, added or removed: nothing to draw
            continue
        dropped = set(case['analysis'].get('dropped_rounds', ()))  # outlier rounds
        logs = [math.log(r['samples']['baseline']['cpu_time_ns'] / r['samples']['candidate']['cpu_time_ns'])
                for r in case['rounds'] if r['round'] not in dropped]
        f = common.case_factors(case['suite'], case['name'])
        suites.setdefault(f['suite'], {})[(f['op'], f['operands'], f['value_type'], f['N'])] = (
            case['analysis']['median_ratio'], statistics.stdev(logs) / math.sqrt(len(logs)))
    return suites


def log_ticks(ax, lo, hi):
    ticks = [t for t in (0.25, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}x' for t in ticks])
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())


def value_type_styles(suites):
    """{tag: (colour, legend label)} for every value type in the comparison, in the order of common.VALUE_TYPES."""
    present = {vt for cases in suites.values() for _, _, vt, _ in cases}
    order = sorted(common.VALUE_TYPES.keys() | present, key=common.value_type_key)  # listed types keep their slot
    return {vt: (PALETTE[i], common.value_type_label(vt)) for i, vt in enumerate(order) if vt in present}


def draw_suite(path, suite, cases, styles, title, floor):
    """One panel per operation and operand kind; bars = median speedup (baseline time / candidate time, log scale)
    per value type and size N, whiskers = ± standard error of the per-round log ratio, grey band = no-change floor."""
    plt = pyplot()
    panels = sorted({(op, operands) for op, operands, _, _ in cases})
    sizes = sorted({size for *_, size in cases})
    value_types = [vt for vt in styles if any(key[2] == vt for key in cases)]
    step = 0.76 / len(value_types)  # the bars of one size share 0.76 of the slot
    values = [speedup for speedup, _ in cases.values()]
    lo, hi = min(0.6, min(values) / 1.2), max(24, max(values) * 1.6)  # one scale for the suite, room for the labels
    widen = max(1, len(value_types) / 2)  # more value types: wider panels, fewer per row, same room per bar label
    ncol = max(1, min(len(panels), round(4 / widen)))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * widen * ncol + 0.8, 3.3 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis('off')
    for ax, (op, operands) in zip(axes.flat, panels):
        for j, vt in enumerate(value_types):
            x = [i + (j - (len(value_types) - 1) / 2) * step for i in range(len(sizes))]
            for xi, size in zip(x, sizes):
                if (op, operands, vt, size) not in cases:
                    continue
                speedup, se = cases[op, operands, vt, size]
                low, high = speedup - speedup * math.exp(-se), speedup * math.exp(se) - speedup
                ax.bar(xi, speedup, width=0.9 * step, color=styles[vt][0], zorder=2)
                ax.errorbar(xi, speedup, yerr=[[low], [high]], fmt='none', ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
                ax.annotate(f'{speedup:.2f}\n±{max(low, high):.2f}', (xi, speedup + high), textcoords='offset points',
                            xytext=(0, 2), ha='center', va='bottom', fontsize=6.3, color=INK2, linespacing=1.0)
        ax.set_yscale('log')
        ax.axhspan(1 / (1 + floor), 1 + floor, color=GRID, alpha=0.7, lw=0, zorder=0)  # the no-change band
        ax.axhline(1, color=AXIS, lw=1, zorder=1)
        ax.set_ylim(lo, hi)
        log_ticks(ax, lo, hi)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], fontsize=7.5)
        ax.set_title(f'{op}({operands})', fontsize=9.5)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='y', labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('speedup (log)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=styles[vt][0], label=styles[vt][1]) for vt in value_types]
    height = fig.get_figheight()  # title, then the legend below it, top left; offsets in inches
    fig.suptitle(f'{title}, Benchmark: {suite}', x=0.01, y=1 - 0.1 / height, ha='left', va='top', fontsize=10, fontweight='bold')
    fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.005, 1 - 0.32 / height), ncol=len(handles), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='example:\n  python3 tools/benchmarks/charts.py benchmark-results')
    parser.add_argument('results', type=Path,
                        help='directory with the comparison.json and metadata.json of one comparison; '
                             'the images and charts.md are written there')
    results = parser.parse_args(argv).results
    manifest = json.loads((results / 'comparison.json').read_text())
    metadata = json.loads((results / 'metadata.json').read_text())
    candidate, baseline = side_label(metadata, 'candidate'), side_label(metadata, 'baseline')
    title = f'{candidate} vs {baseline}'
    floor = metadata['settings']['min_improvement_percent'] / 100
    suites = speedups(manifest)
    if not suites:
        failed = f' (the comparison failed: {manifest["error"]})' if manifest.get('error') else ''
        sys.exit(f'{results}: no measured cases, nothing to draw{failed}')
    styles = value_type_styles(suites)
    # charts.md links the images by file name; push-charts.sh points the links at the uploaded copies for the PR comment.
    page = [f'## Benchmark charts: candidate {candidate} vs baseline {baseline}', '',
            'Median speedup (baseline time / candidate time) per value type and size N, ± standard error; '
            f'grey band = no change (within {100 * floor:g}%).', '']
    for suite, cases in sorted(suites.items()):
        path = results / f'benchmark-{suite}.png'
        draw_suite(path, suite, cases, styles, title, floor)
        page += [f'![{suite}]({path.name})', '']
        print(path)
    (results / 'charts.md').write_text('\n'.join(page))
    print(results / 'charts.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
