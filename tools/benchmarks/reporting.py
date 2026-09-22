"""Console summaries and Markdown comparison reports."""

import common


def print_run_summary(summary, results, outdir, *, count_only=False):
    totals = summary["totals"]
    settings = summary["settings"]
    workers = settings["workers"]
    total_secs = totals["wall_seconds"]
    print(f"\nfamilies   {totals['families']}   (distinct benchmarks, size sweep collapsed)")
    print(f"cases      {totals['cases']}   (registered cases, one per size)")
    print(f"wall       {total_secs}s at {settings['repetitions']} repetition(s)")
    print(f"results    {outdir}")

    if count_only:
        return

    # GB reports only its final timed run per case; the gap to the wall clock is mostly
    # GB's own iteration-count search (it tries 1, 10, 100 ... and discards each trial),
    # plus input generation, which only matters at the largest size.
    measured = sum(common.measured_seconds(entry.get('benchmark', {}))
                   for entry in results['binaries'] if not entry.get('error') and entry.get('exit_code') == 0)
    if workers > 1:
        print(f"measured   {measured:.1f}s summed across parallel workers (not a fraction of wall time)")
    elif measured and total_secs:
        print(f"measured   {measured:.1f}s in timed loops "
              f"({100 * measured / total_secs:.0f}% of wall; the rest is GB's "
              f"iteration search and input generation)")


def write_comparison_report(path, manifest, metadata):
    lines = ['# NDA paired benchmark comparison', '']
    if counts := manifest.get('counts'):
        lines += ['Cases: ' + ', '.join(f'{n} {status}' for status, n in sorted(counts.items())), '']
    for side, build in metadata['builds'].items():
        lines.append(f'{side.title()}: `{build["provenance"]["commit"]}`')
    lines += ['', 'Ratio = baseline / candidate latency; 1.00x is equal, greater than 1 is faster.',
              'Bands are observed min/max ranges, not confidence intervals. Signals do not fail CI.',
              f'Each invocation uses the minimum CPU time of {metadata["settings"]["repetitions"]} Google Benchmark repetition(s).',
              f'Rounds: {metadata["settings"]["rounds"]}; threshold: {metadata["settings"]["threshold_percent"]:g}%.', '']
    if manifest.get('error'):
        lines += [f'Error: {manifest["error"]}', '']
    lines += ['| Suite | Case | Median ratio | Observed range | Status |', '|---|---|---:|---:|---|']
    for case in sorted(manifest['cases'], key=lambda c: (c['suite'], c['name'])):
        analysis = case['analysis']
        ratio, band = '—', '—'
        if 'median_ratio' in analysis:
            ratio = f'{analysis["median_ratio"]:.4f}x'
            band = '–'.join(f'{v:.4f}x' for v in analysis['ratio_range'])
        name = case['name'].replace('|', '\\|')
        lines.append(f'| {case["suite"]} | {name} | {ratio} | {band} | {analysis["status"]} |')
    path.write_text('\n'.join(lines) + '\n')
