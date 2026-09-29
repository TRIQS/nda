"""Build compatibility checks and paired performance analysis."""

import statistics

import rules


def check_comparable(baseline, candidate):
    """Raise when the two builds cannot be compared, else return how their configurations differ.

    Each side builds its own tracked benchmarks with its own benchmark_tracked preset, so a changed
    benchmark source, preset or compiler flag is part of the comparison, like a changed dependency
    pin; the report lists it.
    """
    for key in ('vendor', 'version'):
        if baseline['compiler'][key] != candidate['compiler'][key]:
            raise ValueError(f'The two builds have different compiler {key}')
    presets = baseline['preset'], candidate['preset']
    differences = [{'name': name, 'baseline': presets[0].get(name), 'candidate': presets[1].get(name)}
                   for name in sorted(presets[0].keys() | presets[1].keys()) if presets[0].get(name) != presets[1].get(name)]
    flags = baseline['compiler']['effective_flags'], candidate['compiler']['effective_flags']
    if flags[0] != flags[1]:
        differences.append({'name': 'effective compiler flags', 'baseline': flags[0], 'candidate': flags[1]})
    if baseline['harness_sha256'] != candidate['harness_sha256']:
        differences.append({'name': 'benchmark sources (benchmarks/tracked, sha256)',
                            'baseline': baseline['harness_sha256'][:12], 'candidate': candidate['harness_sha256'][:12]})
    return differences


def analyze(rounds, expected_rounds, alpha, min_improvement, max_noise):
    """Descriptive paired statistics plus the status decided by rules.paired_t.

    A side whose per-round coefficient of variation exceeds max_noise percent makes the
    case too_noisy before the test runs: the launches disagree too much to decide either way.
    """
    if len(rounds) != expected_rounds or any(sample.get('error') for pair in rounds for sample in pair['samples'].values()):
        return {'status': 'incomplete'}
    baseline = [pair['samples']['baseline']['cpu_time_ns'] for pair in rounds]
    candidate = [pair['samples']['candidate']['cpu_time_ns'] for pair in rounds]
    ratios = [b / c for b, c in zip(baseline, candidate)]
    # One estimate, in the measured unit; the percentage is derived from it. A median of
    # six averages the middle pair, and mean(1/x) != 1/mean(x), so a separately computed
    # median of the changes would disagree with this one.
    ratio = statistics.median(ratios)
    noise = {'baseline': rules.noise_percent(baseline), 'candidate': rules.noise_percent(candidate)}
    if max(noise.values()) > max_noise:
        verdict = {'status': 'too_noisy', 'max_noise_percent': max_noise}
    else:
        verdict = rules.paired_t(baseline, candidate, alpha, min_improvement)
    return {
        'status': verdict['status'],
        'median_ratio': ratio,
        'ratio_range': [min(ratios), max(ratios)],
        'median_change_percent': 100 * (1 / ratio - 1),
        'change_percent_range': [100 * (1 / max(ratios) - 1), 100 * (1 / min(ratios) - 1)],
        'baseline_median_ns': statistics.median(baseline),
        'candidate_median_ns': statistics.median(candidate),
        'noise_percent': noise,
        'verdict': verdict,
    }
