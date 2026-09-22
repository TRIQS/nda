"""Build compatibility checks and paired performance analysis."""

import statistics


def check_comparable(baseline, candidate):
    # Each build resolves dependencies independently; changed pins are part of the comparison.
    if baseline['harness_sha256'] != candidate['harness_sha256']:
        raise ValueError('The two builds have different harness_sha256')
    for key in ('vendor', 'version', 'build_type', 'effective_flags'):
        if baseline['compiler'].get(key) != candidate['compiler'].get(key):
            raise ValueError(f'The two builds have different compiler {key}')


def analyze(rounds, expected_rounds, threshold):
    if len(rounds) != expected_rounds or any(
            set(pair['samples']) != {'baseline', 'candidate'} or
            any(sample.get('error') for sample in pair['samples'].values()) for pair in rounds):
        return {'status': 'incomplete'}
    ratios = [pair['samples']['baseline']['cpu_time_ns'] /
              pair['samples']['candidate']['cpu_time_ns'] for pair in rounds]
    # One estimate, in the measured unit; the percentage is derived from it. A median of
    # six averages the middle pair, and mean(1/x) != 1/mean(x), so a separately computed
    # median of the changes would disagree with this one.
    ratio = statistics.median(ratios)
    change = 100 * (1 / ratio - 1)
    low, high = 100 * (1 / max(ratios) - 1), 100 * (1 / min(ratios) - 1)
    if low <= 0 <= high:
        status = 'inconclusive'
    elif low > 0 and change >= threshold:
        status = 'regression_signal'
    elif high < 0 and change <= -threshold:
        status = 'improvement_signal'
    else:
        status = 'below_threshold'
    return {
        'status': status,
        'median_ratio': ratio,
        'ratio_range': [min(ratios), max(ratios)],
        'median_change_percent': change,
        'change_percent_range': [low, high],
        'baseline_median_ns': statistics.median(pair['samples']['baseline']['cpu_time_ns'] for pair in rounds),
        'candidate_median_ns': statistics.median(pair['samples']['candidate']['cpu_time_ns'] for pair in rounds),
    }
