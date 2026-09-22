"""Build compatibility checks and paired performance analysis."""

import statistics


def check_comparable(baseline, candidate):
    # Dependency revisions are recorded in each side's provenance but not required to
    # match: the script shares them across arms in CI, and a PR that changes a pin is
    # measuring that change.
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
    changes = [100 * (1 / ratio - 1) for ratio in ratios]
    ratio = statistics.median(ratios)
    change = statistics.median(changes)
    low, high = min(changes), max(changes)
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
