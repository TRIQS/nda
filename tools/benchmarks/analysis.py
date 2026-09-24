"""Build compatibility checks and paired performance analysis."""

import statistics

import rules


def check_comparable(baseline, candidate):
    # Each build resolves dependencies independently; changed pins are part of the comparison.
    if baseline['harness_sha256'] != candidate['harness_sha256']:
        raise ValueError('The two builds have different harness_sha256')
    for key in ('vendor', 'version', 'build_type', 'effective_flags'):
        if baseline['compiler'].get(key) != candidate['compiler'].get(key):
            raise ValueError(f'The two builds have different compiler {key}')


def analyze(rounds, expected_rounds, threshold, min_improvement, max_noise, rule='paired_t'):
    """Descriptive paired statistics plus the status decided by the named rule in rules.RULES.

    A side whose per-round coefficient of variation exceeds max_noise percent makes the
    case too_noisy before the rule runs: the launches disagree too much to decide either way.
    """
    if len(rounds) != expected_rounds or any(
            set(pair['samples']) != {'baseline', 'candidate'} or
            any(sample.get('error') for sample in pair['samples'].values()) for pair in rounds):
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
        verdict = rules.RULES[rule](baseline, candidate, threshold, min_improvement)
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
