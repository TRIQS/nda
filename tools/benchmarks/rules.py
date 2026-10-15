"""The paired t-test that turns per-round timings into a status.

paired_t takes two equally long lists of CPU times in nanoseconds, one per side,
where entry i of both lists comes from round i. It returns a dict with 'status'
from STATUSES and the evidence next to it. Descriptive statistics and the shared
noise gate live in analysis.py.

Run as a script to re-score an existing comparison.json with other parameters:

    python3 rules.py --alpha 0.001 --min-improvement 5 --max-noise 20 --outlier-sigma 10 path/to/comparison.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

from scipy import stats

STATUSES = ('inconclusive', 'regression_signal', 'improvement_signal', 'too_noisy')
DEFAULT_ALPHA = 0.001
DEFAULT_MIN_IMPROVEMENT = 5.0
DEFAULT_MAX_NOISE = 20.0
DEFAULT_OUTLIER_SIGMA = 10.0


def noise_percent(values):
    """Coefficient of variation of one side's per-round times, in percent."""
    return 100 * statistics.stdev(values) / statistics.mean(values)


def check_parameters(alpha, min_improvement, outlier_sigma=0.0):
    """Reject the parameters that would give wrong verdicts instead of an error."""
    if not 0 < alpha < 0.5:
        raise ValueError('alpha must be a one-sided false-positive rate strictly between 0 and 0.5')
    if not min_improvement >= 0:
        raise ValueError('min_improvement must be nonnegative')
    if not outlier_sigma >= 0:
        raise ValueError('outlier_sigma must be nonnegative')


def paired_t(baseline, candidate, alpha=DEFAULT_ALPHA, min_improvement=DEFAULT_MIN_IMPROVEMENT):
    """Paired t-test of the per-round differences (scipy ttest_rel) at one-sided level alpha.

    Rounds alternate both sides on one core, so the difference within a round cancels
    drift shared by both sides; the standard error is stdev(differences) / sqrt(rounds),
    with rounds - 1 degrees of freedom. The candidate is signalled faster (slower) when
    it beats (loses to) the baseline by more than min_improvement percent of the
    baseline mean, tested one-sided in each direction against the (1 - alpha) quantile
    of Student's t with rounds - 1 degrees of freedom, so the false-positive rate per
    direction is alpha whatever the number of rounds. The test assumes the rounds are
    sufficiently independent and measured under comparable conditions; repetitions
    inside one process are correlated and must not be counted as rounds.

    'inconclusive' means insufficient evidence, not equal performance.
    """
    rounds = len(baseline)
    required_ns = min_improvement / 100 * statistics.mean(baseline)
    differences = [b - c for b, c in zip(baseline, candidate)]
    df = rounds - 1
    cutoff_t = float(stats.t.isf(alpha, df))  # the (1 - alpha) quantile of Student's t: the one-sided cutoff
    forward = stats.ttest_rel(baseline, [c + required_ns for c in candidate], alternative='greater')
    reverse = stats.ttest_rel(candidate, [b + required_ns for b in baseline], alternative='greater')
    t_forward, t_reverse = float(forward.statistic), float(reverse.statistic)
    status = ('improvement_signal' if t_forward > cutoff_t else
              'regression_signal' if t_reverse > cutoff_t else 'inconclusive')
    return {'status': status, 'improvement_ns': statistics.mean(differences),
            'se_diff': statistics.stdev(differences) / math.sqrt(rounds), 'df': df,
            't_statistic': t_forward, 'p_value': float(forward.pvalue),
            'regression_t_statistic': t_reverse, 'regression_p_value': float(reverse.pvalue),
            'alpha': alpha, 'cutoff_t': cutoff_t,
            'min_improvement_percent': min_improvement, 'min_improvement_ns': required_ns}


def main(argv=None):
    import analysis

    parser = argparse.ArgumentParser(description='Re-score an existing comparison.json with other paired t-test parameters.')
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='one-sided false-positive rate per direction; the t cutoff follows from the degrees of freedom')
    parser.add_argument('--min-improvement', type=float, default=DEFAULT_MIN_IMPROVEMENT,
                        help='required change as a percentage of the baseline mean before a signal')
    parser.add_argument('--max-noise', type=float, default=DEFAULT_MAX_NOISE,
                        help='a side whose per-round cv exceeds this percentage makes the case too_noisy')
    parser.add_argument('--outlier-sigma', type=float, default=DEFAULT_OUTLIER_SIGMA,
                        help='drop rounds this many robust sigmas from a side\'s median (0: keep all)')
    parser.add_argument('--list', action='store_true', help='print every case that is not inconclusive')
    args = parser.parse_args(argv)
    check_parameters(args.alpha, args.min_improvement, args.outlier_sigma)
    manifest = json.loads(args.comparison.read_text())
    counts, listed = Counter(), []
    for case in manifest['cases']:
        if case['analysis']['status'] in ('added', 'removed', 'incomplete'):
            counts[case['analysis']['status']] += 1
            continue
        verdict = analysis.analyze(case['rounds'], len(case['rounds']), args.alpha, args.min_improvement, args.max_noise,
                                   args.outlier_sigma)
        counts[verdict['status']] += 1
        if verdict['status'] != 'inconclusive':
            listed.append((case['suite'], case['name'], verdict))
    print(f'paired_t alpha={100 * args.alpha:g}% min_improvement={args.min_improvement:g}% '
          f'max_noise={args.max_noise:g}% outlier_sigma={args.outlier_sigma:g}: '
          + ', '.join(f'{n} {status}' for status, n in sorted(counts.items())))
    if args.list:
        for suite, name, verdict in listed:
            t_statistic = verdict['verdict'].get('t_statistic')
            detail = f't={t_statistic:+.2f}' if t_statistic is not None else ''
            noise = ' '.join(f'{side} cv {value:.1f}%' for side, value in verdict['noise_percent'].items())
            print(f'  {suite}: {name}: {verdict["status"]} change {verdict["median_change_percent"]:+.2f}% {detail} ({noise})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
