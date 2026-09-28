"""Decision rules that turn per-round timings into a status.

A rule takes two equally long lists of CPU times in nanoseconds, one per side,
where entry i of both lists comes from round i. It returns a dict with at least
'status' from STATUSES; the evidence goes next to it. Descriptive statistics and
the shared noise gate live in analysis.py.

Run as a script to re-score an existing comparison.json with other parameters:

    python3 rules.py --rule paired_t --alpha 0.001 --min-improvement 5 --max-noise 20 path/to/comparison.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

STATUSES = ('inconclusive', 'regression_signal', 'improvement_signal', 'indeterminate', 'too_noisy')
DEFAULT_ALPHA = 0.001
DEFAULT_MIN_IMPROVEMENT = 5.0
DEFAULT_MAX_NOISE = 20.0


def noise_percent(values):
    """Coefficient of variation of one side's per-round times, in percent."""
    return 100 * statistics.stdev(values) / statistics.mean(values)


def cutoff(alpha, df):
    """The (1 - alpha) quantile of Student's t with df degrees of freedom: the one-sided cutoff."""
    from scipy import stats  # imported here so the measurement code does not need scipy

    return float(stats.t.isf(alpha, df))


def check_parameters(alpha, min_improvement, baseline, candidate, name):
    if not (math.isfinite(alpha) and 0 < alpha < 0.5):
        raise ValueError('alpha must be a one-sided false-positive rate strictly between 0 and 0.5')
    if not (math.isfinite(min_improvement) and min_improvement >= 0):
        raise ValueError('min_improvement must be finite and nonnegative')
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError(f'{name} needs at least two rounds with both sides')


def paired_t(baseline, candidate, alpha=DEFAULT_ALPHA, min_improvement=DEFAULT_MIN_IMPROVEMENT):
    """Paired t-test of the per-round differences (scipy ttest_rel) at one-sided level alpha.

    Rounds alternate both sides on one core, so the difference within a round cancels
    drift shared by both sides; the standard error is stdev(differences) / sqrt(rounds),
    with rounds - 1 degrees of freedom. The candidate is signalled faster (slower) when
    it beats (loses to) the baseline by more than min_improvement percent of the
    baseline mean, tested one-sided in each direction against the (1 - alpha) quantile
    of Student's t with rounds - 1 degrees of freedom, so the false-positive rate per
    direction is alpha whatever the number of rounds. The rule assumes the rounds are
    sufficiently independent and measured under comparable conditions; repetitions
    inside one process are correlated and must not be counted as rounds.

    Invalid parameters raise ValueError. Identical differences in every round (zero
    standard error) return 'indeterminate' rather than a signal; 'inconclusive' means
    insufficient evidence, not equal performance.
    """
    from scipy import stats  # imported here so the measurement code does not need scipy

    check_parameters(alpha, min_improvement, baseline, candidate, 'paired_t')
    rounds = len(baseline)
    required_ns = min_improvement / 100 * statistics.mean(baseline)
    differences = [b - c for b, c in zip(baseline, candidate)]
    improvement = statistics.mean(differences)
    se_diff = statistics.stdev(differences) / math.sqrt(rounds)
    df = rounds - 1
    result = {'status': 'indeterminate', 'improvement_ns': improvement, 'se_diff': se_diff, 'df': df,
              't_statistic': None, 'p_value': None, 'regression_t_statistic': None, 'regression_p_value': None,
              'alpha': alpha, 'cutoff_t': cutoff(alpha, df),
              'min_improvement_percent': min_improvement, 'min_improvement_ns': required_ns}
    if se_diff == 0:
        return result
    forward = stats.ttest_rel(baseline, [c + required_ns for c in candidate], alternative='greater')
    reverse = stats.ttest_rel(candidate, [b + required_ns for b in baseline], alternative='greater')
    t_forward, t_reverse = float(forward.statistic), float(reverse.statistic)
    if not all(math.isfinite(v) for v in (t_forward, t_reverse)):
        raise ValueError('nonfinite intermediate result')
    return result | {'status': decide(t_forward, t_reverse, result['cutoff_t']),
                     't_statistic': t_forward, 'p_value': float(forward.pvalue),
                     'regression_t_statistic': t_reverse, 'regression_p_value': float(reverse.pvalue)}


def decide(t_forward, t_reverse, cutoff_t):
    if t_forward > cutoff_t:
        return 'improvement_signal'
    if t_reverse > cutoff_t:
        return 'regression_signal'
    return 'inconclusive'


def welch_t(baseline, candidate, alpha=DEFAULT_ALPHA, min_improvement=DEFAULT_MIN_IMPROVEMENT):
    """Welch's unequal-variance t-test of the two sides at one-sided level alpha.

    The sides are treated as independent samples: the standard error is
    sqrt(var(baseline) / n + var(candidate) / n), with Welch-Satterthwaite degrees of
    freedom (about 2 * (rounds - 1) when both sides are equally noisy), so drift shared by
    the two launches of a round is not cancelled. The margin and the one-sided tests in
    each direction are the same as for paired_t; the cutoff is the (1 - alpha) quantile
    of Student's t with the Welch-Satterthwaite degrees of freedom.
    """
    from scipy import stats

    check_parameters(alpha, min_improvement, baseline, candidate, 'welch_t')
    rounds = len(baseline)
    required_ns = min_improvement / 100 * statistics.mean(baseline)
    improvement = statistics.mean(baseline) - statistics.mean(candidate)
    variances = [statistics.variance(side) / rounds for side in (baseline, candidate)]
    se_diff = math.sqrt(sum(variances))
    result = {'status': 'indeterminate', 'improvement_ns': improvement, 'se_diff': se_diff, 'df': None,
              't_statistic': None, 'p_value': None, 'regression_t_statistic': None, 'regression_p_value': None,
              'alpha': alpha, 'cutoff_t': None,
              'min_improvement_percent': min_improvement, 'min_improvement_ns': required_ns}
    if se_diff == 0:
        return result
    # Computed directly rather than with ttest_ind, which warns on nearly identical samples.
    df = sum(variances) ** 2 / sum(v ** 2 / (rounds - 1) for v in variances)
    t_forward, t_reverse = (improvement - required_ns) / se_diff, (-improvement - required_ns) / se_diff
    if not all(math.isfinite(v) for v in (t_forward, t_reverse, df)):
        raise ValueError('nonfinite intermediate result')
    cutoff_t = cutoff(alpha, df)
    return result | {'status': decide(t_forward, t_reverse, cutoff_t), 'df': df, 'cutoff_t': cutoff_t,
                     't_statistic': t_forward, 'p_value': float(stats.t.sf(t_forward, df)),
                     'regression_t_statistic': t_reverse, 'regression_p_value': float(stats.t.sf(t_reverse, df))}


RULES = {'paired_t': paired_t, 'welch_t': welch_t}


def main(argv=None):
    import analysis

    parser = argparse.ArgumentParser(description='Re-score an existing comparison.json with other rule parameters.')
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--rule', choices=sorted(RULES), default='paired_t')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='one-sided false-positive rate per direction; the t cutoff follows from the degrees of freedom')
    parser.add_argument('--min-improvement', type=float, default=DEFAULT_MIN_IMPROVEMENT,
                        help='required change as a percentage of the baseline mean before a signal')
    parser.add_argument('--max-noise', type=float, default=DEFAULT_MAX_NOISE,
                        help='a side whose per-round cv exceeds this percentage makes the case too_noisy')
    parser.add_argument('--list', action='store_true', help='print every case that is not inconclusive')
    args = parser.parse_args(argv)
    manifest = json.loads(args.comparison.read_text())
    counts, listed = Counter(), []
    for case in manifest['cases']:
        rounds = case.get('rounds', [])
        if case['analysis']['status'] in ('added', 'removed', 'incomplete') or not rounds:
            counts[case['analysis']['status']] += 1
            continue
        verdict = analysis.analyze(rounds, len(rounds), args.alpha, args.min_improvement, args.max_noise,
                                   args.rule)
        counts[verdict['status']] += 1
        if verdict['status'] != 'inconclusive':
            listed.append((case['suite'], case['name'], verdict))
    print(f'{args.rule} alpha={100 * args.alpha:g}% min_improvement={args.min_improvement:g}% '
          f'max_noise={args.max_noise:g}%: ' + ', '.join(f'{n} {status}' for status, n in sorted(counts.items())))
    if args.list:
        for suite, name, verdict in listed:
            t_statistic = verdict['verdict'].get('t_statistic')
            detail = f't={t_statistic:+.2f}' if t_statistic is not None else ''
            noise = ' '.join(f'{side} cv {value:.1f}%' for side, value in verdict['noise_percent'].items())
            print(f'  {suite}: {name}: {verdict["status"]} change {verdict["median_change_percent"]:+.2f}% {detail} ({noise})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
