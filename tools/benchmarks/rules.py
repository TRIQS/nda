"""Decision rules that turn per-round timings into a status.

A rule takes two equally long lists of CPU times in nanoseconds, one per side,
where entry i of both lists comes from round i. It returns a dict with at least
'status' from STATUSES; the evidence goes next to it. Descriptive statistics and
the shared noise gate live in analysis.py.

Run as a script to re-score an existing comparison.json with other parameters:

    python3 rules.py --rule paired_t --threshold 3 --min-improvement 5 --max-noise 20 path/to/comparison.json
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
DEFAULT_THRESHOLD = 3.0
DEFAULT_MIN_IMPROVEMENT = 5.0
DEFAULT_MAX_NOISE = 20.0


def noise_percent(values):
    """Coefficient of variation of one side's per-round times, in percent."""
    return 100 * statistics.stdev(values) / statistics.mean(values)


def paired_t(baseline, candidate, threshold=DEFAULT_THRESHOLD, min_improvement=DEFAULT_MIN_IMPROVEMENT):
    """Fixed cutoff on the paired t statistic of the per-round differences (scipy ttest_rel).

    Rounds alternate both sides on one core, so the difference within a round cancels
    drift shared by both sides; the standard error is stdev(differences) / sqrt(rounds),
    with rounds - 1 degrees of freedom. The candidate is signalled faster (slower) when
    it beats (loses to) the baseline by more than min_improvement percent of the
    baseline mean plus threshold standard errors, tested one-sided in each direction.

    threshold is a fixed cutoff, not a sample-size-adjusted critical value: at six
    rounds t > 3 corresponds to about 1.5% one-sided under the null, not 0.135%; the
    reported one-sided p-values use the exact degrees of freedom. The rule assumes the
    rounds are sufficiently independent and measured under comparable conditions;
    repetitions inside one process are correlated and must not be counted as rounds.

    Invalid parameters raise ValueError. Identical differences in every round (zero
    standard error) return 'indeterminate' rather than a signal; 'inconclusive' means
    insufficient evidence, not equal performance.
    """
    from scipy import stats  # imported here so the measurement code does not need scipy

    if not (math.isfinite(threshold) and threshold > 0):
        raise ValueError('threshold must be finite and positive')
    if not (math.isfinite(min_improvement) and min_improvement >= 0):
        raise ValueError('min_improvement must be finite and nonnegative')
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError('paired_t needs at least two rounds with both sides')
    rounds = len(baseline)
    required_ns = min_improvement / 100 * statistics.mean(baseline)
    differences = [b - c for b, c in zip(baseline, candidate)]
    improvement = statistics.mean(differences)
    se_diff = statistics.stdev(differences) / math.sqrt(rounds)
    result = {'status': 'indeterminate', 'improvement_ns': improvement, 'se_diff': se_diff, 'df': rounds - 1,
              't_statistic': None, 'p_value': None, 'regression_t_statistic': None, 'regression_p_value': None,
              'threshold_t': threshold, 'min_improvement_percent': min_improvement, 'min_improvement_ns': required_ns}
    if se_diff == 0:
        return result
    forward = stats.ttest_rel(baseline, [c + required_ns for c in candidate], alternative='greater')
    reverse = stats.ttest_rel(candidate, [b + required_ns for b in baseline], alternative='greater')
    t_forward, t_reverse = float(forward.statistic), float(reverse.statistic)
    if not all(math.isfinite(v) for v in (t_forward, t_reverse)):
        raise ValueError('nonfinite intermediate result')
    if t_forward > threshold:
        status = 'improvement_signal'
    elif t_reverse > threshold:
        status = 'regression_signal'
    else:
        status = 'inconclusive'
    return result | {'status': status, 't_statistic': t_forward, 'p_value': float(forward.pvalue),
                     'regression_t_statistic': t_reverse, 'regression_p_value': float(reverse.pvalue)}


def welch_t(baseline, candidate, threshold=DEFAULT_THRESHOLD, min_improvement=DEFAULT_MIN_IMPROVEMENT):
    """Fixed cutoff on Welch's unequal-variance t statistic of the two sides (scipy ttest_ind).

    The sides are treated as independent samples: the standard error is
    sqrt(var(baseline) / n + var(candidate) / n), with Welch-Satterthwaite degrees of
    freedom (about 2 * (rounds - 1) when both sides are equally noisy), so drift shared by
    the two launches of a round is not cancelled. The margin and the one-sided tests in
    each direction are the same as for paired_t, and threshold is again a fixed cutoff:
    with more degrees of freedom t > 3 is a stricter level than it is for paired_t.
    """
    from scipy import stats

    if not (math.isfinite(threshold) and threshold > 0):
        raise ValueError('threshold must be finite and positive')
    if not (math.isfinite(min_improvement) and min_improvement >= 0):
        raise ValueError('min_improvement must be finite and nonnegative')
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError('welch_t needs at least two rounds with both sides')
    rounds = len(baseline)
    required_ns = min_improvement / 100 * statistics.mean(baseline)
    improvement = statistics.mean(baseline) - statistics.mean(candidate)
    variances = [statistics.variance(side) / rounds for side in (baseline, candidate)]
    se_diff = math.sqrt(sum(variances))
    result = {'status': 'indeterminate', 'improvement_ns': improvement, 'se_diff': se_diff, 'df': None,
              't_statistic': None, 'p_value': None, 'regression_t_statistic': None, 'regression_p_value': None,
              'threshold_t': threshold, 'min_improvement_percent': min_improvement, 'min_improvement_ns': required_ns}
    if se_diff == 0:
        return result
    # Computed directly rather than with ttest_ind, which warns on nearly identical samples.
    df = sum(variances) ** 2 / sum(v ** 2 / (rounds - 1) for v in variances)
    t_forward, t_reverse = (improvement - required_ns) / se_diff, (-improvement - required_ns) / se_diff
    if not all(math.isfinite(v) for v in (t_forward, t_reverse, df)):
        raise ValueError('nonfinite intermediate result')
    if t_forward > threshold:
        status = 'improvement_signal'
    elif t_reverse > threshold:
        status = 'regression_signal'
    else:
        status = 'inconclusive'
    return result | {'status': status, 'df': df,
                     't_statistic': t_forward, 'p_value': float(stats.t.sf(t_forward, df)),
                     'regression_t_statistic': t_reverse, 'regression_p_value': float(stats.t.sf(t_reverse, df))}


RULES = {'paired_t': paired_t, 'welch_t': welch_t}


def main(argv=None):
    import analysis

    parser = argparse.ArgumentParser(description='Re-score an existing comparison.json with other rule parameters.')
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--rule', choices=sorted(RULES), default='paired_t')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='cutoff on the t statistic')
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
        verdict = analysis.analyze(rounds, len(rounds), args.threshold, args.min_improvement, args.max_noise,
                                   args.rule)
        counts[verdict['status']] += 1
        if verdict['status'] != 'inconclusive':
            listed.append((case['suite'], case['name'], verdict))
    print(f'{args.rule} threshold={args.threshold:g} min_improvement={args.min_improvement:g}% '
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
