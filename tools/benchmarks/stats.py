"""The statistics that turn a case's per-round CPU times into a speedup and a verdict."""

import math
import statistics

import pandas as pd
from scipy import stats

ALPHA = 0.001           # one-sided false-positive rate of each direction of the paired t-test
MIN_IMPROVEMENT = 5.0   # percent of the baseline mean a change must exceed, in either direction, to be signalled
MAX_NOISE = 20.0        # percent: a side whose per-round coefficient of variation is larger is too noisy for a verdict
OUTLIER_SIGMA = 10.0    # a round where either side is this many robust sigmas from its median is dropped
MAD_TO_SIGMA = 1.4826   # 1 / Phi^-1(0.75): scales the median absolute deviation to the standard deviation of normal data


def noise_percent(times):
    """Coefficient of variation in percent."""
    return 100 * statistics.stdev(times) / statistics.mean(times)


def outlier_rounds(times):
    """Indices where either timing column is OUTLIER_SIGMA robust sigmas from its median.

    A column whose MAD is zero flags nothing. If more than half the rounds would go, none are dropped.
    """
    deviation = (times - times.median()).abs()
    sigma = MAD_TO_SIGMA * deviation.median()
    outside_band = deviation >= OUTLIER_SIGMA * sigma
    has_spread = sigma > 0  # A zero MAD must not flag every value in the column.
    outliers = (outside_band & has_spread).any(axis='columns')
    dropped = times.index[outliers].tolist()
    if len(dropped) > len(times) / 2:
        return []
    return dropped


def paired_t(baseline, candidate):
    """Paired t-test of the per-round differences baseline - candidate, one-sided in each direction: (status, t).

    Both sides of a round run back to back on one core, so the difference within a round cancels drift that both
    share. The candidate is an 'improvement' ('regression') when it is faster (slower) by more than MIN_IMPROVEMENT
    percent of the baseline mean, at a false-positive rate of ALPHA per direction. 'unchanged' means the evidence is
    insufficient, not that the two are equal. t is the statistic of the better-supported direction.
    """
    differences = [b - c for b, c in zip(baseline, candidate)]  # > 0: the candidate is faster
    rounds = len(differences)
    margin = MIN_IMPROVEMENT / 100 * statistics.mean(baseline)
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(rounds)
    t_faster = (mean - margin) / se
    t_slower = (-mean - margin) / se
    cutoff = stats.t.isf(ALPHA, rounds - 1)  # the (1 - ALPHA) quantile of Student's t with rounds - 1 degrees of freedom
    status = 'improvement' if t_faster > cutoff else 'regression' if t_slower > cutoff else 'unchanged'
    return status, max(t_faster, t_slower)


def analyze(baseline, candidate):
    """Speedup and verdict from the per-round CPU times of the two sides (entry i of both lists is round i).

    Outlier rounds are left out and listed in 'dropped_rounds'. A side whose per-round coefficient of variation exceeds
    MAX_NOISE makes the case 'too_noisy' without a test. speedup = median over rounds of baseline time / candidate
    time (> 1: the candidate is faster); speedup_log_se is the standard error of the per-round log speedup.
    """
    # One row per paired round; dropping a row removes both measurements.
    times = pd.DataFrame({'baseline': baseline, 'candidate': candidate})
    dropped = outlier_rounds(times)
    times = times.drop(index=dropped)
    ratios = times['baseline'] / times['candidate']
    noise = times.apply(noise_percent).max()
    if noise > MAX_NOISE:
        status, t = 'too_noisy', None
    else:
        status, t = paired_t(times['baseline'], times['candidate'])
    return {'status': status, 't': t, 'speedup': float(ratios.median()),
            'speedup_log_se': float(ratios.map(math.log).sem()), 'dropped_rounds': dropped}
