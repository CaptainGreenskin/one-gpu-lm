"""Describe sampled board power without bridging outages or controller attempts."""
from datetime import datetime
import math
import statistics


def summarize(samples, max_interval_seconds=15):
    times = [datetime.fromisoformat(r['utc']).timestamp() for r in samples]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Hardware timestamps must be strictly increasing')
    for r in samples:
        if any(not math.isfinite(float(r[k])) for k in ['temperature_c', 'memory_used_mib', 'power_w', 'utilization_percent']):
            raise ValueError('Nonfinite hardware sample')
    intervals = [(b-a, x, y) for a, b, x, y in zip(times, times[1:], samples, samples[1:])]
    measured = [(dt, x, y) for dt, x, y in intervals
                if dt <= max_interval_seconds and x['attempt'] == y['attempt']]
    gaps = [{'from_utc': x['utc'], 'to_utc': y['utc'], 'seconds': dt,
             'reason': 'attempt_boundary' if x['attempt'] != y['attempt'] else 'sample_interval_exceeded'}
            for dt, x, y in intervals if dt > max_interval_seconds or x['attempt'] != y['attempt']]
    span = times[-1] - times[0]
    covered = sum(dt for dt, _, _ in measured)
    value = {'samples': len(samples), 'max_gap_seconds': max(dt for dt, _, _ in intervals),
             'sampled_span_seconds': span, 'covered_interval_seconds': covered,
             'unobserved_interval_seconds': span-covered, 'interval_coverage_fraction': covered/span,
             'max_energy_interval_seconds': max_interval_seconds, 'unobserved_intervals': gaps,
             'sampled_gpu_energy_kwh': sum(dt*(float(x['power_w'])+float(y['power_w']))/2
                                           for dt, x, y in measured)/3600000,
             'energy_scope': 'Trapezoidal board-power estimate inside sampled intervals only; no outage interpolation or extrapolation.'}
    for name in ['temperature_c', 'memory_used_mib', 'power_w', 'utilization_percent']:
        numbers = [float(r[name]) for r in samples]
        value[name] = {'min': min(numbers), 'median': statistics.median(numbers), 'max': max(numbers)}
    return value


def plot_series(samples, field, max_interval_seconds=15):
    """NaNs break the line at every unobserved interval."""
    x, y = [], []
    previous = None
    start = datetime.fromisoformat(samples[0]['utc']).timestamp()
    for row in samples:
        now = datetime.fromisoformat(row['utc']).timestamp()
        if previous is not None:
            before = datetime.fromisoformat(previous['utc']).timestamp()
            if now-before > max_interval_seconds or row['attempt'] != previous['attempt']:
                x.append(float('nan')); y.append(float('nan'))
        x.append((now-start)/60); y.append(float(row[field])); previous = row
    return x, y
