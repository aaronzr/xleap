"""Rolling percentile-band rendering.

Compute a trailing-window percentile envelope for archive data (median, mid,
lower, upper) and render it as a filled band. The compute and render steps
are decoupled so a heavy full-resolution series is computed once and drawn
many times at varying pixel budgets.
"""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from sparklines_v2.plot.series import first_tick_of_day_with_date


def plot_percentile_band(
    data,
    window_size=10,
    *,
    ax=None,
    lower_q=0.90,
    center_q=0.95,
    upper_q=None,
    median_q=0.50,
    break_gap_s=None,
    value_scale=1.0,
    line_color="red",
    median_color="black",
    band_color="0.25",
    band_alpha=0.18,
    line_label=None,
    median_label=None,
    band_label=None,
    y_label=None,
    x_label="Time",
    show_legend=True,
    render_max_points=None,
    precomputed_band=None,
):
    """Plot trailing upper-tail percentile bands for archive data."""
    if precomputed_band is None:
        band = compute_percentile_band_series(
            data,
            window_size=window_size,
            lower_q=lower_q,
            center_q=center_q,
            upper_q=upper_q,
            median_q=median_q,
            break_gap_s=break_gap_s,
            value_scale=value_scale,
        )
    else:
        band = precomputed_band
    rendered = render_percentile_band_series(
        band,
        render_max_points=render_max_points,
    )

    if ax is None:
        _, ax = plt.subplots()

    if band_label is None:
        if upper_q is None:
            band_label = f"{int(round(lower_q * 100))}th percentile to max"
        else:
            band_label = (
                f"{int(round(lower_q * 100))}th to "
                f"{int(round(upper_q * 100))}th percentile"
            )
    if line_label is None:
        line_label = f"{int(round(center_q * 100))}th percentile"
    if median_label is None:
        median_label = f"{int(round(median_q * 100))}th percentile"

    ax.fill_between(
        rendered["time"],
        rendered["lower"],
        rendered["upper"],
        color=band_color,
        alpha=band_alpha,
        label=band_label,
    )
    ax.plot(rendered["time"], rendered["center"], color=line_color, linewidth=1.5, label=line_label)
    ax.plot(
        rendered["time"],
        rendered["median"],
        color=median_color,
        linewidth=1.5,
        label=median_label,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(first_tick_of_day_with_date(ax)))
    ax.figure.autofmt_xdate()
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(loc="best")

    return ax, rendered


def compute_percentile_band_series(
    data,
    window_size=10,
    *,
    lower_q=0.90,
    center_q=0.95,
    upper_q=None,
    median_q=0.50,
    break_gap_s=None,
    value_scale=1.0,
):
    """Compute full-resolution rolling percentile series once."""
    if not isinstance(data, dict):
        raise ValueError(
            "data must be an archive-style dict with 'values' and 'secondsPastEpoch'"
        )
    if "values" not in data or "secondsPastEpoch" not in data:
        raise ValueError("data must contain 'values' and 'secondsPastEpoch'")
    if not np.isfinite(window_size) or window_size <= 0:
        raise ValueError("window_size must be a positive finite number of seconds")
    if not (0.0 <= lower_q <= center_q <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= lower_q <= center_q <= 1")
    if upper_q is not None and not (center_q <= upper_q <= 1.0):
        raise ValueError("upper_q must be None or satisfy center_q <= upper_q <= 1")
    if not (0.0 <= median_q <= 1.0):
        raise ValueError("median_q must satisfy 0 <= median_q <= 1")

    values = np.asarray(data["values"], dtype=float).ravel()
    sec = np.asarray(data["secondsPastEpoch"], dtype=float).ravel()
    nsec = np.asarray(data.get("nanoseconds", data.get("nanosecond", 0)), dtype=float)
    if nsec.ndim == 0:
        nsec = np.full(sec.shape, float(nsec))
    else:
        nsec = nsec.ravel()
        if nsec.shape != sec.shape:
            raise ValueError(
                "'nanoseconds' must be scalar or match 'secondsPastEpoch' length"
            )
    if values.shape != sec.shape:
        raise ValueError(
            f"time/value length mismatch: len(t)={len(sec)} len(y)={len(values)}"
        )
    if values.size == 0:
        raise ValueError("data is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")

    t_ns = np.rint(sec * 1e9 + nsec).astype(np.int64)
    order = np.argsort(t_ns, kind="mergesort")
    t_ns = t_ns[order]
    values = values[order]

    local_times = [dt.datetime.fromtimestamp(ts_ns / 1e9) for ts_ns in t_ns]
    series = pd.Series(values, index=pd.DatetimeIndex(local_times))
    rolling = series.rolling(pd.Timedelta(seconds=float(window_size)), min_periods=1)
    q_low = rolling.quantile(lower_q).to_numpy() * float(value_scale)
    q_mid = rolling.quantile(center_q).to_numpy() * float(value_scale)
    if upper_q is None:
        q_high = rolling.max().to_numpy() * float(value_scale)
    else:
        q_high = rolling.quantile(upper_q).to_numpy() * float(value_scale)
    q_median = rolling.quantile(median_q).to_numpy() * float(value_scale)
    x = np.asarray(series.index.to_pydatetime(), dtype=object)
    source_points = len(x)

    sample_interval_s = None
    if len(t_ns) > 1:
        dt_s = np.diff(t_ns).astype(float) * 1e-9
        dt_s = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
        if dt_s.size:
            sample_interval_s = float(np.median(dt_s))

    if break_gap_s is None:
        gap_s = (
            None
            if sample_interval_s is None
            else max(float(window_size), 3.0 * sample_interval_s)
        )
    else:
        gap_s = float(break_gap_s)
    if gap_s is not None and (not np.isfinite(gap_s) or gap_s <= 0):
        raise ValueError("break_gap_s must be a positive finite number of seconds")

    segment_ids = np.zeros(len(t_ns), dtype=int)
    if gap_s is not None and len(t_ns) > 1:
        source_gaps = np.diff(t_ns).astype(float) * 1e-9 > gap_s
        if source_gaps.size:
            segment_ids[1:] = np.cumsum(source_gaps, dtype=int)

    return {
        "time": x,
        "time_ns": t_ns,
        "lower": q_low,
        "median": q_median,
        "center": q_mid,
        "upper": q_high,
        "segment_ids": segment_ids,
        "source_points": source_points,
    }


def render_percentile_band_series(band, *, render_max_points=None):
    """Render a precomputed percentile band, optionally thinning in time."""
    if render_max_points is not None:
        if int(render_max_points) < 1:
            raise ValueError("render_max_points must be None or >= 1")
        render_max_points = int(render_max_points)

    x = np.asarray(band["time"], dtype=object)
    t_ns = np.asarray(band["time_ns"], dtype=np.int64)
    q_low = np.asarray(band["lower"], dtype=float)
    q_median = np.asarray(band["median"], dtype=float)
    q_mid = np.asarray(band["center"], dtype=float)
    q_high = np.asarray(band["upper"], dtype=float)
    segment_ids = np.asarray(band["segment_ids"], dtype=int)
    source_points = int(band["source_points"])

    if render_max_points is not None and len(x) > render_max_points:
        bin_edges = np.linspace(
            float(t_ns[0]),
            float(t_ns[-1]),
            num=render_max_points + 1,
        )
        sample_indices = []
        for idx in range(render_max_points):
            left_idx = int(np.searchsorted(t_ns, bin_edges[idx], side="left"))
            if idx == render_max_points - 1:
                right_idx = len(t_ns)
            else:
                right_idx = int(np.searchsorted(t_ns, bin_edges[idx + 1], side="left"))
            if left_idx < right_idx:
                sample_indices.append(left_idx)
        sample_indices = np.asarray(sample_indices, dtype=int)
    else:
        sample_indices = np.arange(len(x), dtype=int)

    x = x[sample_indices].tolist()
    q_low = q_low[sample_indices]
    q_median = q_median[sample_indices]
    q_mid = q_mid[sample_indices]
    q_high = q_high[sample_indices]
    rendered_points = len(sample_indices)
    sampled_segments = segment_ids[sample_indices]

    if len(x) > 1:
        x_gap = [x[0]]
        low_gap = [q_low[0]]
        mid_gap = [q_mid[0]]
        high_gap = [q_high[0]]
        median_gap = [q_median[0]]
        for i in range(1, len(x)):
            if sampled_segments[i] != sampled_segments[i - 1]:
                x_gap.append(x[i])
                low_gap.append(np.nan)
                mid_gap.append(np.nan)
                high_gap.append(np.nan)
                median_gap.append(np.nan)
            x_gap.append(x[i])
            low_gap.append(q_low[i])
            mid_gap.append(q_mid[i])
            high_gap.append(q_high[i])
            median_gap.append(q_median[i])
        x = x_gap
        q_low = np.asarray(low_gap, dtype=float)
        q_mid = np.asarray(mid_gap, dtype=float)
        q_high = np.asarray(high_gap, dtype=float)
        q_median = np.asarray(median_gap, dtype=float)

    return {
        "time": np.asarray(x, dtype=object),
        "lower": q_low,
        "median": q_median,
        "center": q_mid,
        "upper": q_high,
        "source_points": source_points,
        "rendered_points": rendered_points,
    }
