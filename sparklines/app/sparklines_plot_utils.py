"""Common plotting and tuning utilities."""

import datetime as dt
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import requests
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Polygon

try:
    from .sparklines_hierarchy import (
        ARCHIVER_URL,
        ARCHIVE_TIMEOUT_SECONDS,
        LOCAL_TIMEZONE,
        _format_archive_time,
    )
except ImportError:  # pragma: no cover - notebook/script fallback
    from sparklines_hierarchy import (
        ARCHIVER_URL,
        ARCHIVE_TIMEOUT_SECONDS,
        LOCAL_TIMEZONE,
        _format_archive_time,
    )


def get_archive_data(
    pv,
    *,
    from_time=None,
    to_time=None,
    archiver_url=ARCHIVER_URL,
    timeout=ARCHIVE_TIMEOUT_SECONDS,
    local_timezone=LOCAL_TIMEZONE,
):
    """Fetch a single archive PV using the local app archiver client."""
    response = requests.get(
        archiver_url,
        params={
            'pv': pv,
            'from': _format_archive_time(from_time, local_timezone=local_timezone),
            'to': _format_archive_time(to_time, local_timezone=local_timezone),
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f'archive returned empty payload for PV {pv}')

    first = payload[0]
    data = first.get('data')
    if not isinstance(data, list):
        raise RuntimeError(f'archive payload missing data array for PV {pv}')

    return {
        'name': (first.get('meta') or {}).get('name', pv),
        'secondsPastEpoch': np.array([datum['secs'] for datum in data]),
        'values': np.array([datum['val'] for datum in data]),
        'nanoseconds': np.array([datum['nanos'] for datum in data]),
        'severity': np.array([datum['severity'] for datum in data]),
        'status': np.array([datum['status'] for datum in data]),
    }


def first_tick_of_day_with_date(ax):
    """Return a formatter that prefixes the first tick of each day with the date."""
    last_day = [None]

    def _fmt(x, pos=None):
        dt_obj = mdates.num2date(x)
        day = dt_obj.date()
        tick_locs = ax.xaxis.get_majorticklocs()
        show_seconds = False
        if len(tick_locs) > 1:
            min_step_days = min(b - a for a, b in zip(tick_locs, tick_locs[1:]))
            step_seconds = min_step_days * 86400
            show_seconds = round(step_seconds) < 60
        if pos == 0 or last_day[0] != day:
            last_day[0] = day
            return dt_obj.strftime('%Y-%m-%d %H:%M:%S' if show_seconds else '%Y-%m-%d %H:%M')
        return dt_obj.strftime('%H:%M:%S' if show_seconds else '%H:%M')

    return _fmt


def add_pv_series(ax, data, plot_fn='scatter', **kwargs):
    """Plot a PV series on the given axes using the selected plot function."""
    t = data['secondsPastEpoch']
    y = data['values']
    t_dt = [dt.datetime.fromtimestamp(x) for x in t]

    if isinstance(plot_fn, str):
        plot = getattr(ax, plot_fn)
        plot_name = plot_fn
    else:
        plot = plot_fn
        plot_name = None

    if plot_name in {'plot', 'step'}:
        kwargs.pop('linewidths', None)

    plot(t_dt, y, **kwargs)


def plot_pvs(pvs, start, end, colors=None, ax=None, **kwargs):
    """
    Fetch and plot multiple PVs on a shared axis with a color cycle.

    Parameters
    ----------
    pvs : sequence of str
        PV names to fetch and plot.
    start, end : datetime-like or float
        Time bounds passed to `get_archive_data(...)`.
    colors : sequence or str, optional
        Color cycle for each PV. If None, use Tableau colors. If a single
        color string is provided, all series use that color.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure/axis is created.
    **kwargs : dict
        Forwarded to `add_pv_series`/matplotlib (e.g., marker, linewidths,
        plot_fn).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plotted series.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        color_cycle = ['k'] if len(pvs) == 1 else mcolors.TABLEAU_COLORS.keys()
    elif isinstance(colors, (str, bytes)):
        color_cycle = [colors]
    else:
        color_cycle = colors
    colors = itertools.cycle(color_cycle)

    kwargs.setdefault('marker', 'x')
    kwargs.setdefault('linewidths', 0.6)

    for pv in pvs:
        data = get_archive_data(pv, from_time=start, to_time=end)
        add_pv_series(ax, data, color=next(colors), label=pv, **kwargs)

    ax.xaxis.set_major_formatter(FuncFormatter(first_tick_of_day_with_date(ax)))
    ax.legend()
    if fig is not None:
        fig.autofmt_xdate()
        plt.show()
    return ax


def _to_datetime_list(t):
    """Normalize time values to a list of Python datetimes."""
    t_arr = np.asarray(t)
    if np.issubdtype(t_arr.dtype, np.datetime64):
        return [x.astype('datetime64[us]').item() for x in t_arr]
    if t_arr.ndim == 0:
        t_arr = np.asarray([t_arr.item()])
    if len(t_arr) > 0 and isinstance(t_arr[0], dt.datetime):
        return list(t_arr)
    return [dt.datetime.fromtimestamp(float(x)) for x in t_arr]


def _extract_series_time_and_values(series, t):
    """
    Return (t_dt, values) for either archive-style dict input or raw arrays.
    """
    if isinstance(series, dict):
        if 'values' not in series or 'secondsPastEpoch' not in series:
            raise ValueError(
                "dict series must contain 'values' and 'secondsPastEpoch'"
            )
        values = np.asarray(series['values'])
        sec = np.asarray(series['secondsPastEpoch'], dtype=float)
        nsec = np.asarray(series.get('nanoseconds', series.get('nanosecond', 0)),
                         dtype=float)
        if nsec.ndim == 0:
            nsec = np.full(sec.shape, float(nsec))
        elif nsec.shape != sec.shape:
            raise ValueError(
                "'nanoseconds' must be scalar or match 'secondsPastEpoch' length"
            )
        t_dt = _to_datetime_list(sec + nsec * 1e-9)
    else:
        values = np.asarray(series)
        if t is None:
            raise ValueError("time values `t` are required for non-dict `y` inputs")
        t_dt = _to_datetime_list(t)
    if len(t_dt) != len(values):
        raise ValueError(
            f"time/value length mismatch: len(t)={len(t_dt)} len(y)={len(values)}"
        )
    return t_dt, values


def rolling_avg(data, window_size):
    """
    Compute rolling mean/std over a trailing time window for archive data.

    Parameters
    ----------
    data : dict
        Archive-style dict containing `values`, `secondsPastEpoch`, and
        optional `nanoseconds`/`nanosecond`.
    window_size : float
        Window duration in seconds.

    Returns
    -------
    mean : numpy.ndarray
        Rolling mean at each input sample time.
    std : numpy.ndarray
        Rolling standard deviation at each input sample time.

    Notes
    -----
    For times near the left edge where the trailing window extends before the
    first sample, the first value is broadcast backward in time so each sample
    is evaluated on a full `window_size` interval.
    """
    if not isinstance(data, dict):
        raise ValueError(
            "data must be an archive-style dict with 'values' and "
            "'secondsPastEpoch'"
        )
    if not np.isfinite(window_size) or window_size <= 0:
        raise ValueError("window_size must be a positive finite number of seconds")
    if 'values' not in data or 'secondsPastEpoch' not in data:
        raise ValueError("data must contain 'values' and 'secondsPastEpoch'")

    values = np.asarray(data['values'], dtype=float).ravel()
    sec = np.asarray(data['secondsPastEpoch'], dtype=float).ravel()
    nsec = np.asarray(data.get('nanoseconds', data.get('nanosecond', 0)),
                     dtype=float)
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
        return np.array([], dtype=float), np.array([], dtype=float)

    t = sec + nsec * 1e-9
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")

    order = np.argsort(t, kind='mergesort')
    t_sorted = t[order]
    y_sorted = values[order]
    n = len(t_sorted)

    dt_seg = np.diff(t_sorted)
    cum_1 = np.zeros(n, dtype=float)
    cum_2 = np.zeros(n, dtype=float)
    if n > 1:
        cum_1[1:] = np.cumsum(y_sorted[:-1] * dt_seg)
        cum_2[1:] = np.cumsum((y_sorted[:-1] ** 2) * dt_seg)

    t0 = t_sorted[0]
    y0 = y_sorted[0]

    def _integral_at(x, cumulative, y_piecewise):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = x > t0
        if not np.any(mask):
            return out
        idx = np.searchsorted(t_sorted, x[mask], side='right') - 1
        idx = np.clip(idx, 0, n - 1)
        out[mask] = (
            cumulative[idx] +
            y_piecewise[idx] * (x[mask] - t_sorted[idx])
        )
        return out

    end = t_sorted
    start = end - float(window_size)
    start_clipped = np.maximum(start, t0)

    i1 = cum_1 - _integral_at(start_clipped, cum_1, y_sorted)
    i2 = cum_2 - _integral_at(start_clipped, cum_2, y_sorted)

    left_mask = start < t0
    if np.any(left_mask):
        left_duration = t0 - start[left_mask]
        i1[left_mask] += left_duration * y0
        i2[left_mask] += left_duration * (y0 ** 2)

    mean_sorted = i1 / float(window_size)
    var_sorted = i2 / float(window_size) - mean_sorted ** 2
    var_sorted[var_sorted < 0] = 0.0
    std_sorted = np.sqrt(var_sorted)

    mean = np.empty_like(mean_sorted)
    std = np.empty_like(std_sorted)
    mean[order] = mean_sorted
    std[order] = std_sorted
    return mean, std


def plot_on_axis(y, ax, t=None, lock_y=True, label=None, y_ticks=None,
                 pad_frac=0.1, colors=None):
    """
    Plot one or more step series on a twin y-axis sharing the same x-axis.

    Supported inputs:
    - `y` array-like + explicit `t`.
    - `y` archive dict containing `values`, `secondsPastEpoch`, and optional
      `nanoseconds`/`nanosecond` (time inferred from dict).
    - `y` list of arrays/dicts, with per-series time from dict or from `t`.
      For list inputs, `t` may be one shared time array or a list of per-series
      time arrays.
    """
    ax2 = ax.twinx()

    is_scalar_list = isinstance(y, (list, tuple)) and all(np.isscalar(v) for v in y)
    if isinstance(y, (list, tuple)) and not is_scalar_list:
        ys = list(y)
    else:
        ys = [y]

    if label is None:
        labels = [None] * len(ys)
    elif isinstance(label, (list, tuple)):
        labels = list(label)
        if len(labels) == 1 and len(ys) > 1:
            labels = labels * len(ys)
        if len(labels) != len(ys):
            raise ValueError("labels length must match number of series")
    else:
        labels = [label] * len(ys)

    if len(ys) == 1:
        t_items = [t]
    elif t is None:
        t_items = [None] * len(ys)
    elif (isinstance(t, (list, tuple)) and len(t) == len(ys) and
          any(isinstance(item, (list, tuple, np.ndarray)) for item in t)):
        t_items = list(t)
    else:
        t_items = [t] * len(ys)

    if colors is None:
        color_cycle = mcolors.TABLEAU_COLORS.keys()
    elif isinstance(colors, (str, bytes)):
        color_cycle = [colors]
    else:
        color_cycle = colors
    color_iter = itertools.cycle(color_cycle)

    y_mins = []
    y_maxs = []
    for idx, (series, label_i, t_i) in enumerate(zip(ys, labels, t_items)):
        t_dt, values = _extract_series_time_and_values(series, t_i)
        if isinstance(series, dict) and label_i is None:
            label_i = series.get('label')
        y_plot = values.astype(int) if values.dtype == bool else values
        ax2.step(
            t_dt,
            y_plot,
            where='post',
            color=next(color_iter),
            linewidth=1,
            alpha=0.3,
            label=label_i if label_i is not None else f'y{idx + 1}',
        )
        y_mins.append(float(np.min(values)))
        y_maxs.append(float(np.max(values)))

    if lock_y:
        ax2.set_autoscale_on(False)
        ax2.set_navigate(False)
        y_min = min(y_mins)
        y_max = max(y_maxs)
        span = y_max - y_min
        pad = pad_frac * span if span != 0 else pad_frac
        fixed_ylim = (y_min - pad, y_max + pad)
        ax2.set_ylim(*fixed_ylim)
        if not hasattr(ax2, '_locking'):
            ax2._locking = False

        def _lock_power_axis(_):
            if ax2._locking:
                return
            ax2._locking = True
            ax2.set_ylim(*fixed_ylim)
            ax2._locking = False

        ax2.callbacks.connect('ylim_changed', _lock_power_axis)

    if y_ticks is not None:
        ax2.set_yticks(y_ticks)
    if isinstance(label, str):
        ax2.set_ylabel(label)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left')

    fig = ax.figure
    fig.tight_layout()
    fig.autofmt_xdate()
    return ax2


def detect_tuning(t, timeout=300):
    """
    Return a list of paired times (t_on, t_off) representing periods when tuning
    (frequent parameter changes) was occuring.

    Rules:
    - off -> on: whenever a tuning event is detected. t_on is set to time of observed event.
    - on -> on: as long as tuning events continue to be observed and each one occurs
      before `timeout` elapses since the last event. Do not change t_on.
    - on -> off: when `timeout` elapses since the last tuning event before another is
      observed. t_off is set to time of last observed event. Add tuple (t_on, t_off)
      to the list.

    Params:
      t: a list of Unix times at which parameter changes occur.
      timeout: how long to wait after the last-observed time before declaring the tuning
        period to have ended.

    Returns:
      A list of pairs of times [(t1_on, t1_off), (t2_on, t2_off), ...] representing the
      start and end of each tuning period.
    """
    t = np.asarray(t)
    if t.size == 0:
        return []

    t = np.sort(t)
    periods = []
    t_on = t[0]
    last_event = t[0]

    for ti in t[1:]:
        if ti - last_event <= timeout:
            last_event = ti
            continue
        periods.append((t_on, last_event))
        t_on = ti
        last_event = ti

    periods.append((t_on, last_event))
    return periods


def get_tuning_events(t, timeout=300, values=None):
    """
    Return tuning periods or tuning period summaries with value statistics.

    Parameters
    ----------
    t : sequence[float]
        Unix times at which parameter changes occur.
    timeout : float, optional
        Passed to `detect_tuning`.
    values : sequence, optional
        If provided, must match `t` shape. Returns per-period dict summaries
        with keys: t_init, t_final, v_max, v_min, v_final.

    Returns
    -------
    list
        If `values` is None, returns `detect_tuning(t, timeout=timeout)`.
        Otherwise returns a list of dicts.
    """
    if values is None:
        return detect_tuning(t, timeout=timeout)

    t_arr = np.asarray(t, dtype=float)
    v_arr = np.asarray(values)
    if t_arr.shape != v_arr.shape:
        raise ValueError(
            "time/value length mismatch: "
            f"len(t)={t_arr.size} len(values)={v_arr.size}"
        )
    if t_arr.size == 0:
        return []

    mask = np.isfinite(t_arr)
    if np.issubdtype(v_arr.dtype, np.number):
        mask &= np.isfinite(v_arr)
    t_arr = t_arr[mask]
    v_arr = v_arr[mask]
    if t_arr.size == 0:
        return []

    order = np.argsort(t_arr, kind='mergesort')
    t_sorted = t_arr[order]
    v_sorted = v_arr[order]
    periods = detect_tuning(t_sorted, timeout=timeout)

    def _scalar(x):
        return x.item() if isinstance(x, np.generic) else x

    out = []
    for t_on, t_off in periods:
        i0 = int(np.searchsorted(t_sorted, t_on, side='left'))
        i1 = int(np.searchsorted(t_sorted, t_off, side='right') - 1)
        if i1 < i0:
            continue
        v_period = v_sorted[i0:i1 + 1]
        out.append(
            {
                't_init': float(t_sorted[i0]),
                't_final': float(t_sorted[i1]),
                'v_max': _scalar(v_period.max()),
                'v_min': _scalar(v_period.min()),
                'v_final': _scalar(v_sorted[i1]),
            }
        )

    return out


def pv_tuning_json(pvs, start, end, timeout=300):
    """
    Return per-PV tuning endpoint/value pairs as a JSON-like dict.

    For each PV, tuning periods are detected with `detect_tuning(...)` from the
    PV's event times. Each period contributes two entries, the period start and
    end, preserving time order:

      {pv0: [(t_on0, v_on0), (t_off0, v_off0), ...], ...}

    Parameters
    ----------
    pvs : sequence[str] or str
        PV name(s) to fetch from the archiver.
    start, end : datetime-like or float
        Time bounds passed to `get_archive_data(...)`.
    timeout : float, optional
        Passed to `detect_tuning`.

    Returns
    -------
    dict[str, list[tuple[float, scalar]]]
        Mapping of PV name to ordered endpoint/value tuples.
    """
    if isinstance(pvs, (str, bytes)):
        pvs = [pvs]

    out = {}
    for pv in pvs:
        data = get_archive_data(pv, from_time=start, to_time=end)
        values = np.asarray(data.get('values', []))
        sec = np.asarray(data.get('secondsPastEpoch', []), dtype=float)
        nsec = np.asarray(data.get('nanoseconds', data.get('nanosecond', 0)),
                         dtype=float)
        if nsec.ndim == 0:
            nsec = np.full(sec.shape, float(nsec))
        elif nsec.shape != sec.shape:
            raise ValueError(
                f"{pv}: 'nanoseconds' must be scalar or match "
                "'secondsPastEpoch' length"
            )
        if values.shape != sec.shape:
            raise ValueError(
                f"{pv}: time/value length mismatch: "
                f"len(t)={len(sec)} len(y)={len(values)}"
            )
        if sec.size == 0:
            out[pv] = []
            continue

        t = sec + nsec * 1e-9
        mask = np.isfinite(t) & np.isfinite(values)
        t = t[mask]
        values = values[mask]
        if t.size == 0:
            out[pv] = []
            continue

        order = np.argsort(t, kind='mergesort')
        t_sorted = t[order]
        v_sorted = values[order]
        periods = detect_tuning(t_sorted, timeout=timeout)

        endpoint_pairs = []
        for t_on, t_off in periods:
            idx_on = int(np.searchsorted(t_sorted, t_on, side='left'))
            idx_off = int(np.searchsorted(t_sorted, t_off, side='right') - 1)
            v_on = v_sorted[idx_on]
            v_off = v_sorted[idx_off]
            if isinstance(v_on, np.generic):
                v_on = v_on.item()
            if isinstance(v_off, np.generic):
                v_off = v_off.item()
            endpoint_pairs.append((float(t_sorted[idx_on]), v_on))
            endpoint_pairs.append((float(t_sorted[idx_off]), v_off))

        out[pv] = endpoint_pairs

    return out


def plot_tuning_groups(group_pvs, start, end, lock_y=True, timeout=300,
                       group_order=None, colors=None, ax=None):
    """
    Plot tuning periods for groups of PVs on a categorical y-axis.

    Parameters
    ----------
    group_pvs : dict[str, sequence[str]]
        Mapping of group name to PVs in the group.
    start, end : datetime-like or float
        Time bounds passed to `get_archive_data(...)`.
    lock_y : bool, optional
        If True, lock the vertical scale so all categories remain visible.
    timeout : float, optional
        Passed to `detect_tuning`.
    group_order : sequence[str], optional
        Explicit group ordering. Defaults to the mapping order.
    colors : dict[str, color], optional
        Optional mapping of group name to color.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If None, a new figure/axis is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plotted tuning periods.
    """
    if group_order is None:
        group_order = list(group_pvs.keys())

    if ax is None:
        _, ax = plt.subplots()

    if colors is None:
        tableau = list(mcolors.TABLEAU_COLORS.values())
        colors = {name: tableau[i % len(tableau)] for i, name in enumerate(group_order)}

    def combined_event_times(pvs):
        times = []
        pv_data = {}
        for pv in pvs:
            data = get_archive_data(pv, from_time=start, to_time=end)
            pv_data[pv] = data
            times.extend(data['secondsPastEpoch'])
        return times, pv_data

    def _vector_at_time(pv_list, pv_data, t):
        vals = []
        for pv in pv_list:
            data = pv_data[pv]
            t_arr = np.asarray(data.get('secondsPastEpoch', []))
            v_arr = np.asarray(data.get('values', []))
            if t_arr.size == 0 or v_arr.size == 0:
                continue
            idx = np.searchsorted(t_arr, t, side='right') - 1
            if idx < 0:
                idx = 0
            val = float(v_arr[idx])
            if not np.isfinite(val):
                val = 0.0
            vals.append(val)
        return np.asarray(vals, dtype=float)

    tuning_periods_by_group = {}
    group_data = {}
    for name in group_order:
        times, pv_data = combined_event_times(group_pvs.get(name, []))
        group_data[name] = pv_data
        tuning_periods_by_group[name] = detect_tuning(times, timeout=timeout)

    y_positions = {name: i for i, name in enumerate(group_order)}
    height = 0.8

    for name in group_order:
        periods = tuning_periods_by_group.get(name, [])
        y = y_positions[name]
        pv_data = group_data.get(name, {})
        pv_list = [
            pv for pv, data in pv_data.items()
            if np.asarray(data.get('secondsPastEpoch', [])).size > 0
        ]
        prev_vec = None
        if pv_list:
            t0 = min(
                np.asarray(pv_data[pv]['secondsPastEpoch']).min()
                for pv in pv_list
            )
            baseline_vec = _vector_at_time(pv_list, pv_data, t0)
            if baseline_vec.size:
                prev_vec = baseline_vec
        for t_on, t_off in periods:
            start_num = mdates.date2num(dt.datetime.fromtimestamp(t_on))
            end_num = mdates.date2num(dt.datetime.fromtimestamp(t_off))
            if end_num == start_num:
                end_num = start_num + (1.0 / 86400.0)  # 1 second in days
            alpha_i = 0.6
            if pv_list:
                curr_vec = _vector_at_time(pv_list, pv_data, t_off)
                if (prev_vec is not None and curr_vec.size == prev_vec.size
                        and curr_vec.size > 0):
                    rel = symmetric_norm_distance(prev_vec, curr_vec)
                    small_scale = 0.05
                    alpha_i = (0.05 + 0.75 * np.log1p(rel / small_scale)
                               / np.log1p(1.0 / small_scale))
                    alpha_i = max(0.05, min(0.8, alpha_i))
                if curr_vec.size > 0:
                    prev_vec = curr_vec
            ax.broken_barh(
                [(start_num, end_num - start_num)],
                (y - height / 2.0, height),
                facecolors=colors.get(name, 'tab:gray'),
                alpha=alpha_i,
            )

    ax.set_yticks([y_positions[n] for n in group_order])
    ax.set_yticklabels(group_order)
    ax.set_xlabel('Time')
    ax.set_ylabel('Quad group')
    ax.set_title('Tuning periods by quad group')
    ax.xaxis.set_major_formatter(FuncFormatter(first_tick_of_day_with_date(ax)))
    ax.set_xlim(start, end)

    fixed_ylim = (-0.8, len(group_order) - 0.2)
    if lock_y:
        ax.set_ylim(*fixed_ylim)
        ax.set_autoscale_on(False)
        if not hasattr(ax, '_locking'):
            ax._locking = False

        def _lock_axis(_):
            if ax._locking:
                return
            ax._locking = True
            ax.set_ylim(*fixed_ylim)
            ax._locking = False

        ax.callbacks.connect('ylim_changed', _lock_axis)
    else:
        ax.set_ylim(*fixed_ylim)

    fig = ax.figure
    fig.autofmt_xdate()
    return ax


def add_tuning_overlay(ax, tuning_periods=None, timeout=300, color=None,
                       alpha=None, linewidth=0, hide_points=False,
                       shade_range=False, interpolate_line=False):
    """
    Add tuning period triangles and truncated step lines to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    tuning_periods : list of (t_on, t_off), optional
        Precomputed tuning periods. If None, periods are computed per-series
        from the plotted data using `detect_tuning(t, timeout=timeout)`.
    timeout : float, optional
        Passed to `detect_tuning` when `tuning_periods` is None.
    color : color-like or None, optional
        If None, use the plotted series color for each overlay. Otherwise,
        use this color for all overlays.
    alpha : float or None, optional
        Triangle fill alpha. If None, alpha is set per-triangle based on the
        relative change from the previous set point.
    linewidth : float, optional
        Triangle edge line width (0 for no edge).
    hide_points : bool, optional
        If True, hide existing scatter points on the axis (collections) after
        adding the overlay.
    shade_range : bool, optional
        If True, shade the tuning period with axvspan instead of triangles.
    interpolate_line : bool, optional
        If True, draw a diagonal line between successive set points. If False,
        hide the line inside each tuning period (gap between triangles/spans).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the overlays added.
    artists : dict
        Dict with keys 'triangles' and 'steps', each a list of artists.
    """

    if ax is None:
        raise ValueError('ax is required for add_tuning_overlay().')

    # Extract plotted data (scatter + line) from the axis.
    series = []
    for coll in ax.collections:
        try:
            xy = coll.get_offsets()
        except Exception:
            continue
        if xy is None or len(xy) == 0:
            continue
        x_mpl = np.asarray(xy[:, 0])
        y_arr = np.asarray(xy[:, 1])
        mask = np.isfinite(x_mpl) & np.isfinite(y_arr)
        if not mask.any():
            continue
        x_mpl = x_mpl[mask]
        y_arr = y_arr[mask]
        t_arr = x_mpl * 86400.0
        fc = coll.get_facecolor()
        c = color if color is not None else (tuple(fc[0]) if len(fc) else None)
        series.append({'t': t_arr, 'x': x_mpl, 'y': y_arr, 'color': c})
    for line in ax.lines:
        x_mpl = np.asarray(line.get_xdata())
        y_arr = np.asarray(line.get_ydata())
        mask = np.isfinite(x_mpl) & np.isfinite(y_arr)
        if not mask.any():
            continue
        x_mpl = x_mpl[mask]
        y_arr = y_arr[mask]
        t_arr = x_mpl * 86400.0
        c = color if color is not None else line.get_color()
        series.append({'t': t_arr, 'x': x_mpl, 'y': y_arr, 'color': c})

    # Build overlays per-series: triangles + step segments.
    triangles = []
    steps = []

    for s in series:
        t_arr = np.asarray(s['t'])
        x_arr = np.asarray(s['x'])
        y_arr = np.asarray(s['y'])
        if t_arr.size == 0:
            continue
        c = s['color'] if s['color'] is not None else 'k'

        # Compute tuning periods per series unless provided.
        if tuning_periods is None:
            periods = detect_tuning(t_arr, timeout=timeout)
        else:
            periods = tuning_periods

        t_on_list = []
        t_last_list = []
        y_last_list = []

        # If no fixed alpha is given, alpha = percentage change from previous set point
        adaptive_alpha = (alpha is None)

        # Triangles/spans: left edge spans min/max, right vertex at last point in period.
        for t_on, t_off in periods:
            mask = (t_arr >= t_on) & (t_arr <= t_off)
            if not mask.any():
                continue
            y_min = float(y_arr[mask].min())
            y_max = float(y_arr[mask].max())
            idx_last = np.where(mask)[0][-1]
            t_last = t_arr[idx_last]
            y_last = float(y_arr[idx_last])
            x0 = float(x_arr[mask][0])
            x_last = float(x_arr[idx_last])
            if adaptive_alpha:
                last_set_point = y_last_list[-1] if y_last_list else y_arr[0]
                denom = abs(last_set_point) if last_set_point != 0 else 1.0
                rel = abs(y_last - last_set_point) / denom
                small_scale = 0.05
                alpha_i = 0.05 + 0.75 * np.log1p(rel / small_scale) / np.log1p(1.0 / small_scale)
                alpha_i = max(0.05, min(0.8, alpha_i))
            else:
                alpha_i = alpha
            if shade_range:
                span = ax.axvspan(x0, x_last, facecolor=c, alpha=alpha_i,
                                  linewidth=0, zorder=2)
                triangles.append(span)
            else:
                tri = Polygon([(x0, y_min), (x0, y_max), (x_last, y_last)],
                              closed=True, facecolor=c, alpha=alpha_i,
                              linewidth=linewidth, zorder=2)
                ax.add_patch(tri)
                triangles.append(tri)
            t_on_list.append(t_on)
            t_last_list.append(t_last)
            y_last_list.append(y_last)

        # Line segments: flat between tuning regions; optional diagonal within regions.
        step_x = []
        step_y = []
        for i in range(len(t_last_list)):
            x_last = float(x_arr[np.where(t_arr == t_last_list[i])[0][0]])
            x_on = float(x_arr[np.where(t_arr == t_on_list[i])[0][0]])
            # Flat segment between regions (from prev period end to this start).
            if i > 0:
                x_prev_last = float(x_arr[np.where(t_arr == t_last_list[i - 1])[0][0]])
                step_x.extend([x_prev_last, x_on])
                step_y.extend([y_last_list[i - 1], y_last_list[i - 1]])
            # Within tuning region: diagonal or hidden.
            if interpolate_line and i > 0:
                step_x.extend([x_on, x_last])
                step_y.extend([y_last_list[i - 1], y_last_list[i]])
            elif not interpolate_line:
                step_x.append(np.nan)
                step_y.append(np.nan)
        if t_last_list:
            x_end = ax.get_xlim()[1]
            x_last = float(x_arr[np.where(t_arr == t_last_list[-1])[0][0]])
            step_x.extend([x_last, x_end])
            step_y.extend([y_last_list[-1], y_last_list[-1]])
            if not interpolate_line:
                step_x.append(np.nan)
                step_y.append(np.nan)

        if step_x:
            if interpolate_line:
                line = ax.plot(step_x, step_y, color=c, linewidth=1, zorder=1)[0]
            else:
                line = ax.plot(step_x, step_y, drawstyle='steps-post',
                               color=c, linewidth=1, zorder=1)[0]
            steps.append(line)

    # Optionally hide original scatter points for a cleaner overlay-only view.
    if hide_points:
        for coll in ax.collections:
            coll.set_visible(False)

    return ax, {'triangles': triangles, 'steps': steps}

def symmetric_norm_distance(v1, v2, p=2):
    """
    Find the "symmetric percent change" from v1 to v2 using the l_p norm.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    assert v1.shape == v2.shape

    norm = lambda v: np.linalg.norm(v, ord=p)
    denom = norm(v1) + norm(v2)
    if denom == 0:
        return 0.0
    return 2 * norm(v1 - v2) / denom
