"""Tuning-period detection primitives.

Pure-numpy helpers that identify contiguous periods of frequent parameter
changes ("tuning") in a stream of event times. Kept dependency-free of
matplotlib and requests so the detection logic can be unit-tested in
isolation and reused by overlay renderers and JSON exporters.
"""

from __future__ import annotations

import numpy as np


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


def _event_times_from_value_changes(t, y, min_delta=None):
    """Return event times filtered by absolute value change from last setpoint."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.shape != y_arr.shape:
        raise ValueError(
            "time/value length mismatch: "
            f"len(t)={t_arr.size} len(y)={y_arr.size}"
        )
    if t_arr.size == 0:
        return np.array([], dtype=float)

    mask = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_arr = t_arr[mask]
    y_arr = y_arr[mask]
    if t_arr.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(t_arr, kind='mergesort')
    t_sorted = t_arr[order]
    y_sorted = y_arr[order]

    if min_delta is None:
        return t_sorted

    min_delta = float(min_delta)
    if not np.isfinite(min_delta) or min_delta <= 0:
        return t_sorted

    events = []
    last_setpoint = float(y_sorted[0])
    for ti, yi in zip(t_sorted[1:], y_sorted[1:]):
        yi = float(yi)
        if abs(yi - last_setpoint) >= min_delta:
            events.append(float(ti))
            last_setpoint = yi
    return np.asarray(events, dtype=float)


def get_tuning_events(t, timeout=300, values=None):
    """Return tuning periods, or per-period value-statistics summaries."""
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
    """Return per-PV tuning endpoint/value pairs as a JSON-like dict."""
    from sparklines_v2.plot.series import get_archive_data

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
