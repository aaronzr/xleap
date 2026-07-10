"""Vertically-stacked sparkline figures for archive PV payloads."""

from __future__ import annotations

import datetime as dt
import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from sparklines_v2.plot.series import (
    add_tuning_overlay,
    finite_rolling_avg,
    first_tick_of_day_with_date,
)


MEASUREMENT_ROLLING_AVG_WINDOW_SECONDS = 15 * 60


def make_vertical_subfig_axes(
    n, figsize=(10, 6), hspace=0.05, sharex=True, height_ratios=None
):
    """Create a figure with ``n`` vertically stacked axes."""
    if n < 1:
        raise ValueError("n must be >= 1")

    gridspec_kw = {"hspace": hspace}
    if height_ratios is not None:
        if len(height_ratios) != n:
            raise ValueError("height_ratios must match n")
        gridspec_kw["height_ratios"] = height_ratios

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=figsize,
        sharex=sharex,
        constrained_layout=False,
        gridspec_kw=gridspec_kw,
    )

    if n == 1:
        return [axes]

    return list(axes)


def sparklines(
    pv_data: list[dict],
    start,
    end,
    hide_points=True,
    y_lim_init=None,
    y_ticks=True,
    min_subplot_height=1,
    change_tol=0.0,
) -> None:
    """Plot sparklines for MEME archive-style PV payloads."""
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    plotted_data = []
    height_ratios = []
    for data in pv_data:
        sec = np.asarray(data["secondsPastEpoch"], dtype=float)
        nsec = np.asarray(data.get("nanoseconds", data.get("nanosecond", 0)), dtype=float)
        if nsec.ndim == 0:
            t = sec + float(nsec) * 1e-9
        else:
            t = sec + nsec * 1e-9
        y = np.asarray(data["values"], dtype=float)
        in_window = (t >= start_ts) & (t <= end_ts)
        y_window = y[in_window]
        if y_window.size < 2:
            continue
        varies = float(np.max(y_window) - np.min(y_window)) > float(change_tol)
        plotted_data.append((data, t, y, varies))
        height_ratios.append(1.0 if varies else 1.0 / 3.0)

    if not plotted_data:
        print("No PVs had at least two samples in the selected interval.")
        return

    fig_height = max(8.0, float(min_subplot_height) * sum(height_ratios))
    axes = make_vertical_subfig_axes(
        len(plotted_data),
        figsize=(12, fig_height),
        hspace=0,
        sharex=True,
        height_ratios=height_ratios,
    )
    color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS.keys())
    for ax, (data, t, y, _varies) in zip(axes, plotted_data):
        t_dt = [dt.datetime.fromtimestamp(ts) for ts in t]

        color = next(color_cycle)
        if bool(data.get("measurement", False)):
            avg_t, avg_y, _ = finite_rolling_avg(
                data,
                MEASUREMENT_ROLLING_AVG_WINDOW_SECONDS,
            )
            avg_t_dt = [dt.datetime.fromtimestamp(ts) for ts in avg_t]
            ax.plot(avg_t_dt, avg_y, color=color, linewidth=1.4, label=data["name"])
        else:
            ax.scatter(t_dt, y, marker="x", s=14, color=color, label=data["name"])
        if y_lim_init is not None:
            y_min = min(float(y_lim_init[0]), float(np.min(y)))
            y_max = max(float(y_lim_init[1]), float(np.max(y)))
            ax.set_ylim(y_min, y_max)
            tick_start = int(np.ceil(y_min))
            tick_stop = int(np.floor(y_max))
            if tick_start <= tick_stop:
                ax.set_yticks(np.arange(tick_start, tick_stop + 1, dtype=float))
        if y_ticks:
            ax.legend(loc="upper left", fontsize=9, frameon=False)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            label = (
                data["name"].rsplit(" (", 1)[0]
                if data["name"].endswith(" PVs)")
                else data["name"]
            )
            ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.xaxis.grid(True, alpha=0.25)
        ax.yaxis.grid(True, alpha=0.25)

        if not bool(data.get("measurement", False)):
            add_tuning_overlay(ax, hide_points=hide_points)

    for ax in axes[1:]:
        ax.spines["top"].set_visible(False)

    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(labelbottom=False)

    axes[-1].set_xlim(start, end)
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(first_tick_of_day_with_date(axes[-1])))
    axes[-1].figure.autofmt_xdate()
    plt.show()
