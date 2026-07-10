"""Pure matplotlib rendering: sparklines, percentile bands, tuning overlays."""

from sparklines_v2.plot.percentile_band import (
    compute_percentile_band_series,
    plot_percentile_band,
    render_percentile_band_series,
)
from sparklines_v2.plot.series import (
    add_pv_series,
    add_tuning_overlay,
    finite_rolling_avg,
    first_tick_of_day_with_date,
    get_archive_data,
    plot_on_axis,
    plot_pvs,
    plot_tuning_groups,
    rolling_avg,
    symmetric_norm_distance,
)
from sparklines_v2.plot.sparklines import (
    make_vertical_subfig_axes,
    sparklines,
)
from sparklines_v2.plot.tuning import (
    detect_tuning,
    get_tuning_events,
    pv_tuning_json,
)


__all__ = [
    "add_pv_series",
    "add_tuning_overlay",
    "compute_percentile_band_series",
    "detect_tuning",
    "finite_rolling_avg",
    "first_tick_of_day_with_date",
    "get_archive_data",
    "get_tuning_events",
    "make_vertical_subfig_axes",
    "plot_on_axis",
    "plot_percentile_band",
    "plot_pvs",
    "plot_tuning_groups",
    "pv_tuning_json",
    "render_percentile_band_series",
    "rolling_avg",
    "sparklines",
    "symmetric_norm_distance",
]
