"""MEME sparklines package: archive fetch, plot rendering, and Qt UI."""

from sparklines_v2.archive import (
    DEFAULT_MONITOR_SPECS,
    build_composite_hierarchy,
    build_default_composite_hierarchy,
    load_pv_groups,
)
from sparklines_v2.plot import (
    make_vertical_subfig_axes,
    plot_percentile_band,
    sparklines,
)
from sparklines_v2.ui import (
    HierarchySparklineViewer,
    SparklineMainWindow,
)


__all__ = [
    "DEFAULT_MONITOR_SPECS",
    "HierarchySparklineViewer",
    "SparklineMainWindow",
    "build_composite_hierarchy",
    "build_default_composite_hierarchy",
    "load_pv_groups",
    "make_vertical_subfig_axes",
    "plot_percentile_band",
    "sparklines",
]
