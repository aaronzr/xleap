"""Operator UI scaffold package."""

from .sparklines_hierarchy import (
    DEFAULT_MONITOR_SPECS,
    build_default_composite_hierarchy,
    build_composite_hierarchy,
    load_pv_groups,
)
from .sparklines_plotting import (
    make_vertical_subfig_axes,
    plot_percentile_band,
    sparklines,
)
from .sparklines_viewer import HierarchySparklineViewer

__all__ = [
    "DEFAULT_MONITOR_SPECS",
    "HierarchySparklineViewer",
    "build_default_composite_hierarchy",
    "build_composite_hierarchy",
    "load_pv_groups",
    "make_vertical_subfig_axes",
    "plot_percentile_band",
    "sparklines",
]
