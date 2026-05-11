"""Interactive hierarchy viewer extracted from ``sparklines.ipynb``."""

from __future__ import annotations

import datetime as dt
import itertools
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import CheckButtons

try:
    from .sparklines_hierarchy import _extract_time_and_values, _series_varies_in_window
    from .sparklines_plotting import (
        compute_percentile_band_series,
        plot_percentile_band,
        render_percentile_band_series,
    )
    from .sparklines_plot_utils import add_tuning_overlay, first_tick_of_day_with_date
except ImportError:  # pragma: no cover - notebook/script fallback
    from sparklines_hierarchy import _extract_time_and_values, _series_varies_in_window
    from sparklines_plotting import (
        compute_percentile_band_series,
        plot_percentile_band,
        render_percentile_band_series,
    )
    from sparklines_plot_utils import add_tuning_overlay, first_tick_of_day_with_date


class HierarchySparklineViewer:
    """Interactive sparkline viewer with click-to-drill hierarchy navigation."""

    MAX_MONITOR_RENDER_POINTS = 1000

    def __init__(
        self,
        hierarchy: dict,
        start_time,
        end_time,
        draw_report_path: str | Path | None = "sparklines_draw_report.txt",
        figure=None,
    ):
        self.hierarchy = hierarchy
        self.path = []
        self.fig = figure
        self.pick_cid = None
        self.click_cid = None
        self.key_cid = None
        self.resize_cid = None
        self.label_targets = {}
        self.axis_targets = {}
        self.breadcrumb_targets = {}
        self.show_data_points = False
        self.filter_sxr = True
        self.filter_hxr = True
        self.filter_cu = True
        self.filter_sc = True
        self.start_time = start_time
        self.end_time = end_time
        self.initial_xlim = tuple(mdates.date2num(value) for value in (start_time, end_time))
        self.current_xlim = tuple(self.initial_xlim)
        self._tracking_limits_enabled = False
        self._pending_filter_ylims = None
        self.focused_monitor_label = None
        self.show_points_checkbox = None
        self.beamline_filter_checkbox = None
        self.accelerator_filter_checkbox = None
        self.last_draw_report = {"items": [], "total_seconds": 0.0}
        self.draw_report_path = (
            None if draw_report_path is None else Path(draw_report_path)
        )
        self._monitor_band_cache = {}
        self._monitor_render_cache = {}
        self._plot_axes_cache = []
        self._header_axes = []
        self._figure_text_artists = []
        self._controls_canvas_size = None
        self._handling_resize = False

    def _normalize_beam_paths(self, beam_paths) -> tuple[str, ...]:
        if beam_paths is None:
            return ()
        if isinstance(beam_paths, str):
            raw_entries = beam_paths.split(",")
        elif isinstance(beam_paths, (list, tuple, set)):
            raw_entries = []
            for entry in beam_paths:
                if isinstance(entry, str):
                    raw_entries.extend(entry.split(","))
        else:
            return ()

        normalized = []
        for entry in raw_entries:
            token = str(entry).strip().upper()
            if token and token not in normalized:
                normalized.append(token)
        return tuple(normalized)

    def _beam_path_matches_filters(self, beam_paths) -> bool:
        selected_lines = set()
        if self.filter_sxr:
            selected_lines.add("SXR")
        if self.filter_hxr:
            selected_lines.add("HXR")

        selected_accelerators = set()
        if self.filter_cu:
            selected_accelerators.add("CU")
        if self.filter_sc:
            selected_accelerators.add("SC")

        if not selected_lines or not selected_accelerators:
            return False

        normalized_paths = self._normalize_beam_paths(beam_paths)
        if not normalized_paths:
            return False

        for beam_path in normalized_paths:
            parts = set(beam_path.split("_"))
            accelerator = "CU" if "CU" in parts else "SC" if "SC" in parts else None
            line = "SXR" if "SXR" in parts else "HXR" if "HXR" in parts else None

            if selected_accelerators and accelerator not in selected_accelerators:
                continue
            if selected_lines and line not in selected_lines:
                continue
            return True

        return False

    def _monitor_order(self):
        preferred = ("GMD", "XGMD", "GDET 241", "GDET 361")
        labels = list(self.hierarchy.get("monitor_pvs", {}).keys())
        ordered = [label for label in preferred if label in labels]
        ordered.extend(label for label in labels if label not in ordered)
        return tuple(ordered)

    def _current_canvas_size_inches(self, fallback_height: float) -> tuple[float, float]:
        if self.fig is None:
            return 12.0, fallback_height

        canvas = getattr(self.fig, "canvas", None)
        dpi = float(self.fig.get_dpi() or 100.0)
        if canvas is None:
            return 12.0, fallback_height

        pixel_width = 0
        pixel_height = 0
        if hasattr(canvas, "width") and hasattr(canvas, "height"):
            try:
                pixel_width = int(canvas.width())
                pixel_height = int(canvas.height())
            except Exception:
                pixel_width = 0
                pixel_height = 0

        if pixel_width <= 0 or pixel_height <= 0:
            try:
                pixel_width, pixel_height = canvas.get_width_height()
            except Exception:
                pixel_width = 0
                pixel_height = 0

        width_inches = max(pixel_width / dpi, 6.0) if pixel_width > 0 else 12.0
        height_inches = (
            max(pixel_height / dpi, 4.0) if pixel_height > 0 else fallback_height
        )
        return width_inches, height_inches

    def _monitor_items(self):
        items = []
        monitor_pvs = self.hierarchy.get("monitor_pvs", {})
        for label in self._monitor_order():
            monitor = monitor_pvs.get(label)
            if not monitor:
                continue
            item = {
                "label": label,
                "data": monitor,
                "drill_to": None,
                "beam_paths": monitor.get("beam_paths", ()),
                "kind": "monitor",
                "value_scale": float(monitor.get("value_scale", 1.0)),
                "subsample": bool(monitor.get("subsample", True)),
            }
            if self._beam_path_matches_filters(item.get("beam_paths")):
                items.append(item)
        return items

    def _prepend_monitor_items(self, items):
        return self._monitor_items() + items

    def _monitor_band_cache_key(self, item):
        data = item["data"]
        return (
            id(data),
            float(item.get("value_scale", 1.0)),
        )

    def _get_monitor_band(self, item):
        cache_key = self._monitor_band_cache_key(item)
        band = self._monitor_band_cache.get(cache_key)
        if band is None:
            band = compute_percentile_band_series(
                item["data"],
                value_scale=float(item.get("value_scale", 1.0)),
            )
            self._monitor_band_cache[cache_key] = band
        return band

    def _get_rendered_monitor_band(self, item):
        render_max_points = (
            self.MAX_MONITOR_RENDER_POINTS if item.get("subsample", True) else None
        )
        cache_key = (self._monitor_band_cache_key(item), render_max_points)
        rendered = self._monitor_render_cache.get(cache_key)
        if rendered is None:
            rendered = render_percentile_band_series(
                self._get_monitor_band(item),
                render_max_points=render_max_points,
            )
            self._monitor_render_cache[cache_key] = rendered
        return rendered

    def _items_for_path(self):
        if self.focused_monitor_label is not None:
            focused = next(
                (
                    item
                    for item in self._monitor_items()
                    if item["label"] == self.focused_monitor_label
                ),
                None,
            )
            return [focused] if focused is not None else []

        if len(self.path) == 0:
            items = []
            for group_name, group in sorted(self.hierarchy["groups"].items()):
                comp = group.get("composite")
                if comp is None:
                    continue
                items.append(
                    {
                        "label": group_name,
                        "data": comp,
                        "drill_to": [group_name],
                        "beam_paths": group.get("beam_paths", ()),
                        "kind": "composite",
                        "value_scale": 1.0,
                    }
                )
            visible = [
                item
                for item in items
                if self._beam_path_matches_filters(item.get("beam_paths"))
            ]
            return self._prepend_monitor_items(visible)

        if len(self.path) == 1:
            group_name = self.path[0]
            group = self.hierarchy["groups"].get(group_name, {})
            items = []
            for subgroup_name, subgroup in sorted(group.get("subgroups", {}).items()):
                comp = subgroup.get("composite")
                if comp is None:
                    continue
                items.append(
                    {
                        "label": subgroup_name,
                        "data": comp,
                        "drill_to": [group_name, subgroup_name],
                        "beam_paths": subgroup.get("beam_paths", ()),
                        "kind": "composite",
                        "value_scale": 1.0,
                    }
                )
            visible = [
                item
                for item in items
                if self._beam_path_matches_filters(item.get("beam_paths"))
            ]
            return self._prepend_monitor_items(visible)

        if len(self.path) == 2:
            group_name, subgroup_name = self.path
            subgroup = (
                self.hierarchy["groups"]
                .get(group_name, {})
                .get("subgroups", {})
                .get(subgroup_name, {})
            )
            items = []
            for pv_data in subgroup.get("pv_data", []):
                pv_name = pv_data.get("name")
                if not pv_name:
                    continue
                items.append(
                    {
                        "label": pv_name,
                        "data": pv_data,
                        "drill_to": [group_name, subgroup_name, pv_name],
                        "beam_paths": pv_data.get(
                            "beam_paths", subgroup.get("beam_paths", ())
                        ),
                        "kind": "pv",
                        "value_scale": 1.0,
                    }
                )
            visible = [
                item
                for item in items
                if self._beam_path_matches_filters(item.get("beam_paths"))
            ]
            return self._prepend_monitor_items(visible)

        group_name, subgroup_name, pv_name = self.path
        subgroup = (
            self.hierarchy["groups"]
            .get(group_name, {})
            .get("subgroups", {})
            .get(subgroup_name, {})
        )
        pv_data = next(
            (pv for pv in subgroup.get("pv_data", []) if pv.get("name") == pv_name),
            None,
        )
        if pv_data is None:
            return []

        item = {
            "label": pv_data["name"],
            "data": pv_data,
            "drill_to": None,
            "beam_paths": pv_data.get("beam_paths", subgroup.get("beam_paths", ())),
            "kind": "pv",
            "value_scale": 1.0,
        }
        return [item] if self._beam_path_matches_filters(item.get("beam_paths")) else []

    def export_navigation_state(self) -> dict[str, object]:
        return {
            "path": list(self.path),
            "focused_monitor_label": self.focused_monitor_label,
            "show_data_points": self.show_data_points,
            "filter_sxr": self.filter_sxr,
            "filter_hxr": self.filter_hxr,
            "filter_cu": self.filter_cu,
            "filter_sc": self.filter_sc,
        }

    def _normalized_path_for_restore(self, path) -> list[str]:
        candidate = list(path or [])
        groups = self.hierarchy.get("groups", {})

        if len(candidate) >= 1 and candidate[0] not in groups:
            return []

        if len(candidate) >= 2:
            subgroups = groups.get(candidate[0], {}).get("subgroups", {})
            if candidate[1] not in subgroups:
                return candidate[:1]

        if len(candidate) >= 3:
            pv_data = (
                groups.get(candidate[0], {})
                .get("subgroups", {})
                .get(candidate[1], {})
                .get("pv_data", [])
            )
            pv_names = {pv.get("name") for pv in pv_data}
            if candidate[2] not in pv_names:
                return candidate[:2]

        return candidate[:3]

    def restore_navigation_state(self, state: dict[str, object] | None) -> None:
        state = state or {}
        self.show_data_points = bool(state.get("show_data_points", self.show_data_points))
        self.filter_sxr = bool(state.get("filter_sxr", self.filter_sxr))
        self.filter_hxr = bool(state.get("filter_hxr", self.filter_hxr))
        self.filter_cu = bool(state.get("filter_cu", self.filter_cu))
        self.filter_sc = bool(state.get("filter_sc", self.filter_sc))
        monitor_label = state.get("focused_monitor_label")
        monitor_labels = set(self.hierarchy.get("monitor_pvs", {}).keys())
        self.focused_monitor_label = (
            str(monitor_label) if monitor_label in monitor_labels else None
        )
        if self.focused_monitor_label is not None:
            self.path = []
            return

        self.path = self._normalized_path_for_restore(state.get("path"))

    def _title(self):
        if len(self.path) == 0:
            return "All PV Groups"
        if len(self.path) == 1:
            return f"{self.path[0]}"
        if len(self.path) == 2:
            return f"{self.path[0]} / {self.path[1]}"
        return f"{self.path[0]} / {self.path[1]} / {self.path[2]}"

    def _report_hierarchy_path(self) -> str:
        segments = list(self.path)
        if self.focused_monitor_label is not None:
            segments.append(f"[monitor] {self.focused_monitor_label}")
        if not segments:
            return "All PV Groups"
        return " / ".join(str(segment) for segment in segments)

    def _write_draw_report(self):
        if self.draw_report_path is None:
            return

        self.draw_report_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        report_lines = [
            f"{timestamp} | path={self._report_hierarchy_path()} | total={self.last_draw_report['total_seconds']:.4f}s"
        ]
        for item in self.last_draw_report["items"]:
            report_lines.append(
                "  "
                f"{item['kind']} | {item['label']} | draw={item['draw_seconds']:.4f}s | "
                f"points={item['rendered_points']}/{item['original_points']}"
            )
        with self.draw_report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines) + "\n")

    def _breadcrumb_segments(self):
        if self.focused_monitor_label is not None:
            segments = [("All PV Groups", [])]
            for depth, label in enumerate(self.path, start=1):
                segments.append((label, self.path[:depth]))
            segments.append((self.focused_monitor_label, None))
            return segments

        if len(self.path) == 0:
            return [("All PV Groups", None)]

        segments = [("All PV Groups", [])]
        for depth, label in enumerate(self.path, start=1):
            target = self.path[:depth] if depth < len(self.path) else None
            segments.append((label, target))
        return segments

    def _draw_breadcrumbs(self):
        self.breadcrumb_targets = {}
        segments = self._breadcrumb_segments()
        parts = []
        for idx, (label, target) in enumerate(segments):
            parts.append((label, target))
            if idx < len(segments) - 1:
                parts.append((" / ", None))

        title_ax = self.fig.add_axes([0.08, 0.93, 0.84, 0.05])
        title_ax.set_axis_off()
        self._header_axes.append(title_ax)

        renderer = self.fig.canvas.get_renderer()
        if renderer is None:
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()

        probes = [
            title_ax.text(
                0,
                0.5,
                text,
                fontsize=13,
                fontweight="semibold",
                ha="left",
                va="center",
                alpha=0.0,
            )
            for text, _target in parts
        ]
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        title_width = max(title_ax.get_window_extent(renderer=renderer).width, 1.0)
        widths = [
            probe.get_window_extent(renderer=renderer).width / title_width
            for probe in probes
        ]
        for probe in probes:
            probe.remove()

        x = 0.5 - sum(widths) / 2
        for (text, target), width in zip(parts, widths):
            clickable = target is not None
            artist = title_ax.text(
                x,
                0.5,
                text,
                fontsize=13,
                fontweight="semibold",
                ha="left",
                va="center",
                color="tab:blue" if clickable else "black",
                transform=title_ax.transAxes,
            )
            if clickable:
                artist.set_picker(True)
                self.breadcrumb_targets[artist] = list(target)
            x += width

    def _is_leaf_level(self):
        return self.focused_monitor_label is not None or len(self.path) == 3

    def _toolbar_mode_active(self):
        if self.fig is None:
            return False
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is None:
            return False
        mode = str(getattr(toolbar, "mode", "") or "")
        return bool(mode.strip())

    def _plot_axes(self):
        return list(self._plot_axes_cache)

    def _path_key(self):
        return tuple(self.path)

    def _window_date_nums(self):
        return sorted(float(value) for value in self.current_xlim)

    def _item_varies_in_window(self, item) -> bool:
        window_start_num, window_end_num = self._window_date_nums()
        if item.get("kind") == "monitor":
            band = self._get_rendered_monitor_band(item)
            t_num = mdates.date2num(band["time"])
            y = np.asarray(band["center"], dtype=float)
            keep = np.isfinite(t_num) & np.isfinite(y)
            return _series_varies_in_window(
                t_num[keep], y[keep], window_start_num, window_end_num
            )
        t, y = _extract_time_and_values(item["data"])
        t_num = mdates.date2num([dt.datetime.fromtimestamp(ts) for ts in t])
        return _series_varies_in_window(t_num, y, window_start_num, window_end_num)

    def _height_ratios_for_items(self, items):
        height_ratios = []
        for item in items:
            varies = self._item_varies_in_window(item)
            height_ratios.append(1.0 if varies else 1.0 / 3.0)
        return height_ratios

    def _item_key(self, item) -> tuple:
        if item.get("kind") == "monitor":
            return ("__monitor__", item["label"])
        drill_to = item.get("drill_to")
        if drill_to is None:
            return tuple(self.path)
        return tuple(drill_to)

    def _capture_view_limits(self, include_ylim=False):
        axes = self._plot_axes()
        if not axes:
            return
        self.current_xlim = tuple(axes[0].get_xlim())
        if include_ylim:
            self._pending_filter_ylims = {
                self._item_key(item): tuple(ax.get_ylim())
                for ax, item in zip(axes, self._items_for_path())
            }

    def _apply_view_limits(self, axes, items=None):
        pending_ylims = self._pending_filter_ylims or {}
        items = items or []
        for index, ax in enumerate(axes):
            ax.set_xlim(*self.current_xlim)
            if index < len(items):
                key = self._item_key(items[index])
                if key in pending_ylims:
                    ax.set_ylim(*pending_ylims[key])

    def _track_current_xlim(self, ax):
        if not self._tracking_limits_enabled:
            return
        self.current_xlim = tuple(ax.get_xlim())

    def _connect_limit_callbacks(self, axes):
        for ax in axes:
            ax.callbacks.connect("xlim_changed", self._track_current_xlim)

    def _reset_original_view(self, *_args):
        axes = self._plot_axes()
        self.current_xlim = tuple(self.initial_xlim)
        self._pending_filter_ylims = None
        if len(axes) == 0:
            return
        self._apply_view_limits(axes, self._items_for_path())
        self.fig.canvas.draw_idle()

    def _install_toolbar_home_hook(self):
        if self.fig is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is None or getattr(toolbar, "_hierarchy_original_view_hook", False):
            return

        original_home = toolbar.home

        def _home(*args, **kwargs):
            result = original_home(*args, **kwargs)
            self._reset_original_view()
            return result

        toolbar.home = _home
        toolbar._hierarchy_original_view_hook = True

    def _trigger_toolbar_home(self):
        if self.fig is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is None:
            self._reset_original_view()
            return
        toolbar.home()

    def _navigate_to_item(self, item):
        if self._is_leaf_level():
            return
        if not item:
            return

        self._capture_view_limits()
        self._pending_filter_ylims = None
        if item.get("kind") == "monitor":
            self.focused_monitor_label = item["label"]
        elif item.get("drill_to") is not None:
            self.path = list(item["drill_to"])
        else:
            return
        self.draw()

    def _draw_controls(self):
        if self.fig is None:
            return
        current_canvas_size = self._canvas_pixel_size()
        existing_widgets = (
            self.show_points_checkbox,
            self.beamline_filter_checkbox,
            self.accelerator_filter_checkbox,
        )
        if all(
            widget is not None and getattr(widget.ax, "figure", None) is self.fig
            for widget in existing_widgets
        ) and self._controls_canvas_size == current_canvas_size:
            return

        self._teardown_controls()
        self.show_points_checkbox = None
        self.beamline_filter_checkbox = None
        self.accelerator_filter_checkbox = None
        row_y = 0.835
        row_height = 0.06
        widths = [0.20, 0.11, 0.10]
        gap = 0.03
        row_width = sum(widths) + gap * (len(widths) - 1)
        row_x = 0.5 - row_width / 2

        checkbox_ax = self.fig.add_axes([row_x, row_y, widths[0], row_height])
        checkbox_ax.set_frame_on(False)
        self.show_points_checkbox = CheckButtons(
            checkbox_ax, ["Show data points"], [self.show_data_points]
        )
        self.show_points_checkbox.on_clicked(self._on_toggle_show_data_points)

        beamline_ax = self.fig.add_axes(
            [row_x + widths[0] + gap, row_y, widths[1], row_height]
        )
        beamline_ax.set_frame_on(False)
        self.beamline_filter_checkbox = CheckButtons(
            beamline_ax,
            ["SXR", "HXR"],
            [self.filter_sxr, self.filter_hxr],
        )
        self.beamline_filter_checkbox.on_clicked(self._on_toggle_beamline_filter)

        accelerator_ax = self.fig.add_axes(
            [row_x + widths[0] + widths[1] + 2 * gap, row_y, widths[2], row_height]
        )
        accelerator_ax.set_frame_on(False)
        self.accelerator_filter_checkbox = CheckButtons(
            accelerator_ax,
            ["Cu", "SC"],
            [self.filter_cu, self.filter_sc],
        )
        self.accelerator_filter_checkbox.on_clicked(self._on_toggle_accelerator_filter)
        self._controls_canvas_size = current_canvas_size

    def _schedule_draw(self):
        self.draw()

    def _teardown_controls(self):
        for attr in (
            "show_points_checkbox",
            "beamline_filter_checkbox",
            "accelerator_filter_checkbox",
        ):
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            disconnect = getattr(widget, "disconnect_events", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass
            widget_ax = getattr(widget, "ax", None)
            if widget_ax is not None:
                try:
                    widget_ax.remove()
                except Exception:
                    pass
            setattr(self, attr, None)
        self._controls_canvas_size = None

    def _canvas_pixel_size(self) -> tuple[int, int] | None:
        if self.fig is None or getattr(self.fig, "canvas", None) is None:
            return None
        try:
            width, height = self.fig.canvas.get_width_height()
        except Exception:
            return None
        return int(width), int(height)

    def _on_resize(self, _event):
        if self._handling_resize:
            return
        current_canvas_size = self._canvas_pixel_size()
        if current_canvas_size is None or current_canvas_size == self._controls_canvas_size:
            return
        self._handling_resize = True
        try:
            self.draw()
        finally:
            self._handling_resize = False

    def _clear_dynamic_layout(self):
        for ax in self._plot_axes_cache:
            try:
                ax.remove()
            except Exception:
                pass
        self._plot_axes_cache = []

        for ax in self._header_axes:
            try:
                ax.remove()
            except Exception:
                pass
        self._header_axes = []

        for artist in self._figure_text_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._figure_text_artists = []

    def _add_figure_text(self, *args, **kwargs):
        artist = self.fig.text(*args, **kwargs)
        self._figure_text_artists.append(artist)
        return artist

    def _apply_time_tick_label_layout(self, axis):
        view_start, view_end = axis.get_xlim()
        view_hours = max((view_end - view_start) * 24.0, 0.0)
        rotate_labels = view_hours >= 24.0
        rotation = 35 if rotate_labels else 0

        axis.tick_params(axis="x", labelrotation=rotation)
        for label in axis.get_xticklabels():
            label.set_rotation(rotation)
            label.set_rotation_mode("anchor")
            label.set_ha("right" if rotate_labels else "center")

        return rotate_labels

    def draw(self):
        draw_started = time.perf_counter()
        items = self._items_for_path()
        self.last_draw_report = {"items": [], "total_seconds": 0.0}
        item_varies_flags = [self._item_varies_in_window(item) for item in items]

        self._tracking_limits_enabled = False
        pv_level = len(self.path) >= 2 and self.focused_monitor_label is None
        height_ratios = (
            [1.0 if varies else 1.0 / 3.0 for varies in item_varies_flags]
            if items
            else [1.0]
        )
        fig_height = max(7.0, 1.3 * sum(height_ratios))
        hspace = 0
        fig_width, fig_height = self._current_canvas_size_inches(fig_height)

        if self.fig is None:
            self.fig = plt.figure(figsize=(fig_width, fig_height))
        else:
            self.fig.set_size_inches(fig_width, fig_height, forward=True)

        if self.pick_cid is None:
            self.pick_cid = self.fig.canvas.mpl_connect("pick_event", self._on_pick)
            self.click_cid = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_click
            )
            self.resize_cid = self.fig.canvas.mpl_connect(
                "resize_event", self._on_resize
            )

        self._draw_controls()
        self._clear_dynamic_layout()

        if not items:
            self.label_targets = {}
            self.axis_targets = {}
            self.breadcrumb_targets = {}
            self.fig.subplots_adjust(top=0.78)
            self._draw_breadcrumbs()
            self._add_figure_text(
                0.5,
                0.50,
                "No PVs or groups match your selected filters.",
                fontsize=10,
                ha="center",
            )
            self._add_figure_text(
                0.5,
                0.90,
                "Use the Beam_Path checkboxes to expand or narrow the current view.",
                fontsize=9,
                ha="center",
            )
            self._install_toolbar_home_hook()
            self.last_draw_report["total_seconds"] = time.perf_counter() - draw_started
            self._write_draw_report()
            self.fig.canvas.draw_idle()
            return

        axes = self.fig.subplots(
            nrows=len(items),
            ncols=1,
            sharex=True,
            gridspec_kw={"hspace": hspace, "height_ratios": height_ratios},
        )
        if len(items) == 1:
            axes = [axes]
        self._plot_axes_cache = list(axes)

        self.label_targets = {}
        self.axis_targets = {}
        self.breadcrumb_targets = {}
        color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS.keys())

        for ax, item, item_varies in zip(axes, items, item_varies_flags):
            item_draw_started = time.perf_counter()
            color = next(color_cycle)
            is_monitor = item.get("kind") == "monitor"
            clickable = item.get("drill_to") is not None or (
                is_monitor and self.focused_monitor_label is None
            )

            if clickable:
                self.axis_targets[ax] = item

            if is_monitor:
                original_points = int(np.asarray(item["data"].get("values", ())).size)
                if original_points == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"No samples for {item['label']}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    self.last_draw_report["items"].append(
                        {
                            "label": item["label"],
                            "kind": item.get("kind"),
                            "original_points": 0,
                            "rendered_points": 0,
                            "draw_seconds": time.perf_counter() - item_draw_started,
                        }
                    )
                    continue
                precomputed_band = self._get_monitor_band(item)
                _, band_data = plot_percentile_band(
                    item["data"],
                    ax=ax,
                    y_label=None,
                    x_label=None,
                    show_legend=False,
                    render_max_points=(
                        self.MAX_MONITOR_RENDER_POINTS if item.get("subsample", True) else None
                    ),
                    precomputed_band=precomputed_band,
                )
                rendered_points = int(band_data.get("rendered_points", original_points))
                if item_varies:
                    ax.tick_params(axis="y", which="both", left=True, labelleft=True)
                else:
                    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                    ax.set_yticks([])
                ylab = ax.set_ylabel(
                    item["label"],
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                    labelpad=28,
                    color="tab:blue" if clickable else "black",
                )
                if clickable:
                    ylab.set_picker(True)
                    self.label_targets[ylab] = item
            elif pv_level:
                t, y = _extract_time_and_values(item["data"])
                if t.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"No samples for {item['label']}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    self.last_draw_report["items"].append(
                        {
                            "label": item["label"],
                            "kind": item.get("kind"),
                            "original_points": 0,
                            "rendered_points": 0,
                            "draw_seconds": time.perf_counter() - item_draw_started,
                        }
                    )
                    continue
                t_dt = [dt.datetime.fromtimestamp(ts) for ts in t]
                original_points = int(y.size)
                rendered_points = original_points
                display_y = y * float(item.get("value_scale", 1.0))
                points = ax.scatter(t_dt, display_y, marker="x", s=14, color=color)
                ax.tick_params(axis="y", which="both", left=True, labelleft=True)
                ax.set_ylabel("")
                legend = ax.legend(
                    [points],
                    [item["label"]],
                    loc="upper left",
                    fontsize=8,
                    frameon=False,
                    handletextpad=0.3,
                    borderpad=0.1,
                )
                if clickable:
                    points.set_picker(True)
                    self.label_targets[points] = item
                    for text_artist in legend.get_texts():
                        text_artist.set_picker(True)
                        self.label_targets[text_artist] = item
            else:
                t, y = _extract_time_and_values(item["data"])
                if t.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"No samples for {item['label']}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    self.last_draw_report["items"].append(
                        {
                            "label": item["label"],
                            "kind": item.get("kind"),
                            "original_points": 0,
                            "rendered_points": 0,
                            "draw_seconds": time.perf_counter() - item_draw_started,
                        }
                    )
                    continue
                t_dt = [dt.datetime.fromtimestamp(ts) for ts in t]
                original_points = int(y.size)
                rendered_points = original_points
                display_y = y * float(item.get("value_scale", 1.0))
                points = ax.scatter(t_dt, display_y, marker="x", s=14, color=color)
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ylab = ax.set_ylabel(
                    item["label"],
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                    labelpad=28,
                    color="tab:blue" if clickable else "black",
                )
                if clickable:
                    ylab.set_picker(True)
                    self.label_targets[ylab] = item

            ax.xaxis.grid(True, alpha=0.25)
            ax.yaxis.grid(True, alpha=0.25)
            if not is_monitor:
                add_tuning_overlay(ax, hide_points=not self.show_data_points)
            self.last_draw_report["items"].append(
                {
                    "label": item["label"],
                    "kind": item.get("kind"),
                    "original_points": original_points,
                    "rendered_points": rendered_points,
                    "draw_seconds": time.perf_counter() - item_draw_started,
                }
            )

        for ax in axes[1:]:
            ax.spines["top"].set_visible(False)

        for ax in axes[:-1]:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        axes[-1].set_xlim(*self.initial_xlim)
        axes[-1].xaxis.set_major_formatter(
            FuncFormatter(first_tick_of_day_with_date(axes[-1]))
        )
        self._apply_view_limits(axes, items)
        self._pending_filter_ylims = None
        rotate_labels = self._apply_time_tick_label_layout(axes[-1])

        self.fig.subplots_adjust(top=0.78, bottom=0.16 if rotate_labels else 0.12)
        self._draw_breadcrumbs()
        if self.focused_monitor_label is not None or len(self.path) >= 3:
            instruction = (
                "Single-PV view. Use Show data points, Beam_Path filters, or the "
                "breadcrumb path to navigate."
            )
        elif len(self.path) < 2:
            instruction = (
                "Click a plot or blue y-label to descend. Use the Beam_Path "
                "checkboxes to filter the current view."
            )
        elif len(self.path) == 2:
            instruction = (
                "Click a PV plot, trace, legend label, or monitor plot to isolate "
                "a single PV. Use Show data points and Beam_Path filters to "
                "refine the view."
            )
        else:
            instruction = (
                "Single-PV view. Use Show data points, Beam_Path filters, or the "
                "breadcrumb path to navigate."
            )
        self._add_figure_text(0.5, 0.90, instruction, fontsize=9, ha="center")

        self._install_toolbar_home_hook()
        self._connect_limit_callbacks(axes)
        self._tracking_limits_enabled = True

        self.last_draw_report["total_seconds"] = time.perf_counter() - draw_started
        self._write_draw_report()
        self.fig.canvas.draw_idle()

    def _on_toggle_show_data_points(self, _label):
        self._capture_view_limits()
        self._pending_filter_ylims = None
        self.show_data_points = not self.show_data_points
        self._schedule_draw()

    def _on_toggle_beamline_filter(self, label):
        self._capture_view_limits(include_ylim=True)
        if label == "SXR":
            self.filter_sxr = not self.filter_sxr
        elif label == "HXR":
            self.filter_hxr = not self.filter_hxr
        self._schedule_draw()

    def _on_toggle_accelerator_filter(self, label):
        self._capture_view_limits(include_ylim=True)
        if label == "Cu":
            self.filter_cu = not self.filter_cu
        elif label == "SC":
            self.filter_sc = not self.filter_sc
        self._schedule_draw()

    def _on_click(self, event):
        if self._toolbar_mode_active():
            return
        if event.button != 1 or event.inaxes is None:
            return
        control_axes = [
            getattr(getattr(self, attr, None), "ax", None)
            for attr in (
                "show_points_checkbox",
                "beamline_filter_checkbox",
                "accelerator_filter_checkbox",
            )
        ]
        if event.inaxes in control_axes:
            return

        self._navigate_to_item(self.axis_targets.get(event.inaxes))

    def _on_pick(self, event):
        if self._toolbar_mode_active():
            return

        breadcrumb_target = self.breadcrumb_targets.get(event.artist)
        if breadcrumb_target is not None:
            self._capture_view_limits()
            self._pending_filter_ylims = None
            self.focused_monitor_label = None
            self.path = list(breadcrumb_target)
            self.draw()
            return

        self._navigate_to_item(self.label_targets.get(event.artist))

    def back(self):
        if self.focused_monitor_label is not None:
            self._capture_view_limits()
            self._pending_filter_ylims = None
            self.focused_monitor_label = None
            self.draw()
            return
        if self.path:
            self._capture_view_limits()
            self._pending_filter_ylims = None
            self.path = self.path[:-1]
            self.draw()

    def home(self):
        if self.focused_monitor_label is not None or self.path:
            self._capture_view_limits()
            self._pending_filter_ylims = None
            self.focused_monitor_label = None
            self.path = []
            self.draw()
