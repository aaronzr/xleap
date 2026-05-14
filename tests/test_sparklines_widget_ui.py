import datetime as dt
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import app.sparklines_plotting as plotting_module
import app.sparklines_viewer as viewer_module

START = dt.datetime(2026, 2, 25, 6, 0, 0)
END = dt.datetime(2026, 2, 25, 12, 0, 0)


def _noop_add_tuning_overlay(_ax, hide_points=False, **_kwargs):
    return None


def _dummy_time_formatter(_axis):
    return lambda value, pos=None: str(value)


class _StubCheckButtons:
    def __init__(self, ax, labels, actives):
        self.ax = ax
        self.labels = list(labels)
        self.actives = list(actives)
        self._callbacks = []

    def on_clicked(self, callback):
        self._callbacks.append(callback)
        return callback

    def set_active(self, index):
        self.actives[index] = not self.actives[index]
        label = self.labels[index]
        for callback in list(self._callbacks):
            callback(label)


class _FakeToolbar:
    def __init__(self):
        self.mode = ""
        self.home_calls = 0

    def home(self, *args, **kwargs):
        self.home_calls += 1
        return None

    def _wait_cursor_for_draw_cm(self):
        return nullcontext()

    def update(self):
        return None


def _series(name: str, values: list[float], step_seconds: int = 1800) -> dict:
    timestamps = np.array(
        [START.timestamp() + step_seconds * idx for idx in range(len(values))],
        dtype=float,
    )
    seconds = np.floor(timestamps).astype(np.int64)
    nanoseconds = np.rint((timestamps - seconds) * 1e9).astype(np.int64)
    count = len(values)
    return {
        "name": name,
        "secondsPastEpoch": seconds,
        "nanoseconds": nanoseconds,
        "values": np.asarray(values, dtype=float),
        "severity": np.zeros(count, dtype=np.int64),
        "status": np.zeros(count, dtype=np.int64),
    }


def _large_series(name: str, count: int, step_seconds: int = 1) -> dict:
    timestamps = np.array(
        [START.timestamp() + step_seconds * idx for idx in range(count)],
        dtype=float,
    )
    seconds = np.floor(timestamps).astype(np.int64)
    nanoseconds = np.rint((timestamps - seconds) * 1e9).astype(np.int64)
    values = np.linspace(0.0, float(count - 1), num=count, dtype=float)
    return {
        "name": name,
        "secondsPastEpoch": seconds,
        "nanoseconds": nanoseconds,
        "values": values,
        "severity": np.zeros(count, dtype=np.int64),
        "status": np.zeros(count, dtype=np.int64),
    }


def _synthetic_hierarchy() -> dict:
    pv_a1 = _series("PV:A:1", [0.0, 1.0, 2.0, 3.0])
    pv_a2 = _series("PV:A:2", [0.2, 1.1, 1.9, 3.2])
    pv_a3 = _series("PV:A:3", [0.4, 0.8, 1.6, 2.4])
    pv_b1 = _series("PV:B:1", [1.0, 1.5, 1.2, 1.8])
    pv_b2 = _series("PV:B:2", [0.8, 1.4, 1.1, 1.7])

    return {
        "groups": {
            "Group A": {
                "beam_paths": ("CU_SXR", "CU_HXR"),
                "threshold": 1.0,
                "composite": _series("Group A", [0.0, 0.5, 1.0, 1.5]),
                "subgroups": {
                    "Subgroup A1": {
                        "beam_paths": ("CU_SXR",),
                        "threshold": 1.0,
                        "composite": _series("Subgroup A1", [0.0, 0.4, 0.8, 1.2]),
                        "pv_specs": [],
                        "pv_data": [
                            {**pv_a1, "beam_paths": ("CU_SXR",)},
                            {**pv_a2, "beam_paths": ("CU_SXR",)},
                        ],
                    },
                    "Subgroup A2": {
                        "beam_paths": ("CU_HXR",),
                        "threshold": 1.0,
                        "composite": _series("Subgroup A2", [0.1, 0.6, 1.1, 1.4]),
                        "pv_specs": [],
                        "pv_data": [{**pv_a3, "beam_paths": ("CU_HXR",)}],
                    },
                },
            },
            "Group B": {
                "beam_paths": ("SC_SXR", "SC_HXR"),
                "threshold": 1.0,
                "composite": _series("Group B", [0.5, 0.7, 0.9, 1.1]),
                "subgroups": {
                    "Subgroup B1": {
                        "beam_paths": ("SC_SXR", "SC_HXR"),
                        "threshold": 1.0,
                        "composite": _series("Subgroup B1", [0.5, 0.8, 1.0, 1.1]),
                        "pv_specs": [],
                        "pv_data": [
                            {**pv_b1, "beam_paths": ("SC_SXR",)},
                            {**pv_b2, "beam_paths": ("SC_HXR",)},
                        ],
                    },
                },
            },
        }
    }


def _synthetic_hierarchy_with_monitors() -> dict:
    hierarchy = _synthetic_hierarchy()
    hierarchy["monitor_pvs"] = {
        "GMD": {
            **_series("EM1K0:GMD:HPS:milliJoulesPerPulse", [0.0010, 0.0012, 0.0011, 0.0013]),
            "label": "GMD",
            "beam_paths": ("CU_SXR", "SC_SXR"),
            "value_scale": 1000.0,
        },
        "XGMD": {
            **_series("EM2K0:XGMD:HPS:milliJoulesPerPulse", [0.0008, 0.0009, 0.0010, 0.0011]),
            "label": "XGMD",
            "beam_paths": ("CU_SXR", "SC_SXR"),
            "value_scale": 1000.0,
        },
        "GDET 241": {
            **_series("GDET:FEE1:241:ENRC", [0.0015, 0.0014, 0.0016, 0.0017]),
            "label": "GDET 241",
            "beam_paths": ("CU_HXR", "SC_HXR"),
            "value_scale": 1000.0,
        },
        "GDET 361": {
            **_series("GDET:FEE1:361:ENRC", [0.0018, 0.0019, 0.0017, 0.0020]),
            "label": "GDET 361",
            "beam_paths": ("CU_HXR", "SC_HXR"),
            "value_scale": 1000.0,
        },
    }
    return hierarchy


def _synthetic_hierarchy_with_constants() -> dict:
    return {
        "groups": {
            "Constant Group": {
                "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                "threshold": 1.0,
                "composite": _series("Constant Group", [4.0, 4.0, 4.0, 4.0]),
                "subgroups": {
                    "Constant Subgroup": {
                        "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                        "threshold": 1.0,
                        "composite": _series("Constant Subgroup", [2.0, 2.0, 2.0, 2.0]),
                        "pv_specs": [],
                        "pv_data": [
                            {
                                **_series("PV:CONST", [7.0, 7.0, 7.0, 7.0]),
                                "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                            }
                        ],
                    }
                },
            },
            "Varying Group": {
                "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                "threshold": 1.0,
                "composite": _series("Varying Group", [0.0, 1.0, 2.0, 3.0]),
                "subgroups": {
                    "Windowed Subgroup": {
                        "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                        "threshold": 1.0,
                        "composite": _series("Windowed Subgroup", [5.0, 5.0, 5.0, 8.0]),
                        "pv_specs": [],
                        "pv_data": [
                            {
                                **_series("PV:FLAT", [9.0, 9.0, 9.0, 9.0]),
                                "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                            },
                            {
                                **_series("PV:WINDOWED", [1.0, 1.0, 1.0, 4.0]),
                                "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                            },
                        ],
                    },
                    "Varying Subgroup": {
                        "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                        "threshold": 1.0,
                        "composite": _series("Varying Subgroup", [0.0, 0.5, 1.0, 1.5]),
                        "pv_specs": [],
                        "pv_data": [
                            {
                                **_series("PV:VARY", [0.0, 1.0, 2.0, 3.0]),
                                "beam_paths": ("CU_SXR", "CU_HXR", "SC_SXR", "SC_HXR"),
                            }
                        ],
                    },
                },
            },
        }
    }


class HierarchySparklineViewerUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._viewer_CheckButtons = viewer_module.CheckButtons
        cls._viewer_add_tuning_overlay = viewer_module.add_tuning_overlay
        cls._viewer_first_tick = viewer_module.first_tick_of_day_with_date
        cls._plotting_first_tick = plotting_module.first_tick_of_day_with_date

        viewer_module.CheckButtons = _StubCheckButtons
        viewer_module.add_tuning_overlay = _noop_add_tuning_overlay
        viewer_module.first_tick_of_day_with_date = _dummy_time_formatter
        plotting_module.first_tick_of_day_with_date = _dummy_time_formatter

        cls.viewer_class = viewer_module.HierarchySparklineViewer

    @classmethod
    def tearDownClass(cls):
        viewer_module.CheckButtons = cls._viewer_CheckButtons
        viewer_module.add_tuning_overlay = cls._viewer_add_tuning_overlay
        viewer_module.first_tick_of_day_with_date = cls._viewer_first_tick
        plotting_module.first_tick_of_day_with_date = cls._plotting_first_tick

    def setUp(self):
        plt.close("all")
        self.viewer = self.viewer_class(_synthetic_hierarchy(), START, END, draw_report_path=None)
        self.viewer.draw()
        self.viewer.fig.canvas.toolbar = _FakeToolbar()
        self.viewer._install_toolbar_home_hook()
        self.viewer.fig.canvas.draw()

    def tearDown(self):
        plt.close("all")

    def _plot_axes(self):
        return self.viewer._plot_axes()

    def _set_plot_limits(self, xlim, ylim):
        for axis in self._plot_axes():
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
        self.viewer._capture_view_limits(include_ylim=True)
        self.viewer.fig.canvas.draw()

    def _assert_plot_xlimits(self, xlim):
        expected_xlim = tuple(mdates.date2num(value) for value in xlim)
        self._assert_plot_xlimits_raw(expected_xlim)

    def _assert_viewer_current_xlim(self, xlim):
        expected_xlim = tuple(mdates.date2num(value) for value in xlim)
        actual_xlim = self.viewer.current_xlim
        self.assertAlmostEqual(actual_xlim[0], expected_xlim[0], places=6)
        self.assertAlmostEqual(actual_xlim[1], expected_xlim[1], places=6)

    def _assert_plot_xlimits_raw(self, expected_xlim):
        for axis in self._plot_axes():
            actual_xlim = axis.get_xlim()
            self.assertAlmostEqual(actual_xlim[0], expected_xlim[0], places=6)
            self.assertAlmostEqual(actual_xlim[1], expected_xlim[1], places=6)

    def _assert_first_axis_ylim_raw(self, expected_ylim):
        actual_ylim = self._plot_axes()[0].get_ylim()
        self.assertAlmostEqual(actual_ylim[0], expected_ylim[0], places=6)
        self.assertAlmostEqual(actual_ylim[1], expected_ylim[1], places=6)

    def _assert_plot_ylims_raw(self, expected_ylims):
        actual_axes = self._plot_axes()
        self.assertEqual(len(actual_axes), len(expected_ylims))
        for axis, expected_ylim in zip(actual_axes, expected_ylims):
            actual_ylim = axis.get_ylim()
            self.assertAlmostEqual(actual_ylim[0], expected_ylim[0], places=6)
            self.assertAlmostEqual(actual_ylim[1], expected_ylim[1], places=6)

    def _first_axis_limits(self):
        axis = self._plot_axes()[0]
        return tuple(axis.get_xlim()), tuple(axis.get_ylim())

    def _plot_axis_heights(self):
        return {
            item["label"]: axis.get_position().height
            for axis, item in zip(self._plot_axes(), self.viewer._items_for_path())
        }

    def _item_labels(self):
        return [item["label"] for item in self.viewer._items_for_path()]

    def _figure_texts(self):
        return [text.get_text() for text in self.viewer.fig.texts]

    def _default_first_axis_limits_for_path(self, path):
        reference_viewer = self.viewer_class(_synthetic_hierarchy(), START, END)
        try:
            reference_viewer.path = list(path)
            reference_viewer.draw()
            axis = reference_viewer.fig.axes[: len(reference_viewer._items_for_path())][0]
            return tuple(axis.get_xlim()), tuple(axis.get_ylim())
        finally:
            plt.close(reference_viewer.fig)

    def _click_plot_axis(self, index: int = 0):
        axis = self._plot_axes()[index]
        self.viewer._on_click(SimpleNamespace(button=1, inaxes=axis))
        self.viewer.fig.canvas.draw()

    def _click_breadcrumb(self, target_path):
        artist = next(
            artist
            for artist, target in self.viewer.breadcrumb_targets.items()
            if target == target_path
        )
        self.viewer._on_pick(SimpleNamespace(artist=artist))
        self.viewer.fig.canvas.draw()

    def test_plot_limits_are_preserved_across_checkbox_and_navigation(self):
        xlim = (
            START + dt.timedelta(minutes=30),
            START + dt.timedelta(hours=2, minutes=30),
        )
        ylim = (-2.5, 2.5)

        self._set_plot_limits(xlim, ylim)
        self._assert_viewer_current_xlim(xlim)

        self.viewer.show_points_checkbox.set_active(0)
        self.viewer.fig.canvas.draw()
        self._assert_plot_xlimits(xlim)

        self._click_plot_axis(0)
        self._assert_plot_xlimits(xlim)
        self._assert_viewer_current_xlim(xlim)

        self._click_plot_axis(0)
        self._assert_plot_xlimits(xlim)
        self._assert_viewer_current_xlim(xlim)

        self._click_breadcrumb(["Group A"])
        self._assert_plot_xlimits(xlim)
        self._assert_viewer_current_xlim(xlim)

        self._click_breadcrumb([])
        self._assert_plot_xlimits(xlim)
        self._assert_viewer_current_xlim(xlim)

    def test_control_row_is_centered_with_hidden_checkbox_frames(self):
        row_axes = [
            self.viewer.show_points_checkbox.ax,
            self.viewer.beamline_filter_checkbox.ax,
            self.viewer.accelerator_filter_checkbox.ax,
        ]

        for axis in row_axes:
            self.assertFalse(axis.get_frame_on())

        row_left = min(axis.get_position().x0 for axis in row_axes)
        row_right = max(axis.get_position().x1 for axis in row_axes)
        self.assertAlmostEqual((row_left + row_right) / 2, 0.5, places=3)

        row_ys = [axis.get_position().y0 for axis in row_axes]
        for value in row_ys[1:]:
            self.assertAlmostEqual(value, row_ys[0], places=3)

    def test_beam_path_filter_checkboxes_default_to_all_checked(self):
        self.assertEqual(self.viewer.beamline_filter_checkbox.actives, [True, True])
        self.assertEqual(self.viewer.accelerator_filter_checkbox.actives, [True, True])
        self.assertTrue(self.viewer.filter_sxr)
        self.assertTrue(self.viewer.filter_hxr)
        self.assertTrue(self.viewer.filter_cu)
        self.assertTrue(self.viewer.filter_sc)

    def test_beam_path_filters_limit_visible_groups_subgroups_and_pvs(self):
        self.assertEqual(self._item_labels(), ["Group A", "Group B"])

        self.viewer.accelerator_filter_checkbox.set_active(1)
        self.viewer.fig.canvas.draw()
        self.assertEqual(self._item_labels(), ["Group A"])

        self.viewer.path = ["Group A"]
        self.viewer.draw()
        self.viewer.fig.canvas.draw()
        self.assertEqual(self._item_labels(), ["Subgroup A1", "Subgroup A2"])

        self.viewer.beamline_filter_checkbox.set_active(0)
        self.viewer.fig.canvas.draw()
        self.assertEqual(self._item_labels(), ["Subgroup A2"])

        self.viewer.accelerator_filter_checkbox.set_active(0)
        self.viewer.accelerator_filter_checkbox.set_active(1)
        self.viewer.path = ["Group B", "Subgroup B1"]
        self.viewer.draw()
        self.viewer.fig.canvas.draw()
        self.assertEqual(self._item_labels(), ["PV:B:2"])

    def test_beam_path_filters_show_empty_state_without_changing_path_or_xlim(self):
        self.viewer.path = ["Group B", "Subgroup B1", "PV:B:1"]
        self.viewer.draw()
        self.viewer.fig.canvas.draw()

        xlim = (
            START + dt.timedelta(minutes=45),
            START + dt.timedelta(hours=1, minutes=45),
        )
        ylim = (-1.0, 3.0)
        self._set_plot_limits(xlim, ylim)
        self._assert_viewer_current_xlim(xlim)

        self.viewer.beamline_filter_checkbox.set_active(0)
        self.viewer.fig.canvas.draw()

        self.assertEqual(self.viewer.path, ["Group B", "Subgroup B1", "PV:B:1"])
        self.assertEqual(self._item_labels(), [])
        self.assertEqual(self._plot_axes(), [])
        self.assertIn(
            "No PVs or groups match your selected filters.",
            self._figure_texts(),
        )
        self._assert_viewer_current_xlim(xlim)

        self.viewer.beamline_filter_checkbox.set_active(0)
        self.viewer.fig.canvas.draw()

        self.assertEqual(self.viewer.path, ["Group B", "Subgroup B1", "PV:B:1"])
        self.assertEqual(self._item_labels(), ["PV:B:1"])
        self._assert_plot_xlimits(xlim)
        self._assert_first_axis_ylim_raw(ylim)

    def test_unchecking_both_boxes_in_a_filter_pair_shows_empty_state(self):
        self.viewer.beamline_filter_checkbox.set_active(0)
        self.viewer.beamline_filter_checkbox.set_active(1)
        self.viewer.fig.canvas.draw()

        self.assertEqual(self.viewer.path, [])
        self.assertEqual(self._item_labels(), [])
        self.assertEqual(self._plot_axes(), [])
        self.assertIn(
            "No PVs or groups match your selected filters.",
            self._figure_texts(),
        )

    def test_filter_toggle_preserves_per_axis_ylims_when_visible_items_do_not_change(self):
        self.viewer.path = ["Group A", "Subgroup A1"]
        self.viewer.draw()
        self.viewer.fig.canvas.draw()

        expected_ylims = [(-2.0, 2.0), (10.0, 12.0)]
        for axis, ylim in zip(self._plot_axes(), expected_ylims):
            axis.set_ylim(*ylim)
        self.viewer.fig.canvas.draw()

        self.viewer.accelerator_filter_checkbox.set_active(1)
        self.viewer.fig.canvas.draw()

        self.assertEqual(self._item_labels(), ["PV:A:1", "PV:A:2"])
        self._assert_plot_ylims_raw(expected_ylims)

    def test_toolbar_home_resets_to_original_limits_after_redraw(self):
        self._click_plot_axis(0)
        original_xlim, original_ylim = self._first_axis_limits()

        xlim = (
            START + dt.timedelta(minutes=30),
            START + dt.timedelta(hours=2, minutes=0),
        )
        ylim = (-2.0, 2.0)

        self._set_plot_limits(xlim, ylim)
        self.viewer.show_points_checkbox.set_active(0)
        self.viewer.fig.canvas.draw()
        self._assert_plot_xlimits(xlim)
        self._assert_first_axis_ylim_raw(original_ylim)

        self.viewer.fig.canvas.toolbar.home()
        self.viewer.fig.canvas.draw()
        self.assertEqual(self.viewer.path, ["Group A"])
        self.assertEqual(self.viewer.fig.canvas.toolbar.home_calls, 1)
        self._assert_plot_xlimits_raw(original_xlim)
        self._assert_first_axis_ylim_raw(original_ylim)
        self._assert_viewer_current_xlim((START, END))

    def test_viewer_home_returns_to_root_without_resetting_time_window(self):
        self._click_plot_axis(0)

        xlim = (
            START + dt.timedelta(minutes=45),
            START + dt.timedelta(hours=1, minutes=45),
        )
        ylim = (-1.0, 3.0)

        self._set_plot_limits(xlim, ylim)
        self._assert_viewer_current_xlim(xlim)

        self.viewer.home()
        self.viewer.fig.canvas.draw()

        self.assertEqual(self.viewer.path, [])
        self._assert_plot_xlimits(xlim)
        self._assert_viewer_current_xlim(xlim)

    def test_constant_series_axes_shrink_relative_to_varying_series_across_levels(self):
        plt.close("all")
        self.viewer = self.viewer_class(
            _synthetic_hierarchy_with_constants(),
            START,
            END,
            draw_report_path=None,
        )
        self.viewer.draw()
        self.viewer.fig.canvas.draw()

        heights = self._plot_axis_heights()
        self.assertAlmostEqual(
            heights["Varying Group"] / heights["Constant Group"],
            3.0,
            places=1,
        )

        self.viewer.path = ["Varying Group"]
        self.viewer.draw()
        self.viewer.fig.canvas.draw()

        heights = self._plot_axis_heights()
        self.assertAlmostEqual(
            heights["Varying Subgroup"] / heights["Windowed Subgroup"],
            1.0,
            places=1,
        )

        zoom_xlim = (
            START,
            START + dt.timedelta(hours=1),
        )
        for axis in self._plot_axes():
            axis.set_xlim(*zoom_xlim)
        self.viewer.fig.canvas.draw()
        self.viewer.draw()
        self.viewer.fig.canvas.draw()

        heights = self._plot_axis_heights()
        self.assertAlmostEqual(
            heights["Varying Subgroup"] / heights["Windowed Subgroup"],
            3.0,
            places=1,
        )

        self.viewer.path = ["Varying Group", "Windowed Subgroup"]
        self.viewer.draw()
        self.viewer.fig.canvas.draw()

        heights = self._plot_axis_heights()
        # The viewer preserves the current zoom window across levels. In the
        # one-hour window above, PV:WINDOWED only shows its flat samples, so it
        # is treated the same as PV:FLAT for height allocation.
        self.assertAlmostEqual(
            heights["PV:WINDOWED"] / heights["PV:FLAT"],
            1.0,
            places=1,
        )

    def test_monitor_plots_are_prepended_and_follow_beamline_filters(self):
        plt.close("all")
        viewer = self.viewer_class(
            _synthetic_hierarchy_with_monitors(),
            START,
            END,
            draw_report_path=None,
        )
        try:
            viewer.draw()
            viewer.fig.canvas.draw()

            labels = [item["label"] for item in viewer._items_for_path()]
            self.assertEqual(labels[:6], ["GMD", "XGMD", "GDET 241", "GDET 361", "Group A", "Group B"])

            viewer.beamline_filter_checkbox.set_active(1)
            viewer.fig.canvas.draw()
            labels = [item["label"] for item in viewer._items_for_path()]
            self.assertEqual(labels[:4], ["GMD", "XGMD", "Group A", "Group B"])
            self.assertNotIn("GDET 241", labels)
            self.assertNotIn("GDET 361", labels)

            viewer.path = ["Group A", "Subgroup A1"]
            viewer.draw()
            viewer.fig.canvas.draw()
            labels = [item["label"] for item in viewer._items_for_path()]
            self.assertEqual(labels[:4], ["GMD", "XGMD", "PV:A:1", "PV:A:2"])
        finally:
            plt.close(viewer.fig)

    def test_monitor_plot_click_opens_single_pv_and_breadcrumb_returns_to_origin_view(self):
        plt.close("all")
        viewer = self.viewer_class(
            _synthetic_hierarchy_with_monitors(),
            START,
            END,
            draw_report_path=None,
        )
        try:
            viewer.path = ["Group A"]
            viewer.draw()
            viewer.fig.canvas.draw()

            self.assertEqual([item["label"] for item in viewer._items_for_path()][:4], ["GMD", "XGMD", "GDET 241", "GDET 361"])

            monitor_axis = viewer._plot_axes()[0]
            viewer._on_click(SimpleNamespace(button=1, inaxes=monitor_axis))
            viewer.fig.canvas.draw()

            self.assertEqual(viewer.focused_monitor_label, "GMD")
            self.assertEqual([item["label"] for item in viewer._items_for_path()], ["GMD"])

            axis = viewer._plot_axes()[0]
            self.assertEqual(axis.get_ylabel(), "GMD")
            self.assertIsNone(axis.get_legend())
            self.assertEqual(len(axis.lines), 2)
            self.assertEqual(len(axis.collections), 1)

            target_artist = next(
                artist
                for artist, target in viewer.breadcrumb_targets.items()
                if target == ["Group A"]
            )
            viewer._on_pick(SimpleNamespace(artist=target_artist))
            viewer.fig.canvas.draw()

            self.assertIsNone(viewer.focused_monitor_label)
            self.assertEqual(viewer.path, ["Group A"])
            self.assertEqual([item["label"] for item in viewer._items_for_path()][:4], ["GMD", "XGMD", "GDET 241", "GDET 361"])
        finally:
            plt.close(viewer.fig)

    def test_constant_monitors_collapse_and_hide_numeric_y_axis(self):
        plt.close("all")
        hierarchy = _synthetic_hierarchy_with_monitors()
        hierarchy["monitor_pvs"]["GDET 241"]["values"] = np.asarray([0.0015, 0.0015, 0.0015, 0.0015], dtype=float)
        viewer = self.viewer_class(hierarchy, START, END, draw_report_path=None)
        try:
            viewer.draw()
            viewer.fig.canvas.draw()

            heights = {
                item["label"]: axis.get_position().height
                for axis, item in zip(viewer._plot_axes(), viewer._items_for_path())
            }
            self.assertGreater(heights["GDET 361"], heights["GDET 241"])

            axis_by_label = {
                item["label"]: axis
                for axis, item in zip(viewer._plot_axes(), viewer._items_for_path())
            }
            constant_axis = axis_by_label["GDET 241"]
            varying_axis = axis_by_label["GDET 361"]
            self.assertEqual(len(constant_axis.get_yticks()), 0)
            self.assertGreater(len(varying_axis.get_yticks()), 0)
        finally:
            plt.close(viewer.fig)

    def test_monitor_render_points_are_capped_and_reported(self):
        plt.close("all")
        hierarchy = _synthetic_hierarchy_with_monitors()
        hierarchy["monitor_pvs"]["GMD"] = {
            **_large_series("EM1K0:GMD:HPS:milliJoulesPerPulse", 2500),
            "label": "GMD",
            "beam_paths": ("CU_SXR", "SC_SXR"),
            "value_scale": 1000.0,
            "subsample": True,
        }
        viewer = self.viewer_class(hierarchy, START, END, draw_report_path=None)
        captured = {}
        original_plotter = viewer_module.plot_percentile_band

        def _capture_plot_percentile_band(data, *args, **kwargs):
            if data.get("name") == "EM1K0:GMD:HPS:milliJoulesPerPulse":
                captured["source_points"] = len(data["values"])
                captured["render_max_points"] = kwargs.get("render_max_points")
            return original_plotter(data, *args, **kwargs)

        viewer_module.plot_percentile_band = _capture_plot_percentile_band
        try:
            viewer.draw()
            viewer.fig.canvas.draw()
            self.assertEqual(
                captured["source_points"],
                2500,
            )
            self.assertEqual(
                captured["render_max_points"],
                viewer.MAX_MONITOR_RENDER_POINTS,
            )
            report = next(
                item for item in viewer.last_draw_report["items"] if item["label"] == "GMD"
            )
            self.assertEqual(report["original_points"], 2500)
            self.assertEqual(
                report["rendered_points"],
                viewer.MAX_MONITOR_RENDER_POINTS,
            )
            self.assertGreaterEqual(report["draw_seconds"], 0.0)
            self.assertGreaterEqual(viewer.last_draw_report["total_seconds"], 0.0)
        finally:
            viewer_module.plot_percentile_band = original_plotter
            plt.close(viewer.fig)

    def test_monitor_subsample_false_disables_render_cap(self):
        plt.close("all")
        hierarchy = _synthetic_hierarchy_with_monitors()
        hierarchy["monitor_pvs"]["GMD"] = {
            **_large_series("EM1K0:GMD:HPS:milliJoulesPerPulse", 2500),
            "label": "GMD",
            "beam_paths": ("CU_SXR", "SC_SXR"),
            "value_scale": 1000.0,
            "subsample": False,
        }
        viewer = self.viewer_class(hierarchy, START, END, draw_report_path=None)
        captured = {}
        original_plotter = viewer_module.plot_percentile_band

        def _capture_plot_percentile_band(data, *args, **kwargs):
            if data.get("name") == "EM1K0:GMD:HPS:milliJoulesPerPulse":
                captured["render_max_points"] = kwargs.get("render_max_points")
            return original_plotter(data, *args, **kwargs)

        viewer_module.plot_percentile_band = _capture_plot_percentile_band
        try:
            viewer.draw()
            viewer.fig.canvas.draw()
            self.assertIsNone(captured["render_max_points"])
            report = next(
                item for item in viewer.last_draw_report["items"] if item["label"] == "GMD"
            )
            self.assertEqual(report["original_points"], 2500)
            self.assertEqual(report["rendered_points"], 2500)
        finally:
            viewer_module.plot_percentile_band = original_plotter
            plt.close(viewer.fig)

    def test_monitor_quantile_series_are_cached_across_redraws(self):
        plt.close("all")
        hierarchy = _synthetic_hierarchy_with_monitors()
        hierarchy["monitor_pvs"]["GMD"] = {
            **_large_series("EM1K0:GMD:HPS:milliJoulesPerPulse", 2500),
            "label": "GMD",
            "beam_paths": ("CU_SXR", "SC_SXR"),
            "value_scale": 1000.0,
            "subsample": True,
        }
        viewer = self.viewer_class(hierarchy, START, END, draw_report_path=None)
        counts = {"GMD": 0}
        original_compute = viewer_module.compute_percentile_band_series

        def _capture_compute_percentile_band_series(data, *args, **kwargs):
            if data.get("name") == "EM1K0:GMD:HPS:milliJoulesPerPulse":
                counts["GMD"] += 1
            return original_compute(data, *args, **kwargs)

        viewer_module.compute_percentile_band_series = _capture_compute_percentile_band_series
        try:
            viewer.draw()
            viewer.fig.canvas.draw()
            viewer.draw()
            viewer.fig.canvas.draw()
            self.assertEqual(counts["GMD"], 1)
        finally:
            viewer_module.compute_percentile_band_series = original_compute
            plt.close(viewer.fig)

    def test_plot_percentile_band_subsamples_rendered_quantiles_not_source_series(self):
        plt.close("all")
        data = _large_series("PV:LARGE", 2500)
        full_fig, full_ax = plt.subplots()
        limited_fig, limited_ax = plt.subplots()
        try:
            _, full_band = plotting_module.plot_percentile_band(
                data,
                window_size=30,
                ax=full_ax,
                show_legend=False,
                render_max_points=None,
            )
            _, limited_band = plotting_module.plot_percentile_band(
                data,
                window_size=30,
                ax=limited_ax,
                show_legend=False,
                render_max_points=1000,
            )

            self.assertEqual(limited_band["source_points"], 2500)
            self.assertEqual(limited_band["rendered_points"], 1000)

            full_lookup = {
                when: idx for idx, when in enumerate(full_band["time"])
            }
            for idx, when in enumerate(limited_band["time"]):
                full_idx = full_lookup[when]
                self.assertEqual(limited_band["lower"][idx], full_band["lower"][full_idx])
                self.assertEqual(limited_band["median"][idx], full_band["median"][full_idx])
                self.assertEqual(limited_band["center"][idx], full_band["center"][full_idx])
                self.assertEqual(limited_band["upper"][idx], full_band["upper"][full_idx])
        finally:
            plt.close(full_fig)
            plt.close(limited_fig)

    def test_plot_percentile_band_accepts_precomputed_band(self):
        plt.close("all")
        data = _large_series("PV:LARGE", 2500)
        precomputed = plotting_module.compute_percentile_band_series(
            data,
            window_size=30,
        )
        fig, ax = plt.subplots()
        try:
            _, rendered = plotting_module.plot_percentile_band(
                data,
                window_size=30,
                ax=ax,
                show_legend=False,
                render_max_points=1000,
                precomputed_band=precomputed,
            )

            direct = plotting_module.render_percentile_band_series(
                precomputed,
                render_max_points=1000,
            )
            np.testing.assert_array_equal(rendered["time"], direct["time"])
            np.testing.assert_array_equal(rendered["lower"], direct["lower"])
            np.testing.assert_array_equal(rendered["median"], direct["median"])
            np.testing.assert_array_equal(rendered["center"], direct["center"])
            np.testing.assert_array_equal(rendered["upper"], direct["upper"])
            self.assertEqual(rendered["source_points"], direct["source_points"])
            self.assertEqual(rendered["rendered_points"], direct["rendered_points"])
        finally:
            plt.close(fig)

    def test_plot_percentile_band_subsamples_first_point_in_time_bins(self):
        plt.close("all")
        timestamps = np.array(
            [
                START.timestamp() + offset
                for offset in (0, 10, 60, 110, 160)
            ],
            dtype=float,
        )
        seconds = np.floor(timestamps).astype(np.int64)
        nanoseconds = np.rint((timestamps - seconds) * 1e9).astype(np.int64)
        data = {
            "name": "PV:IRREGULAR",
            "secondsPastEpoch": seconds,
            "nanoseconds": nanoseconds,
            "values": np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float),
            "severity": np.zeros(5, dtype=np.int64),
            "status": np.zeros(5, dtype=np.int64),
        }
        fig, ax = plt.subplots()
        try:
            _, band = plotting_module.plot_percentile_band(
                data,
                window_size=1.0,
                ax=ax,
                show_legend=False,
                render_max_points=4,
            )

            expected_times = [
                START,
                START + dt.timedelta(seconds=60),
                START + dt.timedelta(seconds=110),
                START + dt.timedelta(seconds=160),
            ]
            np.testing.assert_array_equal(band["time"], np.asarray(expected_times, dtype=object))
            np.testing.assert_array_equal(band["lower"], np.asarray([10.0, 30.0, 40.0, 50.0]))
            np.testing.assert_array_equal(band["median"], np.asarray([10.0, 30.0, 40.0, 50.0]))
            np.testing.assert_array_equal(band["center"], np.asarray([10.0, 30.0, 40.0, 50.0]))
            np.testing.assert_array_equal(band["upper"], np.asarray([10.0, 30.0, 40.0, 50.0]))
            self.assertEqual(band["source_points"], 5)
            self.assertEqual(band["rendered_points"], 4)
        finally:
            plt.close(fig)

    def test_plot_percentile_band_subsampling_does_not_create_fake_gaps(self):
        plt.close("all")
        data = _large_series("PV:DENSE", 2500, step_seconds=1)
        fig, ax = plt.subplots()
        try:
            _, band = plotting_module.plot_percentile_band(
                data,
                window_size=10.0,
                ax=ax,
                show_legend=False,
                render_max_points=100,
            )

            self.assertEqual(band["rendered_points"], 100)
            self.assertEqual(len(band["time"]), 100)
            self.assertFalse(np.isnan(band["lower"]).any())
            self.assertFalse(np.isnan(band["median"]).any())
            self.assertFalse(np.isnan(band["center"]).any())
            self.assertFalse(np.isnan(band["upper"]).any())
        finally:
            plt.close(fig)

    def test_plot_percentile_band_subsampling_preserves_real_gaps(self):
        plt.close("all")
        timestamps = np.array(
            [
                START.timestamp() + offset
                for offset in (0, 1, 2, 3, 60, 61, 62, 63)
            ],
            dtype=float,
        )
        seconds = np.floor(timestamps).astype(np.int64)
        nanoseconds = np.rint((timestamps - seconds) * 1e9).astype(np.int64)
        data = {
            "name": "PV:GAPPED",
            "secondsPastEpoch": seconds,
            "nanoseconds": nanoseconds,
            "values": np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0], dtype=float),
            "severity": np.zeros(8, dtype=np.int64),
            "status": np.zeros(8, dtype=np.int64),
        }
        fig, ax = plt.subplots()
        try:
            _, band = plotting_module.plot_percentile_band(
                data,
                window_size=10.0,
                ax=ax,
                show_legend=False,
                render_max_points=4,
            )

            self.assertEqual(band["rendered_points"], 2)
            self.assertEqual(len(band["time"]), 3)
            self.assertTrue(np.isnan(band["center"]).any())
        finally:
            plt.close(fig)

    def test_draw_writes_timing_report_with_hierarchy_path(self):
        plt.close("all")
        with tempfile.NamedTemporaryFile(
            "r+",
            suffix=".txt",
            encoding="utf-8",
            dir=Path(__file__).resolve().parent,
        ) as handle:
            viewer = self.viewer_class(
                _synthetic_hierarchy_with_monitors(),
                START,
                END,
                draw_report_path=handle.name,
            )
            try:
                viewer.draw()
                viewer.fig.canvas.draw()
                viewer.path = ["Group A"]
                viewer.draw()
                viewer.fig.canvas.draw()

                handle.seek(0)
                report_text = handle.read()
                self.assertIn("path=All PV Groups", report_text)
                self.assertIn("path=Group A", report_text)
                self.assertIn("monitor | GMD", report_text)
                self.assertIn("composite | Group A", report_text)
                self.assertIn("total=", report_text)
            finally:
                plt.close(viewer.fig)


if __name__ == "__main__":
    unittest.main()
