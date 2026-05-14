import datetime as dt
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from app.main import SparklineMainWindow, resolve_time_range


START = dt.datetime(2026, 3, 30, 22, 0, 0)
END = dt.datetime(2026, 3, 31, 6, 0, 0)


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


def _synthetic_hierarchy() -> dict:
    monitor = _series("EM1K0:GMD:HPS:milliJoulesPerPulse", [0.2, 1.0, 1.8, 1.5, 0.7], 600)
    monitor["label"] = "GMD"
    monitor["beam_paths"] = ("CU_SXR", "SC_SXR")
    monitor["value_scale"] = 1000.0
    monitor["subsample"] = True

    subgroup_composite = _series("Subgroup A", [0.0, 0.4, 0.9, 0.2], 900)
    group_composite = _series("Group A", [0.0, 0.25, 0.5, 0.3], 900)
    pv_data = [_series("PV:A:1", [0.0, 1.0, 0.5, 0.75], 900)]
    pv_data[0]["beam_paths"] = ("CU_SXR",)

    return {
        "groups": {
            "Group A": {
                "beam_paths": ("CU_SXR",),
                "composite": group_composite,
                "subgroups": {
                    "Subgroup A": {
                        "beam_paths": ("CU_SXR",),
                        "composite": subgroup_composite,
                        "pv_data": pv_data,
                    }
                },
            }
        },
        "pv_cache": {
            monitor["name"]: monitor,
            pv_data[0]["name"]: pv_data[0],
        },
        "skipped_pvs": {},
        "monitor_pvs": {"GMD": monitor},
        "timing": {
            "fetch_wall_seconds": 1.25,
            "build_wall_seconds": 0.50,
        },
    }


class SparklineMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_resolve_time_range_defaults_hours_window(self):
        now = dt.datetime(2026, 4, 10, 8, 0, 0)
        start_time, end_time = resolve_time_range(
            start_text=None,
            end_text=None,
            hours=6.0,
            now=now,
        )
        self.assertEqual(end_time, now)
        self.assertEqual(start_time, now - dt.timedelta(hours=6))

    def test_apply_hierarchy_embeds_viewer_and_can_capture_screenshot(self):
        window = SparklineMainWindow(
            start_time=START,
            end_time=END,
            autoload=False,
        )
        try:
            window._apply_hierarchy(_synthetic_hierarchy(), START, END)
            window.show()
            self.app.processEvents()

            self.assertIsNotNone(window.viewer)
            self.assertIs(window.viewer.fig, window.canvas.figure)
            self.assertIs(window.toolbar.canvas, window.canvas)
            self.assertIn("Loaded 2 PVs", window.summary_label.text())
            self.assertGreater(len(window.canvas.figure.axes), 0)

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
                dir=Path(__file__).resolve().parent,
            ) as handle:
                screenshot_path = Path(handle.name)
            try:
                self.assertTrue(window.grab().save(str(screenshot_path)))
                self.assertGreater(screenshot_path.stat().st_size, 0)
            finally:
                screenshot_path.unlink(missing_ok=True)
        finally:
            window.close()

    def test_background_load_builds_viewer(self):
        calls = []

        def _builder(start_time, end_time, pv_groups_path=None):
            calls.append((start_time, end_time, pv_groups_path))
            return _synthetic_hierarchy()

        window = SparklineMainWindow(
            start_time=START,
            end_time=END,
            autoload=False,
            hierarchy_builder=_builder,
        )
        try:
            window._start_load(START, END)
            deadline = time.time() + 5.0
            while window.viewer is None and time.time() < deadline:
                self.app.processEvents()
                time.sleep(0.01)

            self.assertIsNotNone(window.viewer)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], START)
            self.assertEqual(calls[0][1], END)
            self.assertEqual(window.status_label.text(), "Hierarchy loaded.")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
