import datetime as dt
import tempfile
import unittest
from unittest import mock

import numpy as np

import app.sparklines_hierarchy as hierarchy_module


def _archive_series(name: str, values: list[float]) -> dict:
    timestamps = np.array([1_700_000_000 + idx for idx in range(len(values))], dtype=np.int64)
    return {
        "name": name,
        "secondsPastEpoch": timestamps,
        "nanoseconds": np.zeros(len(values), dtype=np.int64),
        "values": np.asarray(values, dtype=float),
        "severity": np.zeros(len(values), dtype=np.int64),
        "status": np.zeros(len(values), dtype=np.int64),
    }


class BuildCompositeHierarchyMonitorTests(unittest.TestCase):
    def test_loads_pv_groups_from_yaml(self):
        yaml_text = """
groups:
  - group_name: test_group
    subgroups: {}
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(yaml_text)
            handle.flush()
            pv_groups = hierarchy_module.load_pv_groups(handle.name)

        self.assertEqual(
            pv_groups,
            {
                "groups": [
                    {
                        "group_name": "test_group",
                        "subgroups": {},
                    }
                ]
            },
        )

    def test_loads_default_monitor_specs_from_yaml(self):
        yaml_text = """
monitor_pvs:
  Test Monitor:
    pv_name: PV:MONITOR
    beam_paths:
      - CU_SXR
      - SC_SXR
    value_scale: 1000.0
    subsample: false
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(yaml_text)
            handle.flush()
            monitor_specs = hierarchy_module._load_default_monitor_specs(handle.name)

        self.assertEqual(
            monitor_specs,
            {
                "Test Monitor": {
                    "pv_name": "PV:MONITOR",
                    "beam_paths": ["CU_SXR", "SC_SXR"],
                    "value_scale": 1000.0,
                    "subsample": False,
                }
            },
        )

    def test_uses_explicit_monitor_specs(self):
        pv_groups = {"groups": []}
        monitor_specs = {
            "Test Monitor": {
                "pv_name": "PV:MONITOR",
                "Beam_Path": "CU_SXR, SC_SXR",
                "value_scale": "1000.0",
                "subsample": False,
            }
        }

        start = dt.datetime(2026, 1, 1, 0, 0, 0)
        end = dt.datetime(2026, 1, 1, 1, 0, 0)
        captured_pvs = {}

        def _fake_fetch(pv_names, *_args, **_kwargs):
            captured_pvs["pv_names"] = tuple(pv_names)
            return {"PV:MONITOR": _archive_series("PV:MONITOR", [1.0, 2.0])}, {}, {}

        with mock.patch.object(hierarchy_module, "_fetch_archive_batch", side_effect=_fake_fetch):
            hierarchy = hierarchy_module.build_composite_hierarchy(
                pv_groups,
                start,
                end,
                monitor_specs=monitor_specs,
            )

        self.assertEqual(captured_pvs["pv_names"], ("PV:MONITOR",))
        self.assertIn("Test Monitor", hierarchy["monitor_pvs"])

        monitor = hierarchy["monitor_pvs"]["Test Monitor"]
        self.assertEqual(monitor["label"], "Test Monitor")
        self.assertEqual(monitor["name"], "PV:MONITOR")
        self.assertEqual(monitor["beam_paths"], ("CU_SXR", "SC_SXR"))
        self.assertEqual(monitor["value_scale"], 1000.0)
        self.assertFalse(monitor["subsample"])

    def test_uses_default_monitor_specs_when_monitor_specs_not_provided(self):
        pv_groups = {"groups": []}

        start = dt.datetime(2026, 1, 1, 0, 0, 0)
        end = dt.datetime(2026, 1, 1, 1, 0, 0)
        captured_pvs = {}
        default_pv_names = [
            spec["pv_name"] for spec in hierarchy_module.DEFAULT_MONITOR_SPECS.values()
        ]

        def _fake_fetch(pv_names, *_args, **_kwargs):
            captured_pvs["pv_names"] = tuple(pv_names)
            return {
                pv_name: _archive_series(pv_name, [1.0, 2.0])
                for pv_name in default_pv_names
            }, {}, {}

        with mock.patch.object(hierarchy_module, "_fetch_archive_batch", side_effect=_fake_fetch):
            hierarchy = hierarchy_module.build_composite_hierarchy(pv_groups, start, end)

        self.assertEqual(captured_pvs["pv_names"], tuple(sorted(default_pv_names)))
        self.assertEqual(
            tuple(hierarchy["monitor_pvs"].keys()),
            tuple(hierarchy_module.DEFAULT_MONITOR_SPECS.keys()),
        )

    def test_build_default_composite_hierarchy_uses_local_monitor_specs_yaml(self):
        start = dt.datetime(2026, 1, 1, 0, 0, 0)
        end = dt.datetime(2026, 1, 1, 1, 0, 0)
        pv_groups = {"groups": []}
        monitor_specs = {
            "Test Monitor": {
                "pv_name": "PV:MONITOR",
                "Beam_Path": "CU_SXR",
                "value_scale": 1000.0,
                "subsample": False,
            }
        }

        with (
            mock.patch.object(hierarchy_module, "load_pv_groups", return_value=pv_groups) as load_mock,
            mock.patch.object(hierarchy_module, "DEFAULT_MONITOR_SPECS", monitor_specs),
            mock.patch.object(
                hierarchy_module,
                "_fetch_archive_batch",
                return_value=({"PV:MONITOR": _archive_series("PV:MONITOR", [1.0, 2.0])}, {}, {}),
            ),
        ):
            hierarchy = hierarchy_module.build_default_composite_hierarchy(start, end)

        load_mock.assert_called_once_with(None)
        self.assertIn("Test Monitor", hierarchy["monitor_pvs"])

    def test_fetch_archive_batch_caps_worker_count(self):
        pv_names = [f"PV:{idx}" for idx in range(hierarchy_module.ARCHIVE_MAX_WORKERS + 5)]
        captured = {}

        class _DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, _count):
                return None

        class _DummyExecutor:
            def __init__(self, *, max_workers):
                captured["max_workers"] = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, *_args, **_kwargs):
                return object()

        with (
            mock.patch.object(hierarchy_module, "ThreadPoolExecutor", _DummyExecutor),
            mock.patch.object(hierarchy_module, "as_completed", return_value=[]),
            mock.patch.object(hierarchy_module, "tqdm", return_value=_DummyProgress()),
        ):
            hierarchy_module._fetch_archive_batch(
                pv_names,
                dt.datetime(2026, 1, 1, 0, 0, 0),
                dt.datetime(2026, 1, 1, 1, 0, 0),
            )

        self.assertEqual(captured["max_workers"], hierarchy_module.ARCHIVE_MAX_WORKERS)

    def test_compress_measurement_series_keeps_period_endpoints_and_extrema(self):
        data = _archive_series("PV:MEAS", [0.0, 0.01, 0.02, 0.5, 0.4, 0.7, 0.02])

        compressed = hierarchy_module._compress_measurement_series_for_composite(
            data,
            deadband=0.1,
            timeout=300,
        )

        np.testing.assert_array_equal(
            compressed["secondsPastEpoch"],
            data["secondsPastEpoch"][[0, 3, 5, 6]],
        )
        np.testing.assert_array_equal(
            compressed["values"],
            np.asarray([0.0, 0.5, 0.7, 0.02]),
        )

    def test_measurement_pvs_are_compressed_before_mixed_group_composite(self):
        pv_groups = {
            "groups": [
                {
                    "group_name": "mixed",
                    "subgroups": {
                        "sub": {
                            "pv": [
                                {
                                    "pv_name": "PV:MEAS",
                                    "measurement": True,
                                    "measurement_deadband": 0.1,
                                },
                                {"pv_name": "PV:CTRL"},
                            ],
                        },
                    },
                },
            ],
        }

        archive = {
            "PV:MEAS": _archive_series(
                "PV:MEAS",
                [0.0, 0.01, 0.02, 0.5, 0.4, 0.7, 0.02],
            ),
            "PV:CTRL": _archive_series("PV:CTRL", [1.0] * 7),
        }

        with mock.patch.object(
            hierarchy_module,
            "_fetch_archive_batch",
            return_value=(archive, {}, {}),
        ):
            hierarchy = hierarchy_module.build_composite_hierarchy(
                pv_groups,
                dt.datetime(2026, 1, 1, 0, 0, 0),
                dt.datetime(2026, 1, 1, 1, 0, 0),
                monitor_specs={},
            )

        composite = hierarchy["groups"]["mixed"]["subgroups"]["sub"]["composite"]
        np.testing.assert_array_equal(
            composite["secondsPastEpoch"],
            archive["PV:MEAS"]["secondsPastEpoch"][[0, 3, 5, 6]],
        )


if __name__ == "__main__":
    unittest.main()
