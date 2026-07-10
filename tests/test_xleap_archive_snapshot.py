import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app.xleap_archive_snapshot as snapshot_module
from app.xleap_snapshot_store import connect, upsert_snapshot_rows


NOMINAL_TIME = dt.datetime(2026, 6, 1, 12, 0, 0)
KACT_PVS = ["USEG:UNDS:2150:KAct", "USEG:UNDS:2250:KAct"]


def _archive_item(pv: str, values: list[float]) -> dict:
    return {
        "pvName": pv,
        "value": {
            "value": {
                "secondsPastEpoch": [
                    int(NOMINAL_TIME.timestamp()) + idx
                    for idx in range(len(values))
                ],
                "values": values,
            }
        },
    }


class XleapArchiveSnapshotTests(unittest.TestCase):
    def test_default_snapshot_delta_is_five_seconds(self):
        self.assertEqual(
            snapshot_module.DEFAULT_SNAPSHOT_DELTA,
            dt.timedelta(seconds=5),
        )

    def test_relative_k_change_uses_either_k_as_denominator(self):
        self.assertFalse(
            snapshot_module._relative_k_change_exceeds_threshold(
                [4.0, 4.003],
                threshold=0.001,
            )
        )
        self.assertTrue(
            snapshot_module._relative_k_change_exceeds_threshold(
                [4.0, 4.005],
                threshold=0.001,
            )
        )

    def test_fetch_filters_kact_rows_when_any_undulator_is_moving(self):
        payload = [
            _archive_item(snapshot_module.GAMMA_PV, [8000.0]),
            _archive_item(KACT_PVS[0], [4.0, 4.005]),
            _archive_item(KACT_PVS[1], [3.0]),
        ]
        captured = {}

        def _fake_get(pv_names, *, from_time, to_time, timeout):
            captured["pv_names"] = pv_names
            captured["from_time"] = from_time
            captured["to_time"] = to_time
            captured["timeout"] = timeout
            return payload

        with mock.patch.object(snapshot_module.meme.archive, "get", side_effect=_fake_get):
            rows = snapshot_module.fetch_archive_snapshot(
                NOMINAL_TIME,
                kact_pvs=KACT_PVS,
            )

        self.assertEqual(captured["pv_names"], [snapshot_module.GAMMA_PV, *KACT_PVS])
        self.assertEqual(captured["from_time"], NOMINAL_TIME)
        self.assertEqual(
            captured["to_time"],
            NOMINAL_TIME + dt.timedelta(seconds=5),
        )
        self.assertEqual(
            rows,
            [
                (
                    NOMINAL_TIME.isoformat(),
                    snapshot_module.GAMMA_PV,
                    NOMINAL_TIME.isoformat(),
                    8000.0,
                )
            ],
        )

    def test_add_snapshot_clears_stale_kact_values_for_moving_time(self):
        payload = [
            _archive_item(snapshot_module.GAMMA_PV, [8100.0]),
            _archive_item(KACT_PVS[0], [4.0, 4.005]),
            _archive_item(KACT_PVS[1], [3.0]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "snapshots.sqlite"
            stale_rows = [
                (
                    NOMINAL_TIME.isoformat(),
                    snapshot_module.GAMMA_PV,
                    NOMINAL_TIME.isoformat(),
                    8000.0,
                ),
                (
                    NOMINAL_TIME.isoformat(),
                    KACT_PVS[0],
                    NOMINAL_TIME.isoformat(),
                    4.0,
                ),
                (
                    NOMINAL_TIME.isoformat(),
                    KACT_PVS[1],
                    NOMINAL_TIME.isoformat(),
                    3.0,
                ),
            ]
            with connect(db_path) as connection:
                upsert_snapshot_rows(connection, stale_rows, update_wide=True)

            with mock.patch.object(snapshot_module.meme.archive, "get", return_value=payload):
                rows = snapshot_module.add_archive_snapshot(
                    NOMINAL_TIME,
                    db_path=db_path,
                    kact_pvs=KACT_PVS,
                )

            with connect(db_path) as connection:
                long_rows = connection.execute(
                    """
                    SELECT pv, value FROM snapshot_values
                    WHERE nominal_time = ?
                    ORDER BY pv
                    """,
                    (NOMINAL_TIME.isoformat(),),
                ).fetchall()
                wide_row = connection.execute(
                    """
                    SELECT "BEND:DMPS:400:BACT", "USEG:UNDS:2150:KAct", "USEG:UNDS:2250:KAct"
                    FROM snapshot_values_wide
                    WHERE nominal_time = ?
                    """,
                    (NOMINAL_TIME.isoformat(),),
                ).fetchone()
                moving_rows = connection.execute(
                    """
                    SELECT pv FROM moving_undulator_snapshots
                    WHERE nominal_time = ?
                    ORDER BY pv
                    """,
                    (NOMINAL_TIME.isoformat(),),
                ).fetchall()

            incomplete_times = snapshot_module.incomplete_snapshot_times(db_path=db_path)

        self.assertEqual(
            rows,
            [
                (
                    NOMINAL_TIME.isoformat(),
                    snapshot_module.GAMMA_PV,
                    NOMINAL_TIME.isoformat(),
                    8100.0,
                )
            ],
        )
        self.assertEqual(long_rows, [(snapshot_module.GAMMA_PV, 8100.0)])
        self.assertEqual(wide_row, (8100.0, None, None))
        self.assertEqual(moving_rows, [(KACT_PVS[0],)])
        self.assertEqual(incomplete_times, [])


if __name__ == "__main__":
    unittest.main()
