"""Fetch one XLEAP archive snapshot and store it in SQLite."""

from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import meme

from sparklines_v2.archive.store import (
    DEFAULT_DB_PATH,
    SnapshotRow,
    _quote_identifier,
    clear_snapshot_pvs,
    connect,
    replace_moving_undulator_markers,
    upsert_snapshot_rows,
)


GAMMA_PV = "BEND:DMPS:400:BACT"
UNDULATOR_LAMBDA_PATTERN = "USEG:UNDS:%:lambda_U"
UNDULATOR_NUM_RE = re.compile(r"USEG:UNDS:(\d+):lambda_U")
DEFAULT_SNAPSHOT_DELTA = timedelta(seconds=5)
DEFAULT_ARCHIVE_TIMEOUT_SECONDS = 30.0
MOVING_K_RELATIVE_THRESHOLD = 0.001


class ArchiveSnapshotResult(NamedTuple):
    rows: list[SnapshotRow]
    moving_undulators: list[str]


def list_kact_pvs() -> list[str]:
    """Return all undulator KAct PVs, sorted by undulator number."""
    lambda_pvs = meme.names.list_pvs(UNDULATOR_LAMBDA_PATTERN)
    undulator_numbers = []
    for pv in lambda_pvs:
        match = UNDULATOR_NUM_RE.fullmatch(pv)
        if match is None:
            continue
        undulator_numbers.append(match.group(1))

    return [
        f"USEG:UNDS:{num}:KAct"
        for num in sorted(undulator_numbers, key=int)
    ]


def _archive_item_to_row(nominal_time: dt, item: dict) -> SnapshotRow | None:
    if item is None:
        return None

    value_payload = item.get("value", {}).get("value", {})
    seconds = value_payload.get("secondsPastEpoch", [])
    values = value_payload.get("values", [])
    if len(seconds) == 0 or len(values) == 0:
        return None

    pv = item["pvName"]
    timestamp = dt.fromtimestamp(seconds[-1]).isoformat()
    return (nominal_time.isoformat(), pv, timestamp, float(values[-1]))


def _relative_k_change_exceeds_threshold(
    values,
    threshold: float = MOVING_K_RELATIVE_THRESHOLD,
) -> bool:
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if len(finite_values) < 2:
        return False

    for idx, first_value in enumerate(finite_values[:-1]):
        for second_value in finite_values[idx + 1:]:
            difference = abs(first_value - second_value)
            if difference == 0:
                continue

            denominators = [
                abs(value)
                for value in (first_value, second_value)
                if abs(value) > 0
            ]
            if not denominators:
                return True
            if any(difference / denominator > threshold for denominator in denominators):
                return True

    return False


def _moving_undulator_pvs(
    payload,
    kact_pvs: list[str],
    *,
    threshold: float = MOVING_K_RELATIVE_THRESHOLD,
) -> list[str]:
    kact_pv_set = set(kact_pvs)
    moving_pvs = []

    for item in payload:
        if item is None:
            continue
        pv = item.get("pvName")
        if pv not in kact_pv_set:
            continue

        values = item.get("value", {}).get("value", {}).get("values", [])
        if len(values) > 1 and _relative_k_change_exceeds_threshold(values, threshold):
            moving_pvs.append(pv)

    return moving_pvs


def _fetch_archive_snapshot_result(
    nominal_time: dt,
    *,
    snapshot_delta: timedelta = DEFAULT_SNAPSHOT_DELTA,
    kact_pvs: list[str] | None = None,
    archive_timeout_seconds: float = DEFAULT_ARCHIVE_TIMEOUT_SECONDS,
) -> ArchiveSnapshotResult:
    kact_pvs = kact_pvs or list_kact_pvs()
    pv_names = [GAMMA_PV, *kact_pvs]
    payload = meme.archive.get(
        pv_names,
        from_time=nominal_time,
        to_time=nominal_time + snapshot_delta,
        timeout=archive_timeout_seconds,
    )

    moving_undulators = _moving_undulator_pvs(payload, kact_pvs)
    kact_pv_set = set(kact_pvs)
    rows = []
    for item in payload:
        row = _archive_item_to_row(nominal_time, item)
        if row is None:
            continue
        if moving_undulators and row[1] in kact_pv_set:
            continue
        rows.append(row)

    return ArchiveSnapshotResult(rows=rows, moving_undulators=moving_undulators)


def fetch_archive_snapshot(
    nominal_time: dt,
    *,
    snapshot_delta: timedelta = DEFAULT_SNAPSHOT_DELTA,
    kact_pvs: list[str] | None = None,
    archive_timeout_seconds: float = DEFAULT_ARCHIVE_TIMEOUT_SECONDS,
) -> list[SnapshotRow]:
    """Fetch gamma and all undulator KAct values for one nominal time."""
    return _fetch_archive_snapshot_result(
        nominal_time,
        snapshot_delta=snapshot_delta,
        kact_pvs=kact_pvs,
        archive_timeout_seconds=archive_timeout_seconds,
    ).rows


def add_archive_snapshot(
    nominal_time: dt | None = None,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    snapshot_delta: timedelta = DEFAULT_SNAPSHOT_DELTA,
    kact_pvs: list[str] | None = None,
    archive_timeout_seconds: float = DEFAULT_ARCHIVE_TIMEOUT_SECONDS,
) -> list[SnapshotRow]:
    """Fetch one snapshot and add it to the SQLite cache."""
    nominal_time = (nominal_time or dt.now()).replace(microsecond=0)
    kact_pvs = kact_pvs or list_kact_pvs()
    result = _fetch_archive_snapshot_result(
        nominal_time,
        snapshot_delta=snapshot_delta,
        kact_pvs=kact_pvs,
        archive_timeout_seconds=archive_timeout_seconds,
    )
    with connect(db_path) as connection:
        replace_moving_undulator_markers(
            connection,
            nominal_time.isoformat(),
            result.moving_undulators,
        )
        if result.moving_undulators:
            clear_snapshot_pvs(
                connection,
                nominal_time.isoformat(),
                kact_pvs,
                update_wide=True,
            )
        upsert_snapshot_rows(connection, result.rows, update_wide=True)
    return result.rows


def incomplete_snapshot_times(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> list[dt]:
    """Return nominal times whose wide value row has at least one NULL PV."""
    with connect(db_path) as connection:
        pv_columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(snapshot_values_wide)")
            if row[1] != "nominal_time"
        ]
        if not pv_columns:
            return []

        where_clause = " OR ".join(
            f"{_quote_identifier(column)} IS NULL" for column in pv_columns
        )
        limit_clause = "" if limit is None else " LIMIT ?"
        params = () if limit is None else (int(limit),)
        rows = connection.execute(
            f"""
            SELECT nominal_time
            FROM snapshot_values_wide
            WHERE ({where_clause})
              AND nominal_time NOT IN (
                  SELECT nominal_time FROM moving_undulator_snapshots
              )
            ORDER BY nominal_time
            {limit_clause}
            """,
            params,
        ).fetchall()

    return [dt.fromisoformat(row[0]) for row in rows]


def backfill_incomplete_snapshots(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    snapshot_delta: timedelta = DEFAULT_SNAPSHOT_DELTA,
    archive_timeout_seconds: float = DEFAULT_ARCHIVE_TIMEOUT_SECONDS,
    max_workers: int = 8,
    batch_size: int = 27,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Fetch and store all incomplete wide-table snapshot rows."""
    times = incomplete_snapshot_times(db_path=db_path, limit=limit)
    if not times:
        return 0, 0, 0

    kact_pvs = list_kact_pvs()
    fetched_count = 0
    failed_count = 0
    stored_rows: list[SnapshotRow] = []

    def fetch_one(nominal_time: dt) -> ArchiveSnapshotResult:
        return _fetch_archive_snapshot_result(
            nominal_time,
            snapshot_delta=snapshot_delta,
            kact_pvs=kact_pvs,
            archive_timeout_seconds=archive_timeout_seconds,
        )

    with connect(db_path) as connection:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            futures = {executor.submit(fetch_one, time): time for time in times}
            for future in as_completed(futures):
                nominal_time = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_count += 1
                    print(f"failed {nominal_time.isoformat()}: {exc}", flush=True)
                    continue
                fetched_count += 1
                replace_moving_undulator_markers(
                    connection,
                    nominal_time.isoformat(),
                    result.moving_undulators,
                )
                if result.moving_undulators:
                    clear_snapshot_pvs(
                        connection,
                        nominal_time.isoformat(),
                        kact_pvs,
                        update_wide=True,
                    )
                stored_rows.extend(result.rows)
                if len(stored_rows) >= batch_size:
                    upsert_snapshot_rows(connection, stored_rows, update_wide=True)
                    stored_rows.clear()
                    print(f"stored {fetched_count}/{len(times)} snapshots", flush=True)

        if stored_rows:
            upsert_snapshot_rows(connection, stored_rows, update_wide=True)

    return fetched_count, len(times), failed_count


def parse_datetime(value: str) -> dt:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed.replace(microsecond=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch one gamma plus undulator K snapshot into SQLite."
    )
    parser.add_argument(
        "--time",
        type=parse_datetime,
        default=None,
        help="Nominal snapshot time. Defaults to now.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_SNAPSHOT_DELTA.total_seconds(),
        help="Archive snapshot window in seconds.",
    )
    parser.add_argument(
        "--archive-timeout-seconds",
        type=float,
        default=DEFAULT_ARCHIVE_TIMEOUT_SECONDS,
        help="Timeout passed to meme.archive.get.",
    )
    parser.add_argument(
        "--backfill-incomplete",
        action="store_true",
        help="Fetch all nominal times with NULLs in snapshot_values_wide.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Thread count for --backfill-incomplete.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=27,
        help="Number of fetched PV rows to upsert per SQLite batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of incomplete nominal times to backfill.",
    )
    args = parser.parse_args(argv)

    snapshot_delta = timedelta(seconds=args.window_seconds)
    if args.backfill_incomplete:
        fetched_count, total_count, failed_count = backfill_incomplete_snapshots(
            db_path=args.db,
            snapshot_delta=snapshot_delta,
            archive_timeout_seconds=args.archive_timeout_seconds,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(
            f"backfilled {fetched_count}/{total_count} incomplete nominal times "
            f"({failed_count} failed)"
        )
        print(f"database: {args.db}")
        return 0

    nominal_time = (args.time or dt.now()).replace(microsecond=0)
    rows = add_archive_snapshot(
        nominal_time,
        db_path=args.db,
        snapshot_delta=snapshot_delta,
        archive_timeout_seconds=args.archive_timeout_seconds,
    )
    print(f"stored {len(rows)} PVs for nominal_time={nominal_time.isoformat()}")
    print(f"database: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
