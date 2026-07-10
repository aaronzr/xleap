"""SQLite storage for XLEAP archive snapshot caches."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


SNAPSHOT_FIELDS = ["nominal_time", "pv", "timestamp", "value"]
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "xleap_snapshots.sqlite"
SnapshotRow = tuple[str, str, str, float]


SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot_values (
    nominal_time TEXT NOT NULL,
    pv TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (nominal_time, pv)
);

CREATE TABLE IF NOT EXISTS snapshot_timestamps (
    nominal_time TEXT NOT NULL,
    pv TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (nominal_time, pv)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_values_pv_time
    ON snapshot_values (pv, nominal_time);

CREATE INDEX IF NOT EXISTS idx_snapshot_timestamps_pv_time
    ON snapshot_timestamps (pv, nominal_time);

CREATE TABLE IF NOT EXISTS moving_undulator_snapshots (
    nominal_time TEXT NOT NULL,
    pv TEXT NOT NULL,
    PRIMARY KEY (nominal_time, pv)
);
"""


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the snapshot SQLite database and ensure its schema exists."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def import_snapshot_csv(connection: sqlite3.Connection, csv_path: str | Path) -> int:
    """Import one long-form snapshot CSV into value and timestamp tables."""
    csv_path = Path(csv_path)
    snapshot_rows: list[SnapshotRow] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SNAPSHOT_FIELDS:
            raise ValueError(
                f"{csv_path} must have columns {SNAPSHOT_FIELDS}; got {reader.fieldnames}"
            )

        for row in reader:
            nominal_time = row["nominal_time"]
            pv = row["pv"]
            timestamp = row["timestamp"]
            value = float(row["value"])
            snapshot_rows.append((nominal_time, pv, timestamp, value))

    upsert_snapshot_rows(connection, snapshot_rows, update_wide=False)
    return len(snapshot_rows)


def upsert_snapshot_rows(
    connection: sqlite3.Connection,
    snapshot_rows: list[SnapshotRow],
    *,
    update_wide: bool = True,
) -> int:
    """Upsert long-form snapshot rows and optionally mirror them into wide tables."""
    value_rows = [
        (nominal_time, pv, value)
        for nominal_time, pv, _timestamp, value in snapshot_rows
    ]
    timestamp_rows = [
        (nominal_time, pv, timestamp)
        for nominal_time, pv, timestamp, _value in snapshot_rows
    ]

    with connection:
        connection.executemany(
            """
            INSERT INTO snapshot_values (nominal_time, pv, value)
            VALUES (?, ?, ?)
            ON CONFLICT(nominal_time, pv) DO UPDATE SET value = excluded.value
            """,
            value_rows,
        )
        connection.executemany(
            """
            INSERT INTO snapshot_timestamps (nominal_time, pv, timestamp)
            VALUES (?, ?, ?)
            ON CONFLICT(nominal_time, pv) DO UPDATE SET timestamp = excluded.timestamp
            """,
            timestamp_rows,
        )
        if update_wide:
            upsert_wide_snapshot_rows(connection, snapshot_rows)

    return len(snapshot_rows)


def clear_snapshot_pvs(
    connection: sqlite3.Connection,
    nominal_time: str,
    pv_names: list[str],
    *,
    update_wide: bool = True,
) -> None:
    """Clear selected PVs for one nominal time from long and wide snapshot tables."""
    if not pv_names:
        return

    placeholders = ", ".join(["?"] * len(pv_names))
    params = [nominal_time, *pv_names]
    with connection:
        connection.execute(
            f"""
            DELETE FROM snapshot_values
            WHERE nominal_time = ? AND pv IN ({placeholders})
            """,
            params,
        )
        connection.execute(
            f"""
            DELETE FROM snapshot_timestamps
            WHERE nominal_time = ? AND pv IN ({placeholders})
            """,
            params,
        )

        if update_wide and _table_exists(connection, "snapshot_values_wide"):
            value_columns = _table_columns(connection, "snapshot_values_wide")
            timestamp_columns = _table_columns(connection, "snapshot_timestamps_wide")
            for pv in pv_names:
                pv_column = _quote_identifier(pv)
                if pv in value_columns:
                    connection.execute(
                        f"""
                        UPDATE snapshot_values_wide
                        SET {pv_column} = NULL
                        WHERE nominal_time = ?
                        """,
                        (nominal_time,),
                    )
                if pv in timestamp_columns:
                    connection.execute(
                        f"""
                        UPDATE snapshot_timestamps_wide
                        SET {pv_column} = NULL
                        WHERE nominal_time = ?
                        """,
                        (nominal_time,),
                    )


def replace_moving_undulator_markers(
    connection: sqlite3.Connection,
    nominal_time: str,
    moving_pvs: list[str],
) -> None:
    """Replace the moving-undulator marker set for one nominal time."""
    with connection:
        connection.execute(
            """
            DELETE FROM moving_undulator_snapshots
            WHERE nominal_time = ?
            """,
            (nominal_time,),
        )
        connection.executemany(
            """
            INSERT INTO moving_undulator_snapshots (nominal_time, pv)
            VALUES (?, ?)
            """,
            [(nominal_time, pv) for pv in moving_pvs],
        )


def import_snapshot_csvs(
    csv_paths: list[str | Path],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[Path, int]:
    """Import multiple snapshot CSV files into a SQLite cache."""
    imported: dict[Path, int] = {}
    with connect(db_path) as connection:
        for csv_path in csv_paths:
            path = Path(csv_path)
            imported[path] = import_snapshot_csv(connection, path)
    return imported


def _ordered_pv_names(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT pv FROM snapshot_values ORDER BY pv"
        ).fetchall()
    ]


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return (
        connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()
        is not None
    )


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(connection, table_name):
        return set()
    return {row[1] for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table_name)})")}


def _ensure_wide_tables(connection: sqlite3.Connection, pv_names: list[str]) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot_values_wide (
            nominal_time TEXT PRIMARY KEY
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot_timestamps_wide (
            nominal_time TEXT PRIMARY KEY
        )
        """
    )

    value_columns = _table_columns(connection, "snapshot_values_wide")
    timestamp_columns = _table_columns(connection, "snapshot_timestamps_wide")
    for pv in pv_names:
        pv_column = _quote_identifier(pv)
        if pv not in value_columns:
            connection.execute(f"ALTER TABLE snapshot_values_wide ADD COLUMN {pv_column} REAL")
        if pv not in timestamp_columns:
            connection.execute(f"ALTER TABLE snapshot_timestamps_wide ADD COLUMN {pv_column} TEXT")


def upsert_wide_snapshot_rows(
    connection: sqlite3.Connection,
    snapshot_rows: list[SnapshotRow],
) -> None:
    """Upsert snapshot rows into wide value/timestamp tables."""
    if not snapshot_rows:
        return

    pv_names = sorted({pv for _nominal_time, pv, _timestamp, _value in snapshot_rows})
    _ensure_wide_tables(connection, pv_names)

    nominal_times = sorted({nominal_time for nominal_time, _pv, _timestamp, _value in snapshot_rows})
    connection.executemany(
        """
        INSERT INTO snapshot_values_wide (nominal_time)
        VALUES (?)
        ON CONFLICT(nominal_time) DO NOTHING
        """,
        [(nominal_time,) for nominal_time in nominal_times],
    )
    connection.executemany(
        """
        INSERT INTO snapshot_timestamps_wide (nominal_time)
        VALUES (?)
        ON CONFLICT(nominal_time) DO NOTHING
        """,
        [(nominal_time,) for nominal_time in nominal_times],
    )

    for nominal_time, pv, timestamp, value in snapshot_rows:
        pv_column = _quote_identifier(pv)
        connection.execute(
            f"""
            UPDATE snapshot_values_wide
            SET {pv_column} = ?
            WHERE nominal_time = ?
            """,
            (value, nominal_time),
        )
        connection.execute(
            f"""
            UPDATE snapshot_timestamps_wide
            SET {pv_column} = ?
            WHERE nominal_time = ?
            """,
            (timestamp, nominal_time),
        )


def rebuild_wide_tables(connection: sqlite3.Connection) -> None:
    """Rebuild wide value/timestamp tables with one column per PV name."""
    pv_names = _ordered_pv_names(connection)
    value_columns = ",\n    ".join(f"{_quote_identifier(pv)} REAL" for pv in pv_names)
    timestamp_columns = ",\n    ".join(f"{_quote_identifier(pv)} TEXT" for pv in pv_names)

    with connection:
        connection.execute("DROP TABLE IF EXISTS snapshot_values_wide")
        connection.execute("DROP TABLE IF EXISTS snapshot_timestamps_wide")
        connection.execute(
            f"""
            CREATE TABLE snapshot_values_wide (
                nominal_time TEXT PRIMARY KEY{"," if value_columns else ""}
                {value_columns}
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE snapshot_timestamps_wide (
                nominal_time TEXT PRIMARY KEY{"," if timestamp_columns else ""}
                {timestamp_columns}
            )
            """
        )

        nominal_times = [
            row[0]
            for row in connection.execute(
                """
                SELECT nominal_time FROM snapshot_values
                UNION
                SELECT nominal_time FROM snapshot_timestamps
                ORDER BY nominal_time
                """
            ).fetchall()
        ]
        if not nominal_times:
            return

        time_placeholders = ", ".join(["?"] * len(nominal_times))
        connection.executemany(
            "INSERT INTO snapshot_values_wide (nominal_time) VALUES (?)",
            [(nominal_time,) for nominal_time in nominal_times],
        )
        connection.executemany(
            "INSERT INTO snapshot_timestamps_wide (nominal_time) VALUES (?)",
            [(nominal_time,) for nominal_time in nominal_times],
        )

        for pv in pv_names:
            pv_column = _quote_identifier(pv)
            connection.execute(
                f"""
                UPDATE snapshot_values_wide
                SET {pv_column} = (
                    SELECT value FROM snapshot_values
                    WHERE snapshot_values.nominal_time = snapshot_values_wide.nominal_time
                      AND snapshot_values.pv = ?
                )
                WHERE nominal_time IN ({time_placeholders})
                """,
                [pv, *nominal_times],
            )
            connection.execute(
                f"""
                UPDATE snapshot_timestamps_wide
                SET {pv_column} = (
                    SELECT timestamp FROM snapshot_timestamps
                    WHERE snapshot_timestamps.nominal_time = snapshot_timestamps_wide.nominal_time
                      AND snapshot_timestamps.pv = ?
                )
                WHERE nominal_time IN ({time_placeholders})
                """,
                [pv, *nominal_times],
            )


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Return row counts for the two snapshot tables."""
    counts = {}
    table_names = [
        row[0]
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name LIKE 'snapshot_%'
            ORDER BY name
            """
        ).fetchall()
    ]
    for table_name in table_names:
        counts[table_name] = connection.execute(
            f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}"
        ).fetchone()[0]
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import XLEAP snapshot CSV caches into SQLite."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Snapshot CSV files with nominal_time,pv,timestamp,value columns.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--no-wide",
        action="store_true",
        help="Skip rebuilding snapshot_values_wide and snapshot_timestamps_wide.",
    )
    args = parser.parse_args(argv)

    imported = import_snapshot_csvs(args.csv_paths, args.db)
    with connect(args.db) as connection:
        if not args.no_wide:
            rebuild_wide_tables(connection)
        counts = table_counts(connection)

    for path, count in imported.items():
        print(f"imported {count} rows from {path}")
    print(f"database: {args.db}")
    for table_name, count in counts.items():
        print(f"{table_name}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
