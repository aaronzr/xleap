from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def iter_kvals_values(csv_path: Path):
    """Yield each data row's K-values from a kvals CSV file."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if not header:
            return
        if header[0].strip().lower() != "datetime":
            raise ValueError(
                "Expected first CSV column to be 'Datetime' (case-insensitive)."
            )
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Malformed CSV row at line {line_number}: expected at least 2 columns."
                )
            yield row[1:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print K values from kvals.csv")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="kvals.csv",
        help="Path to CSV file (default: kvals.csv)",
    )
    args = parser.parse_args(argv)

    try:
        for values in iter_kvals_values(Path(args.csv_path)):
            print(",".join(values))
    except ValueError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
