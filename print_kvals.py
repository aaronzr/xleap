from __future__ import annotations

import argparse
import csv
from pathlib import Path


def iter_kvals_values(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) > 1:
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

    for values in iter_kvals_values(Path(args.csv_path)):
        print(",".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
