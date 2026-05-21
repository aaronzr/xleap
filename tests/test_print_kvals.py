from pathlib import Path

from print_kvals import iter_kvals_values, main


def test_iter_kvals_values_skips_datetime_column(tmp_path: Path):
    csv_path = tmp_path / "kvals.csv"
    csv_path.write_text(
        "Datetime,26.0,27.0\n"
        "2021-01-01 00:00:00,5.1,5.2\n"
        "2021-01-01 04:00:00,5.3,5.4\n",
        encoding="utf-8",
    )

    assert list(iter_kvals_values(csv_path)) == [["5.1", "5.2"], ["5.3", "5.4"]]


def test_main_prints_csv_values(tmp_path: Path, capsys):
    csv_path = tmp_path / "kvals.csv"
    csv_path.write_text(
        "Datetime,26.0,27.0\n"
        "2021-01-01 00:00:00,5.1,5.2\n"
        "2021-01-01 04:00:00,5.3,5.4\n",
        encoding="utf-8",
    )

    assert main([str(csv_path)]) == 0
    assert capsys.readouterr().out == "5.1,5.2\n5.3,5.4\n"
