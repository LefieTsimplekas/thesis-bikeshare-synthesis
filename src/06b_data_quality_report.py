from pathlib import Path

import duckdb
import pandas as pd

"""
Create a data quality report for the project's parquet datasets
(06b: runs after 06, checks the files of the pipeline and the model input).
The raw files (data/raw) are checked separately in 01b and l1b.

The report summarizes, per file and per column:
- data type
- total rows
- missing/null count
- missing/null percentage
- non-null count

Outputs:
- artifacts/data_quality/data_quality_report.csv
- artifacts/data_quality/data_quality_report.xlsx
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data_quality"
CSV_OUTPUT_PATH = OUTPUT_DIR / "data_quality_report.csv"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "data_quality_report.xlsx"

INPUT_DIRS = [
    DATA_DIR / "processed",
    DATA_DIR / "context",
    DATA_DIR / "model_input",
]   # τα αρχικά (data/raw) ελέγχονται στο 01b και στο l1b


def sql_string(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def find_parquet_files() -> list[Path]:
    files: list[Path] = []

    for input_dir in INPUT_DIRS:
        if input_dir.exists():
            files.extend(sorted(input_dir.rglob("*.parquet")))

    return sorted(files)


def get_column_info(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> pd.DataFrame:
    query = f"DESCRIBE SELECT * FROM read_parquet('{sql_string(parquet_path)}')"
    column_info = con.execute(query).fetchdf()
    return column_info.loc[:, ["column_name", "column_type"]]


def build_file_report(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> pd.DataFrame:
    relative_path = parquet_path.relative_to(PROJECT_ROOT)
    parquet_sql_path = sql_string(parquet_path)
    column_info = get_column_info(con, parquet_path)

    # Όλες οι μετρήσεις σε ένα ερώτημα (μία σάρωση ανά αρχείο)
    counts = ", ".join(
        'COUNT("' + c.replace('"', '""') + '")' for c in column_info["column_name"]
    )
    result = con.execute(
        f"SELECT COUNT(*), {counts} FROM read_parquet('{parquet_sql_path}')"
    ).fetchone()
    total_rows, non_null = result[0], result[1:]

    rows = []
    for (_, column), nn in zip(column_info.iterrows(), non_null):
        null_count = total_rows - nn
        rows.append(
            {
                "file": relative_path.as_posix(),
                "column": column["column_name"],
                "dtype": column["column_type"],
                "rows": total_rows,
                "null_count": null_count,
                "null_percent": round(null_count / total_rows * 100, 2) if total_rows else 0.0,
                "non_null_count": nn,
            }
        )
    return pd.DataFrame(rows)


def create_data_quality_report() -> pd.DataFrame:
    parquet_files = find_parquet_files()

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {DATA_DIR}")

    con = duckdb.connect()
    reports = []

    try:
        for parquet_path in parquet_files:
            print(f"Checking: {parquet_path.relative_to(PROJECT_ROOT)}")
            reports.append(build_file_report(con, parquet_path))
    finally:
        con.close()

    report = pd.concat(reports, ignore_index=True)
    return report.sort_values(["file", "column"]).reset_index(drop=True)


def save_report(report: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8")

    try:
        report.to_excel(EXCEL_OUTPUT_PATH, index=False)
    except ImportError:
        print("Excel export skipped because openpyxl is not installed.")
        print("Install it with: pip install openpyxl")
        return

    print("\nSaved reports:")
    print(f"- {CSV_OUTPUT_PATH}")
    print(f"- {EXCEL_OUTPUT_PATH}")


def main() -> None:
    report = create_data_quality_report()

    with_nulls = report[report["null_count"] > 0]
    print(f"\nFiles checked: {report['file'].nunique()} | columns: {len(report)}")
    if with_nulls.empty:
        print("No missing values in any file.")
    else:
        print(f"\nColumns with missing values ({len(with_nulls)}):")
        print(with_nulls.sort_values("null_percent", ascending=False).to_string(index=False))

    save_report(report)


if __name__ == "__main__":
    main()
