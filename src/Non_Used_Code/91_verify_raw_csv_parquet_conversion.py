from pathlib import Path
import importlib.util
import duckdb


def load_csv_conversion_module():
    module_path = Path(__file__).with_name("90_convert_raw_csvs_to_parquet.py")
    spec = importlib.util.spec_from_file_location("csv_to_parquet_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


csv_to_parquet_module = load_csv_conversion_module()
build_type_overrides_sql = csv_to_parquet_module.build_type_overrides_sql
sql_string = csv_to_parquet_module.sql_string
RAW_DIR = csv_to_parquet_module.RAW_DIR


def count_csv_rows(csv_path: Path) -> int:
    con = duckdb.connect()
    type_overrides = build_type_overrides_sql()
    query = f"""
    SELECT COUNT(*)
    FROM read_csv_auto(
        '{sql_string(csv_path)}',
        header=True,
        union_by_name=True,
        types={{ {type_overrides} }},
        sample_size=-1
    )
    """
    try:
        return con.execute(query).fetchone()[0]
    finally:
        con.close()


def count_parquet_rows(parquet_path: Path) -> int:
    con = duckdb.connect()
    query = f"SELECT COUNT(*) FROM read_parquet('{sql_string(parquet_path)}')"
    try:
        return con.execute(query).fetchone()[0]
    finally:
        con.close()


def verify_city(city_dir: Path) -> tuple[int, int]:
    city_name = city_dir.name
    csv_files = sorted(city_dir.glob("*.csv"))

    if not csv_files:
        print(f"\nProcessing city: {city_name}")
        print("No CSV files found.")
        return 0, 0

    print(f"\nProcessing city: {city_name}")
    print(f"Found {len(csv_files)} CSV files to verify")

    verified = 0
    failed = 0

    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")

        if not parquet_path.exists():
            failed += 1
            print(f"FAILED missing parquet: {parquet_path}")
            continue

        try:
            csv_rows = count_csv_rows(csv_path)
            parquet_rows = count_parquet_rows(parquet_path)
        except Exception as exc:
            failed += 1
            print(f"FAILED {csv_path.name}: {exc}")
            continue

        if csv_rows != parquet_rows:
            failed += 1
            print(
                f"FAILED {csv_path.name}: csv_rows={csv_rows}, parquet_rows={parquet_rows}"
            )
            continue

        verified += 1
        print(f"OK {csv_path.name}: {csv_rows} rows")

    print(f"City summary for {city_name}: verified={verified}, failed={failed}")
    return verified, failed


def verify_all_conversions() -> bool:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    city_dirs = sorted(path for path in RAW_DIR.iterdir() if path.is_dir())

    if not city_dirs:
        print(f"No city folders found under: {RAW_DIR}")
        return False

    total_verified = 0
    total_failed = 0

    for city_dir in city_dirs:
        verified, failed = verify_city(city_dir)
        total_verified += verified
        total_failed += failed

    print("\nFinal summary")
    print(f"Verified files: {total_verified}")
    print(f"Failed files: {total_failed}")

    if total_failed == 0:
        print("All CSV to parquet checks passed.")
        return True

    print("Some checks failed. Do not delete the CSV files yet.")
    return False


if __name__ == "__main__":
    verify_all_conversions()
