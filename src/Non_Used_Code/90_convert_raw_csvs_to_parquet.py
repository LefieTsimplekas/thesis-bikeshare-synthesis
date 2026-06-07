from pathlib import Path
import duckdb


RAW_DIR = Path("data/raw")
CSV_TYPE_OVERRIDES = {
    "rideable_type": "VARCHAR",
    "started_at": "VARCHAR",
    "ended_at": "VARCHAR",
    "start_station_name": "VARCHAR",
    "start_station_id": "VARCHAR",
    "end_station_name": "VARCHAR",
    "end_station_id": "VARCHAR",
    "start_lat": "DOUBLE",
    "start_lng": "DOUBLE",
    "end_lat": "DOUBLE",
    "end_lng": "DOUBLE",
    "member_casual": "VARCHAR",
}


def sql_string(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def build_type_overrides_sql() -> str:
    return ", ".join(
        f"'{column}': '{dtype}'" for column, dtype in CSV_TYPE_OVERRIDES.items()
    )


def convert_csv_to_parquet(csv_path: Path, overwrite: bool = False) -> bool:
    parquet_path = csv_path.with_suffix(".parquet")
    temp_parquet_path = csv_path.with_suffix(".tmp.parquet")

    if parquet_path.exists() and not overwrite:
        print(f"Skipping existing parquet: {parquet_path}")
        return True

    print(f"Converting: {csv_path}")

    if temp_parquet_path.exists():
        temp_parquet_path.unlink()

    con = duckdb.connect()
    type_overrides = build_type_overrides_sql()
    csv_sql_path = sql_string(csv_path)
    temp_parquet_sql_path = sql_string(temp_parquet_path)
    query = f"""
    COPY (
        SELECT *
        FROM read_csv_auto(
            '{csv_sql_path}',
            header=True,
            union_by_name=True,
            types={{ {type_overrides} }},
            sample_size=-1
        )
    )
    TO '{temp_parquet_sql_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    try:
        con.execute(query)
    except Exception as exc:
        if temp_parquet_path.exists():
            temp_parquet_path.unlink()
        print(f"FAILED {csv_path.name}: {exc}")
        return False
    finally:
        con.close()

    temp_parquet_path.replace(parquet_path)
    print(f"Saved: {parquet_path}")
    return True


def convert_city_csvs_to_parquet(city_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    city_name = city_dir.name
    csv_files = sorted(city_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found for city: {city_name}")
        return 0, 0

    print(f"\nProcessing city: {city_name}")
    print(f"Found {len(csv_files)} CSV files")

    succeeded = 0
    failed = 0

    for csv_path in csv_files:
        if convert_csv_to_parquet(csv_path, overwrite=overwrite):
            succeeded += 1
        else:
            failed += 1

    print(f"City summary for {city_name}: succeeded={succeeded}, failed={failed}")
    return succeeded, failed


def convert_all_csvs_in_raw(overwrite: bool = False) -> bool:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    city_dirs = sorted(path for path in RAW_DIR.iterdir() if path.is_dir())

    if not city_dirs:
        print(f"No city folders found under: {RAW_DIR}")
        return False

    print(f"Found {len(city_dirs)} city folders under {RAW_DIR}")

    total_succeeded = 0
    total_failed = 0

    for city_dir in city_dirs:
        succeeded, failed = convert_city_csvs_to_parquet(city_dir, overwrite=overwrite)
        total_succeeded += succeeded
        total_failed += failed

    print("\nFinal summary")
    print(f"Converted or skipped files: {total_succeeded}")
    print(f"Failed files: {total_failed}")
    return total_failed == 0


if __name__ == "__main__":
    convert_all_csvs_in_raw(overwrite=False)
