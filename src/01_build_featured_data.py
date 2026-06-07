from pathlib import Path
import duckdb
import glob

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TMP_DIR = Path("data/tmp")


def sql_string(path: str | Path) -> str:
    # DuckDB needs single-quoted paths; escape any literal single quotes.
    return Path(path).as_posix().replace("'", "''")


def _feature_query(escaped_files: str, city: str, distinct: str, out_path: Path) -> str:
    return f"""
    COPY (
        WITH typed_data AS (
            SELECT
                CAST(rideable_type AS VARCHAR) AS rideable_type,
                TRY_CAST(started_at AS TIMESTAMP) AS started_at,
                TRY_CAST(ended_at AS TIMESTAMP) AS ended_at,
                CAST(start_station_id AS VARCHAR) AS start_station_id,
                CAST(end_station_id AS VARCHAR) AS end_station_id,
                TRY_CAST(start_lat AS DOUBLE) AS start_lat,
                TRY_CAST(start_lng AS DOUBLE) AS start_lng,
                TRY_CAST(end_lat AS DOUBLE) AS end_lat,
                TRY_CAST(end_lng AS DOUBLE) AS end_lng,
                CAST(member_casual AS VARCHAR) AS member_casual,
                '{city}' AS city
            FROM read_parquet(
                [{escaped_files}],
                union_by_name=True
            )
        ),
        cleaned_data AS (
            SELECT {distinct}*
            FROM typed_data
            WHERE started_at IS NOT NULL
              AND ended_at IS NOT NULL
              AND start_lat IS NOT NULL
              AND start_lng IS NOT NULL
              AND end_lat IS NOT NULL
              AND end_lng IS NOT NULL
              AND start_lat BETWEEN -90 AND 90
              AND start_lng BETWEEN -180 AND 180
              AND end_lat BETWEEN -90 AND 90
              AND end_lng BETWEEN -180 AND 180
        ),
        base_features AS (
            SELECT
                rideable_type,
                started_at,
                ended_at,
                start_station_id,
                end_station_id,
                start_lat,
                start_lng,
                end_lat,
                end_lng,
                member_casual,
                city,
                date_diff('second', started_at, ended_at) / 60.0 AS trip_duration,
                CASE
                    WHEN start_station_id IS NOT NULL
                     AND end_station_id IS NOT NULL
                     AND start_station_id = end_station_id
                    THEN 1
                    ELSE 0
                END AS same_station,
                CAST(EXTRACT(HOUR FROM started_at) AS INTEGER) AS hour,
                CAST((EXTRACT(ISODOW FROM started_at) + 6) % 7 AS INTEGER) AS day_of_week,
                CAST(EXTRACT(MONTH FROM started_at) AS INTEGER) AS month,
                end_lat - start_lat AS delta_lat,
                end_lng - start_lng AS delta_lng,
                -- straight-line trip distance in km
                6371.0 * 2 * ASIN(
                    SQRT(
                        POWER(SIN(RADIANS(end_lat - start_lat) / 2), 2)
                        + COS(RADIANS(start_lat))
                        * COS(RADIANS(end_lat))
                        * POWER(SIN(RADIANS(end_lng - start_lng) / 2), 2)
                    )
                ) AS haversine_distance_km,
                DEGREES(
                    ATAN2(
                        SIN(RADIANS(end_lng - start_lng)) * COS(RADIANS(end_lat)),
                        COS(RADIANS(start_lat)) * SIN(RADIANS(end_lat))
                        - SIN(RADIANS(start_lat)) * COS(RADIANS(end_lat))
                        * COS(RADIANS(end_lng - start_lng))
                    )
                ) AS bearing_raw
            FROM cleaned_data
        ),
        engineered_data AS (
            SELECT
                rideable_type,
                started_at,
                ended_at,
                start_lat,
                start_lng,
                end_lat,
                end_lng,
                member_casual,
                city,
                trip_duration,
                same_station,
                hour,
                day_of_week,
                month,
                CASE WHEN day_of_week IN (5, 6) THEN 1 ELSE 0 END AS is_weekend,
                CASE
                    WHEN hour BETWEEN 5 AND 10 THEN 'morning_peak'
                    WHEN hour BETWEEN 11 AND 15 THEN 'midday'
                    WHEN hour BETWEEN 16 AND 19 THEN 'evening_peak'
                    ELSE 'night'
                END AS part_of_day,
                -- sin/cos so that hour 23 and 0 are adjacent
                SIN(2 * PI() * hour / 24.0) AS hour_sin,
                COS(2 * PI() * hour / 24.0) AS hour_cos,
                delta_lat,
                delta_lng,
                haversine_distance_km,
                -- ATAN2 can return negative degrees; normalize to 0..360
                CASE
                    WHEN bearing_raw < 0 THEN bearing_raw + 360
                    ELSE bearing_raw
                END AS bearing
            FROM base_features
        )
        SELECT
            rideable_type,
            started_at,
            ended_at,
            start_lat,
            start_lng,
            end_lat,
            end_lng,
            member_casual,
            city,
            trip_duration,
            same_station,
            hour,
            day_of_week,
            month,
            is_weekend,
            part_of_day,
            hour_sin,
            hour_cos,
            delta_lat,
            delta_lng,
            haversine_distance_km,
            bearing,
            SIN(RADIANS(bearing)) AS bearing_sin,
            COS(RADIANS(bearing)) AS bearing_cos
        FROM engineered_data
    )
    TO '{sql_string(out_path)}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """


def _run_query(query: str) -> None:
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory='{sql_string(TMP_DIR)}'")
        con.execute(query)
    finally:
        con.close()


def build_city_featured_parquet(city: str, remove_duplicates: bool = False, batch_size: int = 12) -> None:
    city_dir = RAW_DIR / city
    files = sorted(glob.glob(str(city_dir / "*.parquet")))

    if not files:
        raise ValueError(f"No parquet files found for: {city}")

    print(f"Found {len(files)} files for {city}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / f"{city}_featured.parquet"
    distinct = "DISTINCT " if remove_duplicates else ""

    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    if len(batches) == 1:
        print(f"Processing {city}...")
        escaped = ", ".join(f"'{sql_string(f)}'" for f in files)
        _run_query(_feature_query(escaped, city, distinct, output_path))
    else:
        batch_outputs = []
        for i, batch in enumerate(batches):
            print(f"Processing {city} batch {i + 1}/{len(batches)} ({len(batch)} files)...")
            escaped = ", ".join(f"'{sql_string(f)}'" for f in batch)
            tmp_out = TMP_DIR / f"{city}_batch_{i}.parquet"
            _run_query(_feature_query(escaped, city, distinct, tmp_out))
            batch_outputs.append(tmp_out)

        print(f"Merging {len(batch_outputs)} batches...")
        escaped_batches = ", ".join(f"'{sql_string(p)}'" for p in batch_outputs)
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (SELECT * FROM read_parquet([{escaped_batches}]))
                TO '{sql_string(output_path)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        finally:
            con.close()

        for p in batch_outputs:
            p.unlink(missing_ok=True)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    build_city_featured_parquet("chicago")
    build_city_featured_parquet("nyc")
