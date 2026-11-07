from pathlib import Path
import duckdb
import glob
from config import RAW_DIR, PROCESSED_DIR, TMP_DIR, CITY_BBOX


def sql_string(path: str | Path) -> str:
    # DuckDB needs single-quoted paths; escape any literal single quotes.
    return Path(path).as_posix().replace("'", "''")


def _feature_query(escaped_files: str, city: str, distinct: str, out_path: Path) -> str:
    lat_min, lat_max, lng_min, lng_max = CITY_BBOX[city]
    return f"""
    COPY (
        WITH typed_data AS (
            SELECT
                CAST(rideable_type AS VARCHAR) AS rideable_type,
                TRY_CAST(started_at AS TIMESTAMP) AS started_at,
                TRY_CAST(ended_at AS TIMESTAMP) AS ended_at,
                TRIM(CAST(start_station_id AS VARCHAR)) AS start_station_id,
                TRIM(CAST(end_station_id AS VARCHAR)) AS end_station_id,
                CAST(start_station_name AS VARCHAR) AS start_station_name,
                CAST(end_station_name AS VARCHAR) AS end_station_name,
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
              AND start_station_id IS NOT NULL AND TRIM(start_station_id) <> ''
              AND end_station_id IS NOT NULL AND TRIM(end_station_id) <> ''
              AND COALESCE(start_station_name,'') NOT ILIKE '%public rack%'
              AND COALESCE(end_station_name,'') NOT ILIKE '%public rack%'
              AND start_lat BETWEEN {lat_min} AND {lat_max}
              AND start_lng BETWEEN {lng_min} AND {lng_max}
              AND end_lat BETWEEN {lat_min} AND {lat_max}
              AND end_lng BETWEEN {lng_min} AND {lng_max}
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
                start_station_id,
                end_station_id,
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
            start_station_id,
            end_station_id,
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
        ORDER BY started_at, start_station_id, end_station_id, ended_at
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


def cleaning_report(city: str) -> None:
    """Διαδοχικά πλήθη ανά κανόνα καθαρισμού, για τον πίνακα ροής δεδομένων.
    Εφαρμόζει τους ίδιους κανόνες με το _feature_query. Δεν γράφει τίποτα."""
    lat_min, lat_max, lng_min, lng_max = CITY_BBOX[city]
    src = f"read_parquet('{sql_string(RAW_DIR / city)}/*.parquet', union_by_name=true)"

    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory='{sql_string(TMP_DIR)}'")
        print(f"\n===== Cleaning report: {city} =====")
        print(con.sql(f"""
            WITH t AS (
                SELECT TRY_CAST(started_at AS TIMESTAMP)        AS s,
                       TRY_CAST(ended_at   AS TIMESTAMP)        AS e,
                       TRIM(CAST(start_station_id AS VARCHAR))  AS ss,
                       TRIM(CAST(end_station_id   AS VARCHAR))  AS es,
                       COALESCE(CAST(start_station_name AS VARCHAR), '') AS sn,
                       COALESCE(CAST(end_station_name   AS VARCHAR), '') AS en,
                       TRY_CAST(start_lat AS DOUBLE) AS slat,
                       TRY_CAST(start_lng AS DOUBLE) AS slng,
                       TRY_CAST(end_lat   AS DOUBLE) AS elat,
                       TRY_CAST(end_lng   AS DOUBLE) AS elng
                FROM {src}
            )
            SELECT
                COUNT(*) AS step0_raw,
                COUNT(*) FILTER (WHERE s IS NOT NULL AND e IS NOT NULL) AS step1_valid_time,
                COUNT(*) FILTER (WHERE s IS NOT NULL AND e IS NOT NULL
                                   AND ss <> '' AND es <> '') AS step2_with_station,
                COUNT(*) FILTER (WHERE s IS NOT NULL AND e IS NOT NULL
                   AND ss <> '' AND es <> ''
                   AND sn NOT ILIKE '%public rack%'
                   AND en NOT ILIKE '%public rack%') AS step3_no_public_rack,
                COUNT(*) FILTER (WHERE s IS NOT NULL AND e IS NOT NULL
                   AND ss <> '' AND es <> ''
                   AND sn NOT ILIKE '%public rack%'
                   AND en NOT ILIKE '%public rack%'
                   AND slat BETWEEN {lat_min} AND {lat_max}
                   AND slng BETWEEN {lng_min} AND {lng_max}
                   AND elat BETWEEN {lat_min} AND {lat_max}
                   AND elng BETWEEN {lng_min} AND {lng_max}) AS step4_in_bbox
            FROM t
        """).df().T.to_string(header=False))
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
                COPY (SELECT * FROM read_parquet([{escaped_batches}])
                ORDER BY started_at, start_station_id, end_station_id, ended_at)
                TO '{sql_string(output_path)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        finally:
            con.close()

        for p in batch_outputs:
            p.unlink(missing_ok=True)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    cleaning_report("chicago")
    build_city_featured_parquet("chicago")
    cleaning_report("nyc")
    build_city_featured_parquet("nyc")
