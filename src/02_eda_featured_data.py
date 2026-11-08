from pathlib import Path
import duckdb
from config import (PROCESSED_DIR, TMP_DIR, MIN_TRIP_DURATION_MIN, MAX_TRIP_DURATION_MIN, MIN_MOVING_DISTANCE_KM, MAX_TRIP_DISTANCE_KM, MAX_SPEED_KMH,)
import pandas as pd 
pd.set_option("display.width",250)
pd.set_option("display.max_columns",50)

FILES = [
    str(PROCESSED_DIR / "chicago_keyed.parquet"),
    str(PROCESSED_DIR / "nyc_keyed.parquet"),
]

FILES_SQL = ", ".join(f"'{file}'" for file in FILES)


def run_query(con: duckdb.DuckDBPyConnection, title: str, query: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))  


def main() -> None:
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")

    parquet_source = f"read_parquet([{FILES_SQL}])"
    eda_source = f"""
    (
        SELECT
            *,
            CASE
                WHEN trip_duration <= {MIN_TRIP_DURATION_MIN} THEN 'too_short'
                WHEN trip_duration >= {MAX_TRIP_DURATION_MIN} THEN 'too_long'
                ELSE 'in_range'
            END AS duration_flag,
            CASE
                WHEN haversine_distance_km >= {MAX_TRIP_DISTANCE_KM} THEN 'too_far'
                WHEN haversine_distance_km <= {MIN_MOVING_DISTANCE_KM} AND same_station = 0
                    THEN 'too_short_non_same_station'
                WHEN haversine_distance_km <= {MIN_MOVING_DISTANCE_KM} AND same_station = 1
                    THEN 'same_station_near_zero'
                ELSE 'in_range'
            END AS distance_flag
        FROM {parquet_source}
    )
    """

    query_basic_summary = f"""
    SELECT
        city,
        COUNT(*) AS n_rows,
        AVG(trip_duration) AS avg_duration,
        MEDIAN(trip_duration) AS median_duration,
        MIN(trip_duration) AS min_duration,
        MAX(trip_duration) AS max_duration,
        AVG(haversine_distance_km) AS avg_distance,
        MEDIAN(haversine_distance_km) AS median_distance,
        MIN(haversine_distance_km) AS min_distance,
        MAX(haversine_distance_km) AS max_distance
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_duration_percentiles = f"""
    SELECT
        city,
        quantile_cont(trip_duration, 0.01) AS q01,
        quantile_cont(trip_duration, 0.05) AS q05,
        quantile_cont(trip_duration, 0.25) AS q25,
        quantile_cont(trip_duration, 0.50) AS median,
        quantile_cont(trip_duration, 0.75) AS q75,
        quantile_cont(trip_duration, 0.95) AS q95,
        quantile_cont(trip_duration, 0.99) AS q99
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_distance_percentiles = f"""
    SELECT
        city,
        quantile_cont(haversine_distance_km, 0.01) AS q01,
        quantile_cont(haversine_distance_km, 0.05) AS q05,
        quantile_cont(haversine_distance_km, 0.25) AS q25,
        quantile_cont(haversine_distance_km, 0.50) AS median,
        quantile_cont(haversine_distance_km, 0.75) AS q75,
        quantile_cont(haversine_distance_km, 0.95) AS q95,
        quantile_cont(haversine_distance_km, 0.99) AS q99
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_duration_threshold_impact = f"""
    SELECT
        city,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN trip_duration <= {MIN_TRIP_DURATION_MIN} THEN 1 ELSE 0 END) AS too_short_rows,
        SUM(CASE WHEN trip_duration >= {MAX_TRIP_DURATION_MIN} THEN 1 ELSE 0 END) AS too_long_rows,
        ROUND(100.0 * SUM(CASE WHEN trip_duration <= {MIN_TRIP_DURATION_MIN} THEN 1 ELSE 0 END) / COUNT(*), 4) AS pct_too_short,
        ROUND(100.0 * SUM(CASE WHEN trip_duration >= {MAX_TRIP_DURATION_MIN} THEN 1 ELSE 0 END) / COUNT(*), 4) AS pct_too_long
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_distance_threshold_impact = f"""
    SELECT
        city,
        COUNT(*) AS total_rows,
        SUM(CASE
                WHEN haversine_distance_km <= {MIN_MOVING_DISTANCE_KM} AND same_station = 0
                THEN 1 ELSE 0
            END) AS too_short_non_same_station_rows,
        SUM(CASE
                WHEN haversine_distance_km <= {MIN_MOVING_DISTANCE_KM} AND same_station = 1
                THEN 1 ELSE 0
            END) AS same_station_near_zero_rows,
        SUM(CASE WHEN haversine_distance_km >= {MAX_TRIP_DISTANCE_KM} THEN 1 ELSE 0 END) AS too_far_rows
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_filter_rule_summary = f"""
    SELECT
        city,
        COUNT(*) AS total_rows,
        SUM(CASE
                WHEN trip_duration > {MIN_TRIP_DURATION_MIN}
                 AND trip_duration < {MAX_TRIP_DURATION_MIN}
                 AND (
                    haversine_distance_km > {MIN_MOVING_DISTANCE_KM}
                    OR same_station = 1
                 )
                 AND haversine_distance_km < {MAX_TRIP_DISTANCE_KM}
                THEN 1 ELSE 0
            END) AS rows_kept_by_filter,
        SUM(CASE
                WHEN NOT (
                    trip_duration > {MIN_TRIP_DURATION_MIN}
                    AND trip_duration < {MAX_TRIP_DURATION_MIN}
                    AND (
                        haversine_distance_km > {MIN_MOVING_DISTANCE_KM}
                        OR same_station = 1
                    )
                    AND haversine_distance_km < {MAX_TRIP_DISTANCE_KM}
                )
                THEN 1 ELSE 0
            END) AS rows_removed_by_filter,
        ROUND(
            100.0 * SUM(CASE
                WHEN NOT (
                    trip_duration > {MIN_TRIP_DURATION_MIN}
                    AND trip_duration < {MAX_TRIP_DURATION_MIN}
                    AND (
                        haversine_distance_km > {MIN_MOVING_DISTANCE_KM}
                        OR same_station = 1
                    )
                    AND haversine_distance_km < {MAX_TRIP_DISTANCE_KM}
                )
                THEN 1 ELSE 0
            END) / COUNT(*),
            4
        ) AS pct_removed_by_filter
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_suspicious_examples = f"""
    SELECT
        city,
        duration_flag,
        distance_flag,
        COUNT(*) AS n_rows
    FROM {eda_source}
    WHERE duration_flag <> 'in_range' OR distance_flag <> 'in_range'
    GROUP BY city, duration_flag, distance_flag
    ORDER BY city, n_rows DESC
    """

    query_short_distance_breakdown = f"""
    SELECT
        city,
        same_station,
        rideable_type,
        COUNT(*) AS n_rows,
        AVG(trip_duration) AS avg_duration,
        AVG(haversine_distance_km) AS avg_distance
    FROM {parquet_source}
    WHERE haversine_distance_km <= {MIN_MOVING_DISTANCE_KM}
    GROUP BY city, same_station, rideable_type
    ORDER BY city, same_station, n_rows DESC
    """

    query_long_duration_breakdown = f"""
    SELECT
        city,
        member_casual,
        rideable_type,
        COUNT(*) AS n_rows,
        AVG(trip_duration) AS avg_duration,
        AVG(haversine_distance_km) AS avg_distance
    FROM {parquet_source}
    WHERE trip_duration >= {MAX_TRIP_DURATION_MIN}
    GROUP BY city, member_casual, rideable_type
    ORDER BY city, n_rows DESC
    """

    query_hour_distribution = f"""
    SELECT
        city,
        hour,
        COUNT(*) AS n_trips
    FROM {parquet_source}
    GROUP BY city, hour
    ORDER BY city, hour
    """

    query_weekday_distribution = f"""
    SELECT
        city,
        day_of_week,
        COUNT(*) AS n_trips
    FROM {parquet_source}
    GROUP BY city, day_of_week
    ORDER BY city, day_of_week
    """

    query_member_distribution = f"""
    SELECT
        city,
        member_casual,
        COUNT(*) AS n_trips
    FROM {parquet_source}
    GROUP BY city, member_casual
    ORDER BY city, member_casual
    """

    query_same_station_rate = f"""
    SELECT
        city,
        AVG(same_station) AS same_station_rate
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_rideable_distribution = f"""
    SELECT
        city,
        rideable_type,
        COUNT(*) AS n_trips
    FROM {parquet_source}
    GROUP BY city, rideable_type
    ORDER BY city, rideable_type
    """

    query_duration_distance_corr = f"""
    SELECT
        city,
        CORR(trip_duration, haversine_distance_km) AS duration_distance_corr
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_part_of_day_summary = f"""
    SELECT
        city,
        part_of_day,
        AVG(trip_duration) AS avg_duration
    FROM {parquet_source}
    GROUP BY city, part_of_day
    ORDER BY city, part_of_day
    """

    query_correlation_summary = f"""
    SELECT
        city,
        CORR(trip_duration, hour) AS corr_duration_hour,
        CORR(trip_duration, haversine_distance_km) AS corr_duration_distance,
        CORR(haversine_distance_km, hour) AS corr_distance_hour
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_cross_city = f"""
    SELECT
        city,
        AVG(trip_duration) AS avg_duration,
        AVG(haversine_distance_km) AS avg_distance,
        AVG(same_station) AS same_station_rate
    FROM {parquet_source}
    GROUP BY city
    ORDER BY city
    """

    query_user_type_summary = f"""
    SELECT
        city,
        member_casual,
        AVG(trip_duration) AS avg_duration,
        AVG(haversine_distance_km) AS avg_distance
    FROM {parquet_source}
    GROUP BY city, member_casual
    ORDER BY city, member_casual
    """

    query_peak_hours = f"""
    SELECT
        city,
        hour,
        n_trips
    FROM (
        SELECT
            city,
            hour,
            COUNT(*) AS n_trips,
            ROW_NUMBER() OVER (PARTITION BY city ORDER BY COUNT(*) DESC, hour ASC) AS rn
        FROM {parquet_source}
        GROUP BY city, hour
    )
    WHERE rn <= 3
    ORDER BY city, n_trips DESC, hour
    """

    query_negative_duration = f"""
    SELECT city, COUNT(*) FILTER (WHERE trip_duration < 0) AS negative_duration
    FROM {parquet_source}
    GROUP BY city ORDER BY city
    """

    query_speed = f"""
    SELECT city, rideable_type,
           COUNT(*) FILTER (WHERE haversine_distance_km / (trip_duration / 60) > {MAX_SPEED_KMH}) AS too_fast,
           COUNT(*) AS n
    FROM {parquet_source}
    WHERE trip_duration > {MIN_TRIP_DURATION_MIN} AND trip_duration < {MAX_TRIP_DURATION_MIN}
      AND haversine_distance_km < {MAX_TRIP_DISTANCE_KM} AND same_station = 0
    GROUP BY ALL ORDER BY ALL
    """

    queries = [
        ("Basic Summary Statistics", query_basic_summary),
        ("Trip Duration Percentiles", query_duration_percentiles),
        ("Trip Distance Percentiles", query_distance_percentiles),
        ("Duration Threshold Impact", query_duration_threshold_impact),
        ("Distance Threshold Impact", query_distance_threshold_impact),
        ("Overall Filter Rule Impact", query_filter_rule_summary),
        ("Suspicious Combination Summary", query_suspicious_examples),
        ("Short Distance Breakdown", query_short_distance_breakdown),
        ("Long Duration Breakdown", query_long_duration_breakdown),
        ("Hour Distribution", query_hour_distribution),
        ("Day of Week Distribution", query_weekday_distribution),
        ("Member vs Casual Distribution", query_member_distribution),
        ("Same Station Rate", query_same_station_rate),
        ("Rideable Type Distribution", query_rideable_distribution),
        ("Duration vs Distance Correlation", query_duration_distance_corr),
        ("Average Duration by Part of Day", query_part_of_day_summary),
        ("Correlation Summary", query_correlation_summary),
        ("Cross-City Comparison", query_cross_city),
        ("Summary by User Type", query_user_type_summary),
        ("Top Peak Hours Per City", query_peak_hours),
        ("Negative Durations", query_negative_duration),
        ("Impossible Speeds", query_speed),
    ]

    for title, query in queries:
        run_query(con, title, query)

    con.close()


if __name__ == "__main__":
    main()
