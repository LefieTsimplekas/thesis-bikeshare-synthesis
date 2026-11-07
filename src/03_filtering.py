from pathlib import Path
import duckdb
from config import (PROCESSED_DIR, TMP_DIR,MIN_TRIP_DURATION_MIN, MAX_TRIP_DURATION_MIN,MIN_MOVING_DISTANCE_KM, MAX_TRIP_DISTANCE_KM,MAX_SPEED_KMH,)

"""
Filter the feature-engineered bike trips dataset.

This script removes observations that are likely to be misleading for modeling,
especially unrealistic durations, implausibly long trips, and near-zero
movement trips that do not look like true same-station returns.

Output:
- data/processed/chicago_filtered.parquet
- data/processed/nyc_filtered.parquet
"""

def sql_string(path: Path) -> str:
    # DuckDB paths μπαίνουν μέσα σε single quotes, άρα κάνουμε escape τυχόν '.
    return path.as_posix().replace("'", "''")


def process_city(city_name: str) -> None:
    # Input: το feature-engineered αρχείο από το προηγούμενο βήμα.
    # Output: το φιλτραρισμένο αρχείο που θα χρησιμοποιηθεί στη δειγματοληψία.
    input_path = PROCESSED_DIR / f"{city_name}_keyed.parquet"
    output_path = PROCESSED_DIR / f"{city_name}_filtered.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_sql_path = sql_string(input_path)
    output_sql_path = sql_string(output_path)

    # Χρησιμοποιούμε DuckDB για να φιλτράρουμε μεγάλα parquet files χωρίς να τα
    # φορτώσουμε ολόκληρα σε pandas memory.
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{sql_string(TMP_DIR)}'")

    try:
        print(f"\n{'=' * 50}")
        print(f"Processing city: {city_name}")
        print(f"{'=' * 50}")

        # Πρώτα μετράμε πόσες γραμμές έχει το input, ώστε στο τέλος να δούμε
        # πόσες αφαιρέθηκαν από το filtering.
        count_query = f"""
        SELECT COUNT(*) AS n_rows
        FROM read_parquet('{input_sql_path}')
        """
        initial_rows = con.execute(count_query).fetchone()[0]
        print(f"Rows before filtering: {initial_rows}")

        dur_ok   = f"trip_duration > {MIN_TRIP_DURATION_MIN} AND trip_duration < {MAX_TRIP_DURATION_MIN}"
        mind_ok  = f"(haversine_distance_km > {MIN_MOVING_DISTANCE_KM} OR same_station = 1)"
        maxd_ok  = f"haversine_distance_km < {MAX_TRIP_DISTANCE_KM}"
        speed_ok = f"(same_station = 1 OR haversine_distance_km * 60 <= {MAX_SPEED_KMH} * trip_duration)"

        # Diagnostic report: δεν φιλτράρει ακόμα τα δεδομένα.
        # Μετράει πόσες εγγραφές θα πιαστούν από κάθε κανόνα, για να ξέρουμε
        # αν οι κανόνες είναι λογικοί ή αν αφαιρούν υπερβολικά πολλά rows.
        report_query = f"""
        SELECT
            COUNT(*)                                                                 AS step0_input,
            COUNT(*) FILTER (WHERE trip_duration < 0)                                AS info_negative_duration,
            COUNT(*) FILTER (WHERE {dur_ok})                                         AS step1_duration,
            COUNT(*) FILTER (WHERE {dur_ok} AND {mind_ok})                           AS step2_min_distance,
            COUNT(*) FILTER (WHERE {dur_ok} AND {mind_ok} AND {maxd_ok})             AS step3_max_distance,
            COUNT(*) FILTER (WHERE {dur_ok} AND {mind_ok} AND {maxd_ok} AND {speed_ok}) AS step4_speed
        FROM read_parquet('{input_sql_path}')
        """
        report = con.execute(report_query).fetchdf()

        # Actual filtering query.
        # Δημιουργεί νέο parquet κρατώντας μόνο τις στήλες που χρειάζονται στα
        # επόμενα βήματα και μόνο τις γραμμές που περνούν τους κανόνες.
        filter_query = f"""
        COPY (
            SELECT
                rideable_type,
                started_at,
                start_station_id,
                end_station_id,
                start_station_key,
                end_station_key,
                member_casual,
                city,
                trip_duration,
                same_station,
                day_of_week,
                month,
                is_weekend,
                part_of_day,
                hour_sin,
                hour_cos,
                haversine_distance_km,
                bearing_sin,
                bearing_cos,
                hour
            FROM read_parquet('{input_sql_path}')
            WHERE {dur_ok} AND {mind_ok} AND {maxd_ok} AND {speed_ok}
            ORDER BY started_at, start_station_id, end_station_id, trip_duration
        )
        TO '{output_sql_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """

        con.execute(filter_query)

        # Μετράμε το output για να υπολογίσουμε πόσα rows αφαιρέθηκαν συνολικά.
        final_count_query = f"""
        SELECT COUNT(*) AS n_rows
        FROM read_parquet('{output_sql_path}')
        """
        final_rows = con.execute(final_count_query).fetchone()[0]
        corr = con.execute(f"""
            WITH t AS (
                SELECT trip_duration, haversine_distance_km,
                       rank() OVER (ORDER BY trip_duration)         AS r_dur,
                       rank() OVER (ORDER BY haversine_distance_km) AS r_dist
                FROM read_parquet('{output_sql_path}')
                WHERE same_station = 0
            )
            SELECT CORR(trip_duration, haversine_distance_km) AS pearson,
                   CORR(r_dur, r_dist)                        AS spearman
            FROM t
        """).fetchdf()
        print("\n--- Duration–distance correlation (non-round trips) ---")
        print(corr)

        rows_removed = initial_rows - final_rows
        percentage_removed = (rows_removed / initial_rows) * 100 if initial_rows else 0.0

        print("\n--- Filtering Rules ---")
        print(f"Minimum duration kept: > {MIN_TRIP_DURATION_MIN} minutes")
        print(f"Maximum duration kept: < {MAX_TRIP_DURATION_MIN} minutes")
        print(
            f"Minimum distance kept for non-same-station trips: > {MIN_MOVING_DISTANCE_KM} km"
        )
        print(f"Maximum distance kept: < {MAX_TRIP_DISTANCE_KM} km")
        print(f"Maximum speed kept for non-round trips: <= {MAX_SPEED_KMH} km/h")

        print("\n--- Diagnostic Report ---")
        print(report)

        print("\n--- Filtering Report ---")
        print(f"Rows before: {initial_rows}")
        print(f"Rows after: {final_rows}")
        print(f"Rows removed: {rows_removed}")
        print(f"Percentage removed: {percentage_removed:.2f}%")
        print(f"\nSaved filtered data to: {output_path}")
    finally:
        # Κλείνουμε πάντα το connection, ακόμα και αν κάποιο query αποτύχει.
        con.close()


if __name__ == "__main__":
    # Τρέχει το ίδιο filtering pipeline και για τις δύο πόλεις.
    process_city("chicago")
    process_city("nyc")
