from pathlib import Path
import duckdb

"""
Filter the feature-engineered bike trips dataset.

This script removes observations that are likely to be misleading for modeling,
especially unrealistic durations, implausibly long trips, and near-zero
movement trips that do not look like true same-station returns.

Output:
- data/processed/chicago_filtered.parquet
- data/processed/nyc_filtered.parquet
"""

PROCESSED_DIR = Path("data/processed")

# Filtering thresholds.
# Κρατάμε διαδρομές με ρεαλιστική διάρκεια και απόσταση, ώστε το μοντέλο να μη
# μάθει ακραίες ή προβληματικές εγγραφές από τα raw δεδομένα.
MIN_TRIP_DURATION_MIN = 1.0
MAX_TRIP_DURATION_MIN = 100.0
MIN_MOVING_DISTANCE_KM = 0.05
MAX_TRIP_DISTANCE_KM = 12.0


def sql_string(path: Path) -> str:
    # DuckDB paths μπαίνουν μέσα σε single quotes, άρα κάνουμε escape τυχόν '.
    return path.as_posix().replace("'", "''")


def process_city(city_name: str) -> None:
    # Input: το feature-engineered αρχείο από το προηγούμενο βήμα.
    # Output: το φιλτραρισμένο αρχείο που θα χρησιμοποιηθεί στη δειγματοληψία.
    input_path = PROCESSED_DIR / f"{city_name}_featured.parquet"
    output_path = PROCESSED_DIR / f"{city_name}_filtered.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_sql_path = sql_string(input_path)
    output_sql_path = sql_string(output_path)

    # Χρησιμοποιούμε DuckDB για να φιλτράρουμε μεγάλα parquet files χωρίς να τα
    # φορτώσουμε ολόκληρα σε pandas memory.
    con = duckdb.connect()

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

        # Diagnostic report: δεν φιλτράρει ακόμα τα δεδομένα.
        # Μετράει πόσες εγγραφές θα πιαστούν από κάθε κανόνα, για να ξέρουμε
        # αν οι κανόνες είναι λογικοί ή αν αφαιρούν υπερβολικά πολλά rows.
        report_query = f"""
        SELECT
            COUNT(*) AS total_rows,
            -- Πολύ μικρή διάρκεια: πιθανό λάθος ή μη χρήσιμη εγγραφή.
            SUM(CASE WHEN trip_duration <= {MIN_TRIP_DURATION_MIN} THEN 1 ELSE 0 END) AS too_short_duration,
            -- Πολύ μεγάλη διάρκεια: πιθανό outlier για training.
            SUM(CASE WHEN trip_duration >= {MAX_TRIP_DURATION_MIN} THEN 1 ELSE 0 END) AS too_long_duration,
            -- Πολύ μικρή απόσταση χωρίς ίδια στάση: ύποπτη εγγραφή.
            SUM(CASE
                    WHEN haversine_distance_km <= {MIN_MOVING_DISTANCE_KM} AND same_station = 0
                    THEN 1 ELSE 0
                END) AS near_zero_non_same_station,
            -- Πολύ μικρή απόσταση με ίδια στάση: την αναγνωρίζουμε ξεχωριστά,
            -- γιατί μπορεί να είναι πραγματική round-trip/return συμπεριφορά.
            SUM(CASE
                    WHEN haversine_distance_km <= {MIN_MOVING_DISTANCE_KM} AND same_station = 1
                    THEN 1 ELSE 0
                END) AS near_zero_same_station,
            -- Πολύ μεγάλη ευθύγραμμη απόσταση.
            SUM(CASE WHEN haversine_distance_km >= {MAX_TRIP_DISTANCE_KM} THEN 1 ELSE 0 END) AS too_long_distance,
            -- Συνολικές γραμμές που δεν περνούν όλους τους κανόνες.
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
                END) AS total_rows_removed
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
            WHERE
                -- Κρατάμε διαδρομές με διάρκεια μέσα στο αποδεκτό εύρος.
                trip_duration > {MIN_TRIP_DURATION_MIN}
                AND trip_duration < {MAX_TRIP_DURATION_MIN}
                AND (
                    -- Αν η απόσταση είναι σχεδόν μηδενική, την κρατάμε μόνο
                    -- όταν είναι δηλωμένη ως same-station trip.
                    haversine_distance_km > {MIN_MOVING_DISTANCE_KM}
                    OR same_station = 1
                )
                -- Αφαιρούμε πολύ μεγάλες ευθύγραμμες αποστάσεις.
                AND haversine_distance_km < {MAX_TRIP_DISTANCE_KM}
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

        rows_removed = initial_rows - final_rows
        percentage_removed = (rows_removed / initial_rows) * 100 if initial_rows else 0.0

        print("\n--- Filtering Rules ---")
        print(f"Minimum duration kept: > {MIN_TRIP_DURATION_MIN} minutes")
        print(f"Maximum duration kept: < {MAX_TRIP_DURATION_MIN} minutes")
        print(
            f"Minimum distance kept for non-same-station trips: > {MIN_MOVING_DISTANCE_KM} km"
        )
        print(f"Maximum distance kept: < {MAX_TRIP_DISTANCE_KM} km")

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
