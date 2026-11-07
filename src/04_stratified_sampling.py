from pathlib import Path
import duckdb

from config import PROCESSED_DIR, TMP_DIR, STRATA_COLS, SAMPLING_SEED
from config import TARGET_ROWS as TARGET_ROWS_PER_CITY

"""
Validate strata and create sampled datasets for synthetic data training.

This script:
1. Loads filtered datasets per city
2. Reports useful strata diagnostics before sampling
3. Applies stratified sampling with an exact target row count per city
4. Saves per-city samples (the mixed scenario-C set is built after the day split)

Outputs:
- data/processed/chicago_sampled.parquet
- data/processed/nyc_sampled.parquet
"""
def sql_string(path: Path) -> str:
    # DuckDB paths μπαίνουν μέσα σε single quotes, άρα κάνουμε escape τυχόν '.
    return path.as_posix().replace("'", "''")


def get_filtered_path(city: str) -> Path:
    return PROCESSED_DIR / f"{city}_filtered.parquet"


def print_city_strata_report(city: str, target_rows: int) -> None:
    input_path = get_filtered_path(city)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_sql = sql_string(input_path)
    con = duckdb.connect()

    print(f"\n{'=' * 60}")
    print(f"Strata report: {city}")
    print(f"{'=' * 60}")

    summary_query = f"""
    WITH strata AS (
        SELECT {', '.join(STRATA_COLS)}, COUNT(*) AS strata_size
        FROM read_parquet('{input_sql}')
        GROUP BY {', '.join(STRATA_COLS)}
    )
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{input_sql}')) AS total_rows,
        COUNT(*) AS n_strata,
        MIN(strata_size) AS min_strata_size,
        MAX(strata_size) AS max_strata_size,
        AVG(strata_size) AS avg_strata_size,
        SUM(CASE WHEN strata_size < 10 THEN 1 ELSE 0 END) AS strata_below_10,
        SUM(CASE WHEN strata_size < 50 THEN 1 ELSE 0 END) AS strata_below_50,
        SUM(CASE WHEN strata_size < 100 THEN 1 ELSE 0 END) AS strata_below_100,
        LEAST({target_rows}, (SELECT COUNT(*) FROM read_parquet('{input_sql}'))) AS effective_target_rows
    FROM strata
    """
    print(con.execute(summary_query).fetchdf())

    smallest_query = f"""
    SELECT {', '.join(STRATA_COLS)}, COUNT(*) AS n
    FROM read_parquet('{input_sql}')
    GROUP BY {', '.join(STRATA_COLS)}
    ORDER BY n ASC, {', '.join(STRATA_COLS)}
    LIMIT 10
    """
    print("\n--- Smallest strata ---")
    print(con.execute(smallest_query).fetchdf())
    con.close()


def _build_allocation_table(input_path: Path, target_rows: int) -> Path:
    """Pre-compute exact per-stratum row counts and save as a small parquet (192 rows)."""
    strata_cols_sql = ", ".join(STRATA_COLS)
    allocation_path = TMP_DIR / f"{input_path.stem}_allocation.parquet"
    input_sql = sql_string(input_path)

    con = duckdb.connect()
    try:
        con.execute(f"""
        COPY (
            WITH total_count AS (
                SELECT COUNT(*) AS total_rows FROM read_parquet('{input_sql}')
            ),
            effective_target AS (
                SELECT LEAST({target_rows}, total_rows) AS target_rows, total_rows
                FROM total_count
            ),
            strata_counts AS (
                SELECT {strata_cols_sql}, COUNT(*) AS strata_size
                FROM read_parquet('{input_sql}')
                GROUP BY {strata_cols_sql}
            ),
            raw_targets AS (
                SELECT s.*, e.target_rows, e.total_rows,
                    FLOOR(s.strata_size * 1.0 * e.target_rows / e.total_rows) AS base_target,
                    (s.strata_size * 1.0 * e.target_rows / e.total_rows)
                    - FLOOR(s.strata_size * 1.0 * e.target_rows / e.total_rows) AS fractional_part
                FROM strata_counts s CROSS JOIN effective_target e
            ),
            ranked_targets AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        ORDER BY fractional_part DESC, {strata_cols_sql}
                    ) AS fractional_rank,
                    CAST(base_target AS BIGINT) AS base_target_int,
                    CAST(target_rows - SUM(CAST(base_target AS BIGINT)) OVER () AS BIGINT) AS extra_rows_to_assign
                FROM raw_targets
            )
            SELECT {strata_cols_sql}, strata_size,
                base_target_int
                + CASE WHEN fractional_rank <= extra_rows_to_assign THEN 1 ELSE 0 END AS target_in_stratum
            FROM ranked_targets
        )
        TO '{sql_string(allocation_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()

    return allocation_path


def _sample_city(input_path: Path, target_rows: int, output_path: Path) -> None:
    """Sample a city to target_rows using proportional stratified sampling.

    Processes one month at a time to keep each query's working set at ~4M rows
    instead of the full 45M+, which avoids OOM on the window function.
    """
    tmp_dir = TMP_DIR.resolve()

    print("  Computing allocation table...")
    allocation_path = _build_allocation_table(input_path, target_rows)

    # month is one of the STRATA_COLS so each monthly chunk naturally maps to
    # a subset of strata rows. The window partition excludes month since we
    # already filter to a single month.
    non_month_cols = [c for c in STRATA_COLS if c != "month"]
    partition_sql = ", ".join(f"d.{c}" for c in non_month_cols)
    join_condition = " AND ".join(f"d.{col} = a.{col}" for col in STRATA_COLS)

    month_outputs = []
    for month in range(1, 13):
        month_out = tmp_dir / f"{input_path.stem}_month_{month}.parquet"

        query = f"""
        COPY (
            WITH sampled AS (
                SELECT d.*,
                    a.target_in_stratum,
                    ROW_NUMBER() OVER (
                        PARTITION BY {partition_sql}
                        ORDER BY hash(d.started_at, d.start_station_id, d.end_station_id, d.trip_duration, {SAMPLING_SEED}),
                            d .started_at, d.start_station_id, d.end_station_id, d.trip_duration
                    ) AS rn
                FROM read_parquet('{sql_string(input_path)}') d
                JOIN read_parquet('{sql_string(allocation_path)}') a ON {join_condition}
                WHERE d.month = {month}
            )
            SELECT * EXCLUDE (target_in_stratum, rn)
            FROM sampled
            WHERE rn <= target_in_stratum
        )
        TO '{sql_string(month_out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        con = duckdb.connect()
        try:
            con.execute(f"SET temp_directory='{sql_string(tmp_dir)}'")
            con.execute(query)
        finally:
            con.close()

        month_outputs.append(month_out)
        print(f"  month {month}/12 done")

    escaped = ", ".join(f"'{sql_string(p)}'" for p in month_outputs)
    con = duckdb.connect()
    try:
        con.execute(f"""
            COPY (SELECT * FROM read_parquet([{escaped}])
                ORDER BY started_at, start_station_id, end_station_id, trip_duration)
            TO '{sql_string(output_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()

    allocation_path.unlink(missing_ok=True)
    for p in month_outputs:
        p.unlink(missing_ok=True)

def representativeness_report(city: str, sample_path: Path) -> None:
    pop = f"read_parquet('{sql_string(get_filtered_path(city))}')"
    smp = f"read_parquet('{sql_string(sample_path)}')"
    con = duckdb.connect()
    try:
        print(f"\n--- Representativeness: {city} (TVD, sample vs population) ---")
        for col in STRATA_COLS + ["hour", "day_of_week", "same_station"]:
            tvd = con.execute(f"""
                WITH p AS (SELECT {col} AS v, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS f FROM {pop} GROUP BY 1),
                     s AS (SELECT {col} AS v, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS f FROM {smp} GROUP BY 1)
                SELECT 0.5 * SUM(ABS(COALESCE(p.f, 0) - COALESCE(s.f, 0)))
                FROM p FULL OUTER JOIN s USING (v)
            """).fetchone()[0]
            print(f"  {col:<15} {tvd:.5f}")

        print("\n  Quantiles (population vs sample):")
        for col in ["trip_duration", "haversine_distance_km"]:
            q = "[0.05, 0.25, 0.5, 0.75, 0.95]"
            qp = con.execute(f"SELECT quantile_cont({col}, {q}) FROM {pop}").fetchone()[0]
            qs = con.execute(f"SELECT quantile_cont({col}, {q}) FROM {smp}").fetchone()[0]
            print(f"  {col:<22} pop {[round(x, 2) for x in qp]}")
            print(f"  {'':<22} smp {[round(x, 2) for x in qs]}")
    finally:
        con.close()

def create_sampled_datasets() -> None:
    city_names = ["chicago", "nyc"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Creating sampled datasets")
    print(f"{'=' * 60}")

    for city_name in city_names:
        print_city_strata_report(city_name, TARGET_ROWS_PER_CITY)

    for city_name in city_names:
        out = PROCESSED_DIR / f"{city_name}_sampled_run1.parquet"
        print(f"\nSampling {city_name}...")
        _sample_city(get_filtered_path(city_name), TARGET_ROWS_PER_CITY, out)
        print(f"  Saved to: {out}")
        representativeness_report(city_name, out)

    # Το μικτό σύνολο του σεναρίου C (N/2 + N/2) δημιουργείται μετά το split
    # ανά ημέρες, από τα train κάθε πόλης, ώστε να έχει το ίδιο N με τα A και B.


if __name__ == "__main__":
    create_sampled_datasets()
