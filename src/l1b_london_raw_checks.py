"""
l1b_london_raw_checks.py

Έλεγχοι στα ΑΡΧΙΚΑ δεδομένα του Λονδίνου (TfL 2025), που τεκμηριώνουν τις
επιλογές του l1 (§2.2, §2.3). Δεν αλλάζει κανένα αρχείο δεδομένων.

  1α. Number: πλήθος γραμμών, μοναδικά Number, διπλές εγγραφές.
  1β. Bike model: τιμές και ποσοστά (-> BIKE_MODEL_MAP του l1).
  1γ. Μήκος κωδικού σταθμού (αρχής και τέλους), με παραδείγματα για τα ασυνήθιστα μήκη.
  1δ. Κενές τιμές ανά στήλη.
  1ε. Εύρος ημερομηνιών έναρξης και διάρκειας.

Είσοδος : data/raw/london/london.parquet
Έξοδος  : artifacts/logs/l1_london_raw_checks.txt  (και στην οθόνη)
"""

from pathlib import Path

import duckdb
import pandas as pd

from config import RAW_DIR

LONDON_PARQUET = RAW_DIR / "london" / "london.parquet"
LOG = Path("artifacts/logs/l1_london_raw_checks.txt")

COLS = ["Number", "Start date", "End date", "Start station number", "Start station",
        "End station number", "End station", "Bike model", "Total duration (ms)"]


def q(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def main() -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    src = "read_parquet('" + LONDON_PARQUET.as_posix().replace("'", "''") + "')"
    con = duckdb.connect()
    cols = [c for c in COLS if c in con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns]

    checks = [
        ("1α. Number: διπλές εγγραφές", f"""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT "Number") AS distinct_number,
                   COUNT(*) - COUNT(DISTINCT "Number") AS duplicates
            FROM {src}"""),
        ("1β. Bike model: τιμές", f"""
            SELECT CAST("Bike model" AS VARCHAR) AS bike_model, COUNT(*) AS trips,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM {src} GROUP BY 1 ORDER BY trips DESC"""),
    ]
    for side in ["Start", "End"]:
        c = q(f"{side} station number")
        checks.append((f"1γ. Μήκος κωδικού σταθμού ({side})", f"""
            SELECT LENGTH(TRIM(CAST({c} AS VARCHAR))) AS id_length,
                   COUNT(*) AS trips, COUNT(DISTINCT CAST({c} AS VARCHAR)) AS n_ids,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct,
                   list(DISTINCT CAST({c} AS VARCHAR))[1:5] AS examples
            FROM {src} GROUP BY 1 ORDER BY 1"""))
    checks += [
        ("1δ. Κενές τιμές ανά στήλη", " UNION ALL ".join(
            f"SELECT '{c}' AS column, COUNT(*) FILTER (WHERE {q(c)} IS NULL "
            f"OR TRIM(CAST({q(c)} AS VARCHAR)) = '') AS empty FROM {src}" for c in cols)),
        ("1ε. Εύρος ημερομηνιών και διάρκειας", f"""
            SELECT MIN(CAST("Start date" AS VARCHAR)) AS min_start,
                   MAX(CAST("Start date" AS VARCHAR)) AS max_start,
                   MIN(TRY_CAST("Total duration (ms)" AS BIGINT)) / 60000.0 AS min_duration_min,
                   quantile_cont(TRY_CAST("Total duration (ms)" AS BIGINT), 0.5) / 60000.0 AS median_duration_min,
                   MAX(TRY_CAST("Total duration (ms)" AS BIGINT)) / 60000.0 AS max_duration_min
            FROM {src}"""),
    ]

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("# Έλεγχοι στα αρχικά δεδομένα του Λονδίνου (TfL 2025)\n", encoding="utf-8")
    for title, sql in checks:
        df = con.execute(sql).df()
        text = f"\n=== london | {title} ===\n{df.to_string(index=False)}\n"
        print(text)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(text)
    con.close()
    print(f"Saved to {LOG}")


if __name__ == "__main__":
    main()
