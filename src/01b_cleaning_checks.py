"""
01b_cleaning_checks.py

Έλεγχοι στα ΑΡΧΙΚΑ δεδομένα 2025 (Chicago, NYC), που τεκμηριώνουν τους κανόνες
καθαρισμού του 01 (§2.2). Δεν αλλάζει κανένα αρχείο δεδομένων.

  2. Διαδρομές χωρίς σταθμό αφετηρίας ή προορισμού, ανά rideable_type,
     και ποσοστό e-bikes πριν και μετά τον κανόνα «μόνο σταθμός σε σταθμό».
  3. Εκατοστημόρια συντεταγμένων (διαδρομές με σταθμό) -> όρια CITY_BBOX.
  4. Διαδρομές με σταθμό αλλά συντεταγμένες εκτός CITY_BBOX.
  5. Σταθμοί «Public Rack»: πλήθος IDs, διαδρομές, ποσοστό e-bikes.

Είσοδος : data/raw/{city}/*.parquet
Έξοδος  : artifacts/logs/01_cleaning_checks.txt  (και στην οθόνη)
"""

from pathlib import Path

import duckdb
import pandas as pd

from config import RAW_DIR, CITY_BBOX

CITIES = ["chicago", "nyc"]
COORDS = ["start_lat", "start_lng", "end_lat", "end_lng"]
LOG = Path("artifacts/logs/01_cleaning_checks.txt")

HAS_STATION = "sid IS NOT NULL AND sid <> '' AND eid IS NOT NULL AND eid <> ''"


def raw(city: str) -> str:
    """Αρχικά δεδομένα με την ίδια μετατροπή τύπων με το 01."""
    path = (RAW_DIR / city).as_posix().replace("'", "''")
    return f"""(
        SELECT CAST(rideable_type AS VARCHAR)            AS rideable_type,
               TRIM(CAST(start_station_id AS VARCHAR))   AS sid,
               TRIM(CAST(end_station_id AS VARCHAR))     AS eid,
               CAST(start_station_name AS VARCHAR)       AS sn,
               TRY_CAST(start_lat AS DOUBLE) AS start_lat,
               TRY_CAST(start_lng AS DOUBLE) AS start_lng,
               TRY_CAST(end_lat AS DOUBLE)   AS end_lat,
               TRY_CAST(end_lng AS DOUBLE)   AS end_lng
        FROM read_parquet('{path}/*.parquet', union_by_name=true)
    )"""


def checks(city: str) -> list[tuple[str, str]]:
    src = raw(city)
    lat0, lat1, lng0, lng1 = CITY_BBOX[city]
    in_bbox = (f"start_lat BETWEEN {lat0} AND {lat1} AND start_lng BETWEEN {lng0} AND {lng1} "
               f"AND end_lat BETWEEN {lat0} AND {lat1} AND end_lng BETWEEN {lng0} AND {lng1}")
    no_st = f"NOT ({HAS_STATION})"

    q2 = f"""
        SELECT rideable_type,
               COUNT(*)                              AS trips,
               COUNT(*) FILTER (WHERE {no_st})       AS no_station,
               ROUND(100.0 * COUNT(*) FILTER (WHERE {no_st}) / COUNT(*), 2) AS pct_no_station
        FROM {src} GROUP BY rideable_type
        UNION ALL
        SELECT 'TOTAL', COUNT(*), COUNT(*) FILTER (WHERE {no_st}),
               ROUND(100.0 * COUNT(*) FILTER (WHERE {no_st}) / COUNT(*), 2)
        FROM {src}
        ORDER BY rideable_type"""

    q2b = f"""
        SELECT ROUND(100.0 * AVG((rideable_type = 'electric_bike')::INT), 2) AS pct_ebike_all,
               ROUND(100.0 * AVG((rideable_type = 'electric_bike')::INT)
                     FILTER (WHERE {HAS_STATION}), 2)                        AS pct_ebike_with_station
        FROM {src}"""

    q3 = " UNION ALL ".join(f"""
        SELECT '{c}' AS coord,
               MIN({c})                   AS min,
               quantile_cont({c}, 0.0001) AS q0_01pct,
               quantile_cont({c}, 0.001)  AS q0_1pct,
               quantile_cont({c}, 0.999)  AS q99_9pct,
               quantile_cont({c}, 0.9999) AS q99_99pct,
               MAX({c})                   AS max
        FROM {src} WHERE {HAS_STATION}""" for c in COORDS)

    q4 = f"""
        SELECT COUNT(*)                                  AS trips_with_station,
               COUNT(*) FILTER (WHERE NOT ({in_bbox}))   AS outside_bbox,
               ROUND(100.0 * COUNT(*) FILTER (WHERE NOT ({in_bbox})) / COUNT(*), 5) AS pct_outside
        FROM {src} WHERE {HAS_STATION}"""

    q5 = f"""
        SELECT CASE WHEN sn ILIKE '%public rack%' THEN 'public_rack' ELSE 'station' END AS kind,
               COUNT(DISTINCT sid) AS n_ids,
               COUNT(*)            AS start_trips,
               ROUND(100.0 * AVG((rideable_type = 'electric_bike')::INT), 2) AS pct_ebike
        FROM {src} WHERE {HAS_STATION}
        GROUP BY kind ORDER BY kind"""

    return [
        ("2. Διαδρομές χωρίς σταθμό ανά rideable_type", q2),
        ("2β. Ποσοστό e-bikes πριν και μετά τον κανόνα σταθμών", q2b),
        ("3. Εκατοστημόρια συντεταγμένων (διαδρομές με σταθμό)", q3),
        (f"4. Διαδρομές με σταθμό εκτός CITY_BBOX {CITY_BBOX[city]}", q4),
        ("5. Public racks (όνομα σταθμού αφετηρίας ILIKE '%public rack%')", q5),
    ]


def main() -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("# Έλεγχοι καθαρισμού του 01 στα αρχικά δεδομένα 2025\n", encoding="utf-8")
    con = duckdb.connect()
    for city in CITIES:
        for title, sql in checks(city):
            df = con.execute(sql).df()
            text = f"\n=== {city} | {title} ===\n{df.to_string(index=False)}\n"
            print(text)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(text)
    con.close()
    print(f"Saved to {LOG}")


if __name__ == "__main__":
    main()
