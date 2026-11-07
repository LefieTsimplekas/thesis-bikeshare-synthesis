"""
02b_station_table.py

Πίνακας σταθμών ανά πόλη (Chicago, NYC): μία γραμμή ανά ΦΥΣΙΚΟ σταθμό.

Γιατί χρειάζεται:
  - Τα προφίλ OSM και το origin_type (§3) ορίζονται ανά σταθμό.
  - Η δικτυακή απόσταση υπολογίζεται μία φορά ανά ζεύγος σταθμών (§3.4).
  - Η απόσταση υπολογίζεται από τη θέση των σταθμών σε όλες τις πόλεις,
    όπως στο Λονδίνο (BikePoint).

Βήματα:
  1. Θέση ανά ID σταθμού: διάμεσος των συντεταγμένων αφετηρίας και προορισμού.
     Αν ένα ID έχει τουλάχιστον MIN_CLASSIC καταγραφές classic_bike, η θέση
     υπολογίζεται μόνο από αυτές (στη Divvy ένα μικρό ποσοστό καταγραφών e-bike
     φέρει GPS αντί για τις συντεταγμένες του σταθμού).
  2. Ομαδοποίηση IDs σε φυσικούς σταθμούς: IDs με θέσεις σε απόσταση έως
     MERGE_RADIUS_M ενώνονται (συνεκτικές συνιστώσες). Χρειάζεται επειδή η Divvy
     άλλαξε σύστημα αναγνωριστικών μέσα στο 2025, οπότε ο ίδιος σταθμός
     εμφανίζεται με περισσότερα IDs. Ο ίδιος κανόνας εφαρμόζεται σε όλες τις πόλεις.
  3. Κλειδί και θέση φυσικού σταθμού: το ID με τις περισσότερες καταγραφές
     στην ομάδα, και η θέση του.

Είσοδος : data/processed/{city}_featured.parquet  (ΠΛΗΡΕΣ έτος, πριν το sampling)
Έξοδοι  :
  data/processed/station_ids_{city}.parquet  : μία γραμμή ανά ID (ενδιάμεσο, για έλεγχο)
  data/processed/station_map_{city}.parquet  : station_id -> station_key
  data/processed/stations_{city}.parquet     : μία γραμμή ανά φυσικό σταθμό

Στήλες stations_{city} (ίδια μορφή και για το Λονδίνο):
  station_key : κλειδί φυσικού σταθμού
  lat, lng    : θέση
  n_trips     : πλήθος αφετηριών + προορισμών (όλων των IDs της ομάδας)
  n_ids       : πλήθος IDs που ενώθηκαν
  member_ids  : τα IDs της ομάδας, χωρισμένα με ' | '
  source      : 'trips'
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from config import PROCESSED_DIR, TMP_DIR, MERGE_RADIUS_M
from config import MIN_CLASSIC_FOR_POSITION as MIN_CLASSIC


CITIES = ["chicago", "nyc"]
MIN_TRIPS_FLAG = 10     # όριο για σταθμούς με πολύ λίγες καταγραφές


def sql_string(path) -> str:
    return Path(path).as_posix().replace("'", "''")


def haversine_sql(lat1: str, lng1: str, lat2: str, lng2: str) -> str:
    return f"""6371000 * 2 * ASIN(SQRT(
        POWER(SIN(RADIANS({lat2} - {lat1}) / 2), 2)
      + COS(RADIANS({lat1})) * COS(RADIANS({lat2}))
      * POWER(SIN(RADIANS({lng2} - {lng1}) / 2), 2)))"""


def haversine_m(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Βήμα 1: θέση ανά ID
# ---------------------------------------------------------------------------
def build_id_table(con: duckdb.DuckDBPyConnection, city: str) -> pd.DataFrame:
    in_path = PROCESSED_DIR / f"{city}_featured.parquet"
    out_path = PROCESSED_DIR / f"station_ids_{city}.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε: {in_path}. Τρέξε πρώτα το 01.")

    src = f"read_parquet('{sql_string(in_path)}')"

    cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()}
    missing = {"start_station_id", "end_station_id"} - cols
    if missing:
        raise ValueError(f"{in_path} δεν έχει {missing}. Ξανατρέξε το 01 με τη διόρθωση.")

    con.execute(f"""
    COPY (
        WITH ends AS (
            SELECT start_station_id AS station_id, start_lat AS lat, start_lng AS lng,
                   rideable_type
            FROM {src}
            UNION ALL
            SELECT end_station_id, end_lat, end_lng, rideable_type
            FROM {src}
        ),
        counts AS (
            SELECT station_id,
                   COUNT(*) AS n_trips,
                   SUM(CASE WHEN rideable_type = 'classic_bike' THEN 1 ELSE 0 END) AS n_classic
            FROM ends
            GROUP BY station_id
        ),
        pos_records AS (
            SELECT e.station_id, e.lat, e.lng
            FROM ends e
            JOIN counts c USING (station_id)
            WHERE c.n_classic < {MIN_CLASSIC}
               OR e.rideable_type = 'classic_bike'
        ),
        pos AS (
            SELECT station_id,
                   median(lat) AS lat,
                   median(lng) AS lng
            FROM pos_records
            GROUP BY station_id
        )
        SELECT p.station_id,
               p.lat,
               p.lng,
               c.n_trips,
               CASE WHEN c.n_classic >= {MIN_CLASSIC} THEN 'classic' ELSE 'all' END AS pos_source
        FROM pos p
        JOIN counts c USING (station_id)
        ORDER BY p.station_id
    ) TO '{sql_string(out_path)}' (FORMAT PARQUET);
    """)

    return con.sql(f"SELECT * FROM read_parquet('{sql_string(out_path)}')").df()


# ---------------------------------------------------------------------------
# Βήμα 2: ομαδοποίηση IDs σε φυσικούς σταθμούς
# ---------------------------------------------------------------------------
def find_close_pairs(con: duckdb.DuckDBPyConnection, ids: pd.DataFrame) -> pd.DataFrame:
    """Ζεύγη IDs με θέσεις σε απόσταση έως MERGE_RADIUS_M."""
    con.register("ids", ids)
    # Πρώτο φίλτρο με κουτί ~0.001° (~80–110 m) για ταχύτητα, μετά ακριβής απόσταση
    return con.sql(f"""
        SELECT a.station_id AS id_a,
               b.station_id AS id_b,
               {haversine_sql('a.lat', 'a.lng', 'b.lat', 'b.lng')} AS dist_m
        FROM ids a
        JOIN ids b
          ON a.station_id < b.station_id
         AND ABS(a.lat - b.lat) < 0.001
         AND ABS(a.lng - b.lng) < 0.001
        WHERE {haversine_sql('a.lat', 'a.lng', 'b.lat', 'b.lng')} <= {MERGE_RADIUS_M}
    """).df()


def connected_components(nodes: list[str], pairs: pd.DataFrame) -> dict[str, int]:
    """Union-find: κάθε ID -> αριθμός ομάδας."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["id_a"], pairs["id_b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    roots = {n: find(n) for n in nodes}
    root_to_group = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {n: root_to_group[r] for n, r in roots.items()}


def build_station_tables(con: duckdb.DuckDBPyConnection, city: str, ids: pd.DataFrame) -> None:
    pairs = find_close_pairs(con, ids)
    group_of = connected_components(ids["station_id"].tolist(), pairs)
    ids = ids.assign(group=ids["station_id"].map(group_of))

    # Βήμα 3: κλειδί = ID με τις περισσότερες καταγραφές (ισοπαλία: αλφαβητικά)
    key_rows = (
        ids.sort_values(["group", "n_trips", "station_id"], ascending=[True, False, True])
           .drop_duplicates("group")
           .set_index("group")
    )
    ids["station_key"] = ids["group"].map(key_rows["station_id"])

    # Πίνακας αντιστοίχισης ID -> φυσικός σταθμός, με απόσταση από τη θέση του κλειδιού
    key_lat = ids["group"].map(key_rows["lat"])
    key_lng = ids["group"].map(key_rows["lng"])
    ids["dist_to_key_m"] = haversine_m(ids["lat"], ids["lng"], key_lat, key_lng)

    station_map = ids[["station_id", "station_key", "dist_to_key_m"]].sort_values("station_id")
    station_map.to_parquet(PROCESSED_DIR / f"station_map_{city}.parquet", index=False)

    stations = (
        ids.groupby("station_key")
           .agg(n_trips=("n_trips", "sum"),
                n_ids=("station_id", "size"),
                member_ids=("station_id", lambda s: " | ".join(sorted(s))))
           .join(key_rows.set_index("station_id")[["lat", "lng"]])
           .reset_index()
           [["station_key", "lat", "lng", "n_trips", "n_ids", "member_ids"]]
           .assign(source="trips")
           .sort_values("station_key")
    )
    stations.to_parquet(PROCESSED_DIR / f"stations_{city}.parquet", index=False)

    # ---- Σύνοψη ----
    print(f"IDs σταθμών                 : {len(ids):,}")
    print(f"Φυσικοί σταθμοί             : {len(stations):,}")
    print(f"Σταθμοί με >1 ID            : {(stations['n_ids'] > 1).sum():,}")
    print(f"Μέγιστο πλήθος IDs ανά σταθμό: {stations['n_ids'].max()}")
    print(f"Μέγιστη απόσταση ID-κλειδιού : {ids['dist_to_key_m'].max():.1f} m")
    print(f"Σταθμοί με < {MIN_TRIPS_FLAG} καταγραφές : {(stations['n_trips'] < MIN_TRIPS_FLAG).sum():,}")

    print("\nΔείγμα σταθμών με τα περισσότερα IDs:")
    print(stations.sort_values(["n_ids", "n_trips"], ascending=False)
                  .head(10)[["station_key", "n_ids", "n_trips", "member_ids"]]
                  .to_string(index=False))


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        print(f"\n===== {city} =====")
        con = duckdb.connect()
        try:
            con.execute(f"SET temp_directory='{sql_string(TMP_DIR)}'")
            ids = build_id_table(con, city)
            build_station_tables(con, city, ids)
        finally:
            con.close()
        print(f"Saved station_ids_{city}, station_map_{city}, stations_{city}")


if __name__ == "__main__":
    main()