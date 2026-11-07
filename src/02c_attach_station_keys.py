"""
02c_attach_station_keys.py

Συνδέει κάθε διαδρομή με τον φυσικό σταθμό αφετηρίας και προορισμού
(station_key από το 02b) και ορίζει τις κυκλικές διαδρομές με βάση τον
φυσικό σταθμό.

Γιατί χρειάζεται: ο ίδιος φυσικός σταθμός μπορεί να έχει περισσότερα IDs
(αλλαγή αρίθμησης Divvy, ασυνεπής μορφοποίηση Citi Bike). Με σύγκριση IDs,
μια διαδρομή από το παλιό στο νέο ID του ίδιου σταθμού δεν μετρά ως κυκλική
και αφαιρείται λανθασμένα από τον κανόνα ελάχιστης απόστασης του 03.

Είσοδοι : data/processed/{city}_featured.parquet
          data/processed/station_map_{city}.parquet   (από το 02b)
Έξοδος  : data/processed/{city}_keyed.parquet

Αλλαγές σε σχέση με το featured:
  + start_station_key, end_station_key
  ~ same_station: πλέον 1 όταν start_station_key = end_station_key
    (το όνομα κρατιέται ώστε το 03 να δουλεύει χωρίς αλλαγή)
"""

from pathlib import Path
import duckdb
from config import PROCESSED_DIR, TMP_DIR

CITIES = ["chicago", "nyc"]


def sql_string(path) -> str:
    return Path(path).as_posix().replace("'", "''")


def attach_keys(city: str) -> None:
    featured = PROCESSED_DIR / f"{city}_featured.parquet"
    station_map = PROCESSED_DIR / f"station_map_{city}.parquet"
    out_path = PROCESSED_DIR / f"{city}_keyed.parquet"
    for p in (featured, station_map):
        if not p.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε: {p}")

    f_src = f"read_parquet('{sql_string(featured)}')"
    m_src = f"read_parquet('{sql_string(station_map)}')"

    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory='{sql_string(TMP_DIR)}'")

        # Το station_map είναι μικρός πίνακας: φορτώνεται μία φορά στη μνήμη
        con.execute(f"CREATE TEMP TABLE m AS SELECT station_id, station_key FROM {m_src}")

        # LEFT JOIN: καμία διαδρομή δεν χάνεται σιωπηλά. Αν λείπει ID από το
        # station_map, το key βγαίνει NULL και ο έλεγχος μετά το γράψιμο το πιάνει.
        con.execute(f"""
        COPY (
            SELECT f.* REPLACE (
                       CASE WHEN ms.station_key = me.station_key THEN 1 ELSE 0 END AS same_station
                   ),
                   ms.station_key AS start_station_key,
                   me.station_key AS end_station_key
            FROM {f_src} f
            LEFT JOIN m ms ON f.start_station_id = ms.station_id
            LEFT JOIN m me ON f.end_station_id   = me.station_id
            ORDER BY started_at, start_station_id, end_station_id, ended_at
        ) TO '{sql_string(out_path)}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)

        # Έλεγχοι στο αρχείο εξόδου (το COUNT(*) σε parquet διαβάζεται από τα metadata)
        n_in = con.execute(f"SELECT COUNT(*) FROM {f_src}").fetchone()[0]
        n_out, n_round, n_missing = con.execute(f"""
            SELECT COUNT(*),
                   SUM(same_station),
                   COUNT(*) FILTER (WHERE start_station_key IS NULL OR end_station_key IS NULL)
            FROM read_parquet('{sql_string(out_path)}')
        """).fetchone()
        if n_in != n_out:
            raise ValueError(f"{city}: γραμμές {n_in:,} -> {n_out:,}.")
        if n_missing:
            out_path.unlink(missing_ok=True)
            raise ValueError(f"{city}: {n_missing:,} διαδρομές με ID που λείπει από το "
                             f"station_map. Ξανατρέξε το 02b.")

        print(f"{city}: {n_out:,} διαδρομές, κυκλικές (φυσικός σταθμός): {n_round:,} "
              f"({100 * n_round / n_out:.2f}%)")
        print(f"Saved to {out_path}")
    finally:
        con.close()


if __name__ == "__main__":
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        attach_keys(city)