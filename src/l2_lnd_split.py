"""
l2_lnd_split.py

Διαχωρισμός του Λονδίνου ανά ημέρες και δειγματοληψία κάθε κομματιού.

Σειρά (το test ορίζεται πρώτο, πάνω σε όλο το καθαρισμένο Λονδίνο):
  1. Κάθε TEST_EVERY_NTH_DAY-οστή ημέρα του έτους -> ημέρες test.
     Οι υπόλοιπες -> ημέρες train/validation.
  2. Δειγματοληψία με τον μηχανισμό του 04 (ίδια στρώματα, hash, seed):
       ημέρες test            -> LONDON_TEST_ROWS
       ημέρες train/validation -> LONDON_TRAINVAL_ROWS
  3. Από το δείγμα train/validation, κάθε VAL_EVERY_NTH_DAY-οστή ημέρα
     (μετρώντας μόνο τις μη-test ημέρες) -> validation του oracle D.
  4. Κλείδωμα του test: λίστα ημερών και SHA-256 του αρχείου σε manifest.
     Αν το script ξανατρέξει και το test βγει διαφορετικό, σταματά.

Είσοδος : data/processed/london_filtered.parquet  (από το l1)
Έξοδοι  : data/processed/london_test.parquet      (κλειδωμένο)
          data/processed/london_train.parquet
          data/processed/london_val.parquet
          data/processed/london_split_manifest.json
"""

import hashlib
import importlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from config import (
    PROCESSED_DIR, TMP_DIR, STRATA_COLS,
    LONDON_TEST_ROWS, LONDON_TRAINVAL_ROWS,
    TEST_EVERY_NTH_DAY, VAL_EVERY_NTH_DAY,
)

# Ίδια συνάρτηση δειγματοληψίας με τις πόλεις-πηγές
sampling = importlib.import_module("04_stratified_sampling")

INPUT_PATH = PROCESSED_DIR / "london_filtered.parquet"
TEST_DAYS_PATH = TMP_DIR / "london_test_days.parquet"
TRAINVAL_DAYS_PATH = TMP_DIR / "london_trainval_days.parquet"
TRAINVAL_SAMPLE_PATH = TMP_DIR / "london_trainval_sampled.parquet"

TEST_PATH = PROCESSED_DIR / "london_test.parquet"
TRAIN_PATH = PROCESSED_DIR / "london_train.parquet"
VAL_PATH = PROCESSED_DIR / "london_val.parquet"
MANIFEST_PATH = PROCESSED_DIR / "london_split_manifest.json"

SORT_COLS = ["started_at", "start_station_id", "end_station_id", "trip_duration"]


def sql_string(path) -> str:
    return Path(path).as_posix().replace("'", "''")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tvd(a: pd.Series, b: pd.Series) -> float:
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


# ---------------------------------------------------------------------------
# Βήμα 1: ημέρες test και train/validation
# ---------------------------------------------------------------------------
def assign_days(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Πίνακας ημερών του έτους με τον ρόλο της καθεμίας.
    day_idx: 0 = 1 Ιανουαρίου. test: day_idx % N == N-1.
    validation: από τις μη-test ημέρες, με τη σειρά, κάθε M-οστή."""
    days = con.execute(f"""
        SELECT DISTINCT CAST(started_at AS DATE) AS day
        FROM read_parquet('{sql_string(INPUT_PATH)}')
        ORDER BY day
    """).df()
    days["day"] = pd.to_datetime(days["day"])
    days["day_idx"] = days["day"].dt.dayofyear - 1
    days["role"] = "trainval"
    days.loc[days["day_idx"] % TEST_EVERY_NTH_DAY == TEST_EVERY_NTH_DAY - 1, "role"] = "test"

    nontest = days[days["role"] != "test"].reset_index(drop=True)
    val_days = nontest.loc[nontest.index % VAL_EVERY_NTH_DAY == VAL_EVERY_NTH_DAY - 1, "day"]
    days.loc[days["day"].isin(val_days), "role"] = "val"
    return days


def write_day_split(con: duckdb.DuckDBPyConnection, days: pd.DataFrame) -> None:
    con.register("days", days[["day", "role"]])
    for role_sql, path in [("role = 'test'", TEST_DAYS_PATH),
                           ("role IN ('trainval', 'val')", TRAINVAL_DAYS_PATH)]:
        con.execute(f"""
            COPY (
                SELECT f.*
                FROM read_parquet('{sql_string(INPUT_PATH)}') f
                JOIN days d ON CAST(f.started_at AS DATE) = CAST(d.day AS DATE)
                WHERE d.{role_sql}
                ORDER BY {', '.join('f.' + c for c in SORT_COLS)}
            ) TO '{sql_string(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)


# ---------------------------------------------------------------------------
# Βήμα 3: validation από το δείγμα train/validation
# ---------------------------------------------------------------------------
def split_train_val(con: duckdb.DuckDBPyConnection, days: pd.DataFrame) -> None:
    con.register("val_days", days.loc[days["role"] == "val", ["day"]])
    src = f"read_parquet('{sql_string(TRAINVAL_SAMPLE_PATH)}')"
    for cond, path in [("NOT IN", TRAIN_PATH), ("IN", VAL_PATH)]:
        con.execute(f"""
            COPY (
                SELECT * FROM {src}
                WHERE CAST(started_at AS DATE) {cond} (SELECT CAST(day AS DATE) FROM val_days)
                ORDER BY {', '.join(SORT_COLS)}
            ) TO '{sql_string(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)


# ---------------------------------------------------------------------------
# Έλεγχοι
# ---------------------------------------------------------------------------
def representativeness(name: str, population: Path, sample: Path) -> None:
    cols = STRATA_COLS + ["hour", "day_of_week", "same_station"]
    pop = pd.read_parquet(population, columns=cols + ["trip_duration", "haversine_distance_km"])
    smp = pd.read_parquet(sample, columns=cols + ["trip_duration", "haversine_distance_km"])
    print(f"\n--- Representativeness: {name} ({len(smp):,} of {len(pop):,}) ---")
    for c in cols:
        print(f"  {c:<15} TVD {tvd(pop[c], smp[c]):.5f}")
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    for c in ["trip_duration", "haversine_distance_km"]:
        print(f"  {c:<22} pop {[round(x, 2) for x in pop[c].quantile(q)]}")
        print(f"  {'':<22} smp {[round(x, 2) for x in smp[c].quantile(q)]}")


def check_and_write_manifest(days: pd.DataFrame) -> None:
    test_hash = sha256(TEST_PATH)
    test_days = days.loc[days["role"] == "test", "day"].dt.strftime("%Y-%m-%d").tolist()
    val_days = days.loc[days["role"] == "val", "day"].dt.strftime("%Y-%m-%d").tolist()

    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if old["test_sha256"] != test_hash or old["test_days"] != test_days:
            raise RuntimeError(
                "Το London test διαφέρει από το κλειδωμένο (manifest). "
                "Αν η αλλαγή είναι σκόπιμη, σβήσε το manifest και ξανατρέξε, "
                "και ξανατρέξε όλα τα πειράματα."
            )
        print("\nTest unchanged: matches the locked manifest.")

    manifest = {
        "test_every_nth_day": TEST_EVERY_NTH_DAY,
        "val_every_nth_day": VAL_EVERY_NTH_DAY,
        "n_test_days": len(test_days),
        "n_val_days": len(val_days),
        "test_days": test_days,
        "val_days": val_days,
        "test_rows": int(pd.read_parquet(TEST_PATH, columns=["city"]).shape[0]),
        "train_rows": int(pd.read_parquet(TRAIN_PATH, columns=["city"]).shape[0]),
        "val_rows": int(pd.read_parquet(VAL_PATH, columns=["city"]).shape[0]),
        "test_sha256": test_hash,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written to {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε: {INPUT_PATH}. Τρέξε πρώτα το l1.")

    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory='{sql_string(TMP_DIR)}'")

        # 1. Ημέρες
        days = assign_days(con)
        print("Days per role:")
        print(days["role"].value_counts().reindex(["test", "trainval", "val"]).to_string())
        print("Test days per weekday (0 = Mon):")
        print(days.loc[days["role"] == "test", "day"].dt.dayofweek.value_counts().sort_index().to_string())
        write_day_split(con, days)

        n_test_days = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_string(TEST_DAYS_PATH)}')").fetchall()[0][0]
        n_trainval_days = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_string(TRAINVAL_DAYS_PATH)}')").fetchall()[0][0]
        print(f"\nTrips on test days:        {n_test_days:,} ({100 * n_test_days / (n_test_days + n_trainval_days):.1f}%)")
        print(f"Trips on train/val days:   {n_trainval_days:,}")

        # 2. Δειγματοληψία με τον μηχανισμό του 04
        print(f"\nSampling test days -> {LONDON_TEST_ROWS:,}")
        sampling._sample_city(TEST_DAYS_PATH, LONDON_TEST_ROWS, TEST_PATH)
        print(f"Sampling train/val days -> {LONDON_TRAINVAL_ROWS:,}")
        sampling._sample_city(TRAINVAL_DAYS_PATH, LONDON_TRAINVAL_ROWS, TRAINVAL_SAMPLE_PATH)

        # 3. Validation
        split_train_val(con, days)
    finally:
        con.close()

    for p, name in [(TEST_PATH, "test"), (TRAIN_PATH, "train"), (VAL_PATH, "val")]:
        n = pd.read_parquet(p, columns=["city"]).shape[0]
        print(f"{name:<6} {n:,} rows -> {p}")

    representativeness("test", TEST_DAYS_PATH, TEST_PATH)
    representativeness("train+val", TRAINVAL_DAYS_PATH, TRAINVAL_SAMPLE_PATH)

    # 4. Κλείδωμα
    check_and_write_manifest(days)

    for p in (TEST_DAYS_PATH, TRAINVAL_DAYS_PATH, TRAINVAL_SAMPLE_PATH):
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()