"""
06_prepare_model_input.py

Τελικοί πίνακες εισόδου των μοντέλων για τα σενάρια A-D του Επιπέδου 1 και το
κλειδωμένο London test (§2.3, §2.4, §4).

Στήλες γένεσης (ίδιες σε όλα τα σενάρια και στο test):
  hour          κατηγορική 0-23    ώρα έναρξης
  day_of_week   κατηγορική 0-6     0 = Δευτέρα
  month         κατηγορική 1-12
  rideable_type κατηγορική         classic_bike / electric_bike
  origin_type   κατηγορική 0-5     τύπος σταθμού αφετηρίας (05b)
  round_trip    κατηγορική 0/1     ίδιος φυσικός σταθμός αρχής και τέλους
  duration_min  συνεχής            διάρκεια σε λεπτά
  distance_km   συνεχής            δικτυακή απόσταση σταθμών (05c)· 0 στις κυκλικές

Τα is_weekend και part_of_day ΔΕΝ παράγονται: υπολογίζονται από τα hour και
day_of_week μετά τη γένεση, ώστε να μην προκύπτουν αντιφάσεις. Η κωδικοποίηση
(one-hot, κανονικοποίηση) γίνεται μέσα σε κάθε μοντέλο, μόνο με τα δεδομένα
εκπαίδευσης του σεναρίου. Εδώ οι στήλες μένουν στις αρχικές τους μονάδες.

Μεγέθη (ίδια σε όλα τα σενάρια, ώστε να διαφέρει μόνο η πηγή των δεδομένων):
  N = γραμμές του london_train (720.809), V = γραμμές του london_val (79.191)
  A  Chicago        train N  από chicago_train,        val V από chicago_val
  B  NYC            train N  από nyc_train,            val V από nyc_val
  C  Chicago + NYC  train ⌈N/2⌉ + ⌊N/2⌋,                val ⌈V/2⌉ + ⌊V/2⌋
  D  London (oracle) train = london_train,             val = london_val
Η υποδειγματοληψία χρησιμοποιεί τη στρωματοποιημένη δειγματοληψία του 04 (ίδια
στρώματα, ίδιο hash με seed). Γι' αυτό το μισό του Chicago στο C είναι
(σχεδόν πλήρως) υποσύνολο του A, και το μισό της NYC στο C υποσύνολο του B.

Πριν από οτιδήποτε, ελέγχεται ότι το london_test ταιριάζει με το SHA-256 του manifest.

Είσοδοι : data/processed/{chicago,nyc}_{train,val}.parquet, london_{train,val,test}.parquet
          data/processed/london_split_manifest.json
          data/context/osm/od_network_{city}.parquet  (05c)
Έξοδοι  : data/model_input/{A,B,C,D}_{train,val}.parquet, data/model_input/london_test.parquet
          data/model_input/manifest.json              (μεγέθη και SHA-256 κάθε αρχείου)
          artifacts/model_input/schema.json           (τύποι και κατηγορίες στηλών)
          artifacts/model_input/gap_vs_london_test.csv
"""

import hashlib
import importlib
import json
from math import ceil, floor
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OSM_DIR, TMP_DIR

sampling = importlib.import_module("04_stratified_sampling")

MODEL_INPUT_DIR = Path("data/model_input")
ARTIFACT_DIR = Path("artifacts/model_input")
LONDON_MANIFEST = PROCESSED_DIR / "london_split_manifest.json"

CATEGORICAL = {
    "hour": list(range(24)),
    "day_of_week": list(range(7)),
    "month": list(range(1, 13)),
    "rideable_type": ["classic_bike", "electric_bike"],
    "origin_type": None,          # γεμίζει αυτόματα στο main() από το station_types.parquet (05b)
    "round_trip": [0, 1],
}
CONTINUOUS = ["duration_min", "distance_km"]
COLUMNS = list(CATEGORICAL) + CONTINUOUS

READ_COLS = ["hour", "day_of_week", "month", "rideable_type", "trip_duration",
             "same_station", "start_station_key", "end_station_key"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_locked_test() -> None:
    manifest = json.loads(LONDON_MANIFEST.read_text(encoding="utf-8"))
    if sha256(PROCESSED_DIR / "london_test.parquet") != manifest["test_sha256"]:
        raise RuntimeError("Το london_test.parquet δεν ταιριάζει με το manifest του l2. Σταματώ.")
    print("London test matches the locked manifest.")


# ---------------------------------------------------------------------------
def subsample(city: str, split: str, n: int) -> Path:
    """Στρωματοποιημένο υποδείγμα n γραμμών από το {city}_{split} (μηχανισμός του 04)."""
    src = PROCESSED_DIR / f"{city}_{split}.parquet"
    out = TMP_DIR / f"06_{city}_{split}_{n}.parquet"
    print(f"Subsampling {src.name} -> {n:,}")
    sampling._sample_city(src, n, out)
    return out


def to_model_table(path: Path, city: str, od: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Μετατροπή διαδρομών στις στήλες γένεσης."""
    df = pd.read_parquet(path, columns=READ_COLS)
    df = df.merge(od[city], on=["start_station_key", "end_station_key"], how="left",
                  validate="many_to_one")

    round_trip = df.start_station_key == df.end_station_key
    if not (round_trip == (df.same_station == 1)).all():
        raise ValueError(f"{city}: same_station differs from station-key equality")
    if df.start_type.isna().any():
        raise ValueError(f"{city}: {int(df.start_type.isna().sum())} trips without origin type")
    if df.loc[~round_trip, "network_distance_km"].isna().any():
        raise ValueError(f"{city}: non-round trips without network distance")

    out = pd.DataFrame({
        "hour": df.hour.astype("int8"),
        "day_of_week": df.day_of_week.astype("int8"),
        "month": df.month.astype("int8"),
        "rideable_type": df.rideable_type.astype(str),
        "origin_type": df.start_type.astype("int8"),
        "round_trip": round_trip.astype("int8"),
        "duration_min": df.trip_duration.astype("float64"),
        "distance_km": np.where(round_trip, 0.0, df.network_distance_km).astype("float64"),
    })
    unknown = set(out.rideable_type) - set(CATEGORICAL["rideable_type"])
    if unknown:
        raise ValueError(f"{city}: unexpected rideable_type values {unknown}")
    return out


# ---------------------------------------------------------------------------
def gap_table(tables: dict[str, pd.DataFrame], test: pd.DataFrame) -> pd.DataFrame:
    """Απόσταση κάθε συνόλου εκπαίδευσης από το London test, πριν από κάθε μοντέλο:
    TVD για τις κατηγορικές, διάμεσος και Wasserstein-1 για τις συνεχείς."""
    from scipy.stats import wasserstein_distance

    rows = []
    for name, df in tables.items():
        r = {"train_set": name, "rows": len(df)}
        for c in CATEGORICAL:
            p = df[c].value_counts(normalize=True)
            q = test[c].value_counts(normalize=True)
            r[f"TVD_{c}"] = round(0.5 * p.subtract(q, fill_value=0).abs().sum(), 4)
        for c in CONTINUOUS:
            r[f"median_{c}"] = round(float(df[c].median()), 3)
            r[f"W1_{c}"] = round(float(wasserstein_distance(df[c], test[c])), 4)
        rows.append(r)
    r = {"train_set": "london_test", "rows": len(test)}
    r.update({f"median_{c}": round(float(test[c].median()), 3) for c in CONTINUOUS})
    rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    check_locked_test()
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    od = {}
    for city in ["chicago", "nyc", "london"]:
        o = pd.read_parquet(OSM_DIR / f"od_network_{city}.parquet",
                            columns=["start_station_key", "end_station_key",
                                     "network_distance_km", "start_type"])
        od[city] = o

    n_train = len(pd.read_parquet(PROCESSED_DIR / "london_train.parquet", columns=["hour"]))
    n_val = len(pd.read_parquet(PROCESSED_DIR / "london_val.parquet", columns=["hour"]))
    print(f"N (train) = {n_train:,}   V (val) = {n_val:,}")

    sources = {   # σενάριο -> split -> [(πόλη, πλήθος)]
        "A": {"train": [("chicago", n_train)], "val": [("chicago", n_val)]},
        "B": {"train": [("nyc", n_train)], "val": [("nyc", n_val)]},
        "C": {"train": [("chicago", ceil(n_train / 2)), ("nyc", floor(n_train / 2))],
              "val": [("chicago", ceil(n_val / 2)), ("nyc", floor(n_val / 2))]},
        "D": {"train": [("london", None)], "val": [("london", None)]},
    }

    written, train_tables = {}, {}
    for exp, splits in sources.items():
        for split, parts in splits.items():
            pieces = []
            for city, n in parts:
                path = (PROCESSED_DIR / f"{city}_{split}.parquet" if n is None
                        else subsample(city, split, n))
                pieces.append(to_model_table(path, city, od))
            df = pd.concat(pieces, ignore_index=True)
            # σταθερή σειρά γραμμών, ανεξάρτητη από τη σειρά επεξεργασίας
            df = df.sort_values(COLUMNS, kind="mergesort").reset_index(drop=True)
            out = MODEL_INPUT_DIR / f"{exp}_{split}.parquet"
            df.to_parquet(out, index=False)
            written[out.name] = df
            if split == "train":
                train_tables[exp] = df
            print(f"{exp}_{split:<5} {len(df):>9,} rows -> {out}")

    test = to_model_table(PROCESSED_DIR / "london_test.parquet", "london", od)
    test = test.sort_values(COLUMNS, kind="mergesort").reset_index(drop=True)
    test_out = MODEL_INPUT_DIR / "london_test.parquet"
    test.to_parquet(test_out, index=False)
    written[test_out.name] = test
    print(f"london_test {len(test):>9,} rows -> {test_out}")

    # Σχήμα: οι κατηγορίες του origin_type είναι όλοι οι τύποι που όρισε το 05b
    types = sorted(int(t) for t in
                   pd.read_parquet(OSM_DIR / "station_types.parquet", columns=["origin_type"])
                   .origin_type.unique())
    schema = {
        "categorical": {c: (types if v is None else v) for c, v in CATEGORICAL.items()},
        "continuous": CONTINUOUS,
        "derived_after_generation": {
            "is_weekend": "day_of_week in (5, 6)",
            "part_of_day": "από το hour, με τα όρια του 01",
            "distance_km": "0 όταν round_trip = 1",
        },
        "N_train": n_train, "V_val": n_val,
    }
    (ARTIFACT_DIR / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                                              encoding="utf-8")

    manifest = {name: {"rows": len(df), "sha256": sha256(MODEL_INPUT_DIR / name)}
                for name, df in written.items()}
    (MODEL_INPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Περιγραφή κάθε συνόλου
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print("\nShare of rows per category (train sets and London test):")
    for c in ["rideable_type", "round_trip", "origin_type"]:
        t = pd.DataFrame({k: v[c].value_counts(normalize=True) for k, v in
                          {**train_tables, "london_test": test}.items()}).mul(100).round(1)
        print(f"\n{c} (%)\n{t.to_string()}")

    gap = gap_table(train_tables, test)
    gap.to_csv(ARTIFACT_DIR / "gap_vs_london_test.csv", index=False)
    print("\nDistance of each training set to the London test, before any model:")
    print(gap.T.to_string(header=False))
    print(f"\nSaved schema, manifest and gap table ({ARTIFACT_DIR}, {MODEL_INPUT_DIR})")


if __name__ == "__main__":
    main()
