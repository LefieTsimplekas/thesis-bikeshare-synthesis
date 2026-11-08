"""
04b_source_val_split.py

Διαχωρισμός των πόλεων-πηγών σε train και validation, ανά ημέρες.

Κανόνας (ίδια λογική με το Λονδίνο): κάθε VAL_EVERY_NTH_DAY-οστή ημέρα του έτους
-> validation. Εδώ δεν υπάρχει test, οπότε μετράμε σε όλες τις ημέρες.
day_idx: 0 = 1 Ιανουαρίου. validation: day_idx % N == N-1.

Το validation χρησιμοποιείται μόνο για το tuning των σεναρίων A, B, C (§4).
Το μικτό σύνολο του C (N/2 + N/2) φτιάχνεται αργότερα, στο 06, από αυτά τα αρχεία.

Είσοδος : data/processed/{city}_sampled.parquet   (από το 04)
Έξοδοι  : data/processed/{city}_train.parquet
          data/processed/{city}_val.parquet
          data/processed/{city}_split_manifest.json
"""

import json
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, STRATA_COLS, VAL_EVERY_NTH_DAY

CITIES = ["chicago", "nyc"]
SORT_COLS = ["started_at", "start_station_id", "end_station_id", "trip_duration"]


def tvd(a: pd.Series, b: pd.Series) -> float:
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


def split_city(city: str) -> None:
    in_path = PROCESSED_DIR / f"{city}_sampled.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε: {in_path}. Τρέξε πρώτα το 04.")

    df = pd.read_parquet(in_path)
    day_idx = df["started_at"].dt.dayofyear - 1
    is_val = (day_idx % VAL_EVERY_NTH_DAY) == (VAL_EVERY_NTH_DAY - 1)

    train = df[~is_val].sort_values(SORT_COLS).reset_index(drop=True)
    val = df[is_val].sort_values(SORT_COLS).reset_index(drop=True)

    train.to_parquet(PROCESSED_DIR / f"{city}_train.parquet", index=False, compression="zstd")
    val.to_parquet(PROCESSED_DIR / f"{city}_val.parquet", index=False, compression="zstd")

    val_days = sorted(val["started_at"].dt.strftime("%Y-%m-%d").unique().tolist())

    print(f"\n===== {city} =====")
    print(f"train {len(train):,}  val {len(val):,}  ({100 * len(val) / len(df):.1f}% validation)")
    print(f"validation days: {len(val_days)}")
    print("validation days per weekday (0 = Mon):")
    print(pd.to_datetime(pd.Series(val_days)).dt.dayofweek.value_counts().sort_index().to_string())
    print("TVD validation vs whole sample:")
    for c in STRATA_COLS + ["hour", "same_station"]:
        print(f"  {c:<15} {tvd(df[c], val[c]):.5f}")

    manifest = {
        "city": city,
        "val_every_nth_day": VAL_EVERY_NTH_DAY,
        "n_val_days": len(val_days),
        "val_days": val_days,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
    }
    (PROCESSED_DIR / f"{city}_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    for city in CITIES:
        split_city(city)
