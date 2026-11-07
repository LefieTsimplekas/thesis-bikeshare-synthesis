from pathlib import Path
import numpy as np
import pandas as pd
import requests
from config import (RAW_DIR, PROCESSED_DIR, CONTEXT_DIR, MERGE_RADIUS_M,MIN_TRIP_DURATION_MIN, MAX_TRIP_DURATION_MIN, MIN_MOVING_DISTANCE_KM, MAX_TRIP_DISTANCE_KM, MAX_SPEED_KMH,LONDON_INVALID_STATION_IDS,)


LONDON_PARQUET = RAW_DIR / "london" / "london.parquet"
STATIONS_PARQUET = CONTEXT_DIR / "london_stations.parquet"      # αντίγραφο BikePoint
OUTPUT_PATH = PROCESSED_DIR / "london_filtered.parquet"
STATION_TABLE_PATH = PROCESSED_DIR / "stations_london.parquet"
STATION_MAP_PATH = PROCESSED_DIR / "station_map_london.parquet"

TFL_BIKEPOINT_URL = "https://api.tfl.gov.uk/BikePoint"

DATE_FORMAT = "%Y-%m-%d %H:%M"  # επιβεβαιώθηκε: 0 αδιάβαστες στο σύνολο.
BIKE_MODEL_MAP = {"CLASSIC": "classic_bike", "PBSC_EBIKE": "electric_bike"}  # ΝΑ ΕΠΙΒΕΒΑΙΩΘΕΙ από τον 1γ


def load_or_fetch_stations() -> pd.DataFrame:
    if STATIONS_PARQUET.exists():
        print(f"Using cached stations from {STATIONS_PARQUET}")
        df = pd.read_parquet(STATIONS_PARQUET)
        df["station_id"] = df["station_id"].astype(str).str.strip()
        return df

    print("Fetching TfL BikePoint station coordinates...")
    resp = requests.get(TFL_BIKEPOINT_URL, timeout=30)
    resp.raise_for_status()

    # TerminalName is the zero-padded station number (e.g. "001043") that
    # matches the station numbers in the trip data directly.
    stations = []
    for p in resp.json():
        terminal = next(
            (prop["value"] for prop in p.get("additionalProperties", []) if prop["key"] == "TerminalName"),
            None,
        )
        if terminal:
            stations.append({
                "station_id": terminal,
                "station_name": p["commonName"],
                "lat": p["lat"],
                "lng": p["lon"],
            })

    df = pd.DataFrame(stations)
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["fetched_at"] = pd.Timestamp.now(tz="UTC")   # πότε ανακτήθηκε (για τη μεθοδολογία)

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STATIONS_PARQUET, index=False)
    print(f"Saved {len(df)} stations to {STATIONS_PARQUET}")
    return df


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lng1, lat2, lng2):
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlng = np.radians(lng2 - lng1)
    x = np.sin(dlng) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlng)
    deg = np.degrees(np.arctan2(x, y))
    return np.where(deg < 0, deg + 360, deg)

def tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance δύο κατηγορικών κατανομών."""
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()

def group_stations(stations: pd.DataFrame, n_trips: pd.Series) -> pd.DataFrame:
    """Ίδιος κανόνας με το 02b: IDs σε απόσταση έως MERGE_RADIUS_M = ένας φυσικός σταθμός.
    Κλειδί = το ID με τις περισσότερες διαδρομές στην ομάδα."""
    s = stations.reset_index()[["station_id", "lat", "lng"]].copy()
    s["n_trips"] = s["station_id"].map(n_trips).fillna(0).astype(int)

    lat, lng = s["lat"].to_numpy(), s["lng"].to_numpy()
    dist_m = haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :]) * 1000
    i_idx, j_idx = np.where(np.triu(dist_m <= MERGE_RADIUS_M, k=1))

    parent = list(range(len(s)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in zip(i_idx, j_idx):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    s["group"] = [find(i) for i in range(len(s))]

    key = (s.sort_values(["group", "n_trips", "station_id"], ascending=[True, False, True])
             .drop_duplicates("group").set_index("group")["station_id"])
    s["station_key"] = s["group"].map(key)
    return s

def resolve_old_ids(df: pd.DataFrame, stations: pd.DataFrame) -> dict:
    """IDs της μορφής <βασικό ID>444 ή <βασικό ID>4444 με όνομα '<όνομα σταθμού>_OLD'
    είναι παλαιότερη εκδοχή του ίδιου σταθμού: αντιστοιχίζονται στο βασικό ID."""
    known = set(stations.index)
    names = stations["station_name"]
    mapping = {}
    for id_col, name_col in [("start_station_id", "start_station_name"),
                             ("end_station_id", "end_station_name")]:
        cand = (df.loc[~df[id_col].isin(known) & df[name_col].str.endswith("_OLD", na=False),
                       [id_col, name_col]]
                  .drop_duplicates())
        for sid, name in cand.itertuples(index=False):
            for suffix in ("4444", "444"):
                base = sid[:-len(suffix)]
                if sid.endswith(suffix) and base in known and names[base] == name[:-len("_OLD")]:
                    mapping[sid] = base
                    break
    return mapping

def build_london_featured_parquet() -> pd.DataFrame:
    stations = load_or_fetch_stations().set_index("station_id")
    steps = {}  # διαδοχικά πλήθη για τον πίνακα ροής

    print("Reading London parquet...")
    df = pd.read_parquet(LONDON_PARQUET, columns=[
        "Number","Start date",
        "Start station number","Start station",
        "End station number","End station",
        "Bike model", "Total duration (ms)",
    ])

    df = df.rename(columns={
        "Number":              "trip_id",
        "Start date":           "started_at",
        "Start station number": "start_station_id",
        "Start station":        "start_station_name",
        "End station number":   "end_station_id",
        "End station":          "end_station_name",
        "Bike model":           "rideable_type",
        "Total duration (ms)":  "duration_ms",
    })
    steps["raw"] = len(df)

    # Τα εβδομαδιαία αρχεία του TfL μπορεί να επικαλύπτονται
    df = df.drop_duplicates(subset=["trip_id"])
    steps["after_dedup"] = len(df)

    for c in ["start_station_id", "end_station_id", "start_station_name", "end_station_name"]:
        df[c] = df[c].astype("string").str.strip()

    df["started_at"]    = pd.to_datetime(df["started_at"],format=DATE_FORMAT, errors="coerce")
    df["trip_duration"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 60000.0
    df["rideable_type"] = df["rideable_type"].map(BIKE_MODEL_MAP)
    df["member_casual"] = "unknown"
    df["city"]          = "london"
    df = df.drop(columns=["duration_ms"])

    print(f"Unparsed start dates: {df['started_at'].isna().sum():,}")
    print(f"Missing duration:     {df['trip_duration'].isna().sum():,}")
    print(f"Unknown bike model:   {df['rideable_type'].isna().sum():,}")
    df = df.dropna(subset=["started_at", "trip_duration", "rideable_type"])
    steps["valid_time_duration_bike"] = len(df)

    # Μόνο διαδρομές που ξεκινούν το 2025 (τα εβδομαδιαία αρχεία ξεπερνούν τα όρια του έτους)
    df = df[df["started_at"].dt.year == 2025]
    steps["year_2025"] = len(df)

    df = df[df["start_station_id"].fillna("").ne("") & df["end_station_id"].fillna("").ne("")]
    steps["with_station"] = len(df)

    # ---- 3. Χρονικά χαρακτηριστικά (πριν το join, για το TVD) ----
    df["hour"]        = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.dayofweek      # 0 = Δευτέρα, όπως στις ΗΠΑ
    df["month"]       = df["started_at"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["part_of_day"] = np.select(
        [
            (df["hour"] >= 5)  & (df["hour"] <= 10),
            (df["hour"] >= 11) & (df["hour"] <= 15),
            (df["hour"] >= 16) & (df["hour"] <= 19),
        ],
        ["morning_peak", "midday", "evening_peak"],
        default="night",
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # ---- 4. Αντιστοίχιση με το BikePoint ----
    old_map = resolve_old_ids(df, stations)
    n_old = (df["start_station_id"].isin(old_map.keys()) | df["end_station_id"].isin(old_map.keys())).sum()
    print(f"\nOld station IDs mapped to current ID: {len(old_map)} IDs, {n_old:,} trips")
    for old, new in sorted(old_map.items()):
        print(f"  {old} -> {new}  ({stations.loc[new, 'station_name']})")
    for c in ["start_station_id", "end_station_id"]:
        df[c] = df[c].replace(old_map)
    df = df.drop(columns=["start_station_name", "end_station_name"])

    invalid = (df["start_station_id"].isin(LONDON_INVALID_STATION_IDS)
               | df["end_station_id"].isin(LONDON_INVALID_STATION_IDS))
    print(f"Trips with reused station ID (invalid BikePoint position): {invalid.sum():,}")
    df = df[~invalid]
    steps["valid_station_position"] = len(df)

    known = set(stations.index)
    start_ok = df["start_station_id"].isin(known)
    end_ok   = df["end_station_id"].isin(known)
    print(f"\nUnmatched start station: {(~start_ok).sum():,} ({100 * (~start_ok).mean():.2f}%)")
    print(f"Unmatched end station:   {(~end_ok).sum():,} ({100 * (~end_ok).mean():.2f}%)")
    print("Most frequent unmatched station numbers:")
    print(pd.concat([df.loc[~start_ok, "start_station_id"],
                     df.loc[~end_ok, "end_station_id"]]).value_counts().head(20).to_string())

    before = df
    df = df[start_ok & end_ok]
    print("TVD before vs after station matching (§3.4):")
    for c in ["month", "rideable_type", "hour", "is_weekend"]:
        print(f"  {c:<14} {tvd(before[c], df[c]):.5f}")
    del before
    steps["matched_bikepoint"] = len(df)

    coords = stations[["lat", "lng"]]
    df = df.join(coords.rename(columns={"lat": "start_lat", "lng": "start_lng"}), on="start_station_id")
    df = df.join(coords.rename(columns={"lat": "end_lat",   "lng": "end_lng"}),   on="end_station_id")

    # ---- 5. Φυσικοί σταθμοί (ίδιος κανόνας με το 02b) και κυκλικές ----
    n_trips = pd.concat([df["start_station_id"], df["end_station_id"]]).value_counts()
    s = group_stations(stations, n_trips)

    s[["station_id", "station_key"]].to_parquet(STATION_MAP_PATH, index=False)
    station_table = (
        s.groupby("station_key")
         .agg(n_trips=("n_trips", "sum"),
              n_ids=("station_id", "size"),
              member_ids=("station_id", lambda x: " | ".join(sorted(x))))
         .join(s.set_index("station_id")[["lat", "lng"]])
         .reset_index()
         .assign(source="bikepoint")
    )
    station_table.to_parquet(STATION_TABLE_PATH, index=False)
    print(f"\nPhysical stations: {len(station_table):,} (from {len(s):,} BikePoint IDs, "
          f"{(station_table['n_ids'] > 1).sum():,} with >1 ID, "
          f"{(station_table['n_trips'] == 0).sum():,} without 2025 trips)")

    key_of = s.set_index("station_id")["station_key"]
    df["start_station_key"] = df["start_station_id"].map(key_of)
    df["end_station_key"]   = df["end_station_id"].map(key_of)
    df["same_station"] = (df["start_station_key"] == df["end_station_key"]).astype(int)


    # ---- 6. Απόσταση ----
    df["haversine_distance_km"] = haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])
    bearing = bearing_deg(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])
    df["bearing_sin"] = np.sin(np.radians(bearing))
    df["bearing_cos"] = np.cos(np.radians(bearing))
    df = df.drop(columns=["start_lat", "start_lng", "end_lat", "end_lng"])


    # ---- 7. Φιλτράρισμα: ίδιοι κανόνες με το 03, με την ίδια σειρά ----
    df = df[(df["trip_duration"] > MIN_TRIP_DURATION_MIN) & (df["trip_duration"] < MAX_TRIP_DURATION_MIN)]
    steps["duration"] = len(df)
    df = df[(df["haversine_distance_km"] > MIN_MOVING_DISTANCE_KM) | (df["same_station"] == 1)]
    steps["min_distance"] = len(df)
    df = df[df["haversine_distance_km"] < MAX_TRIP_DISTANCE_KM]
    steps["max_distance"] = len(df)
    df = df[(df["same_station"] == 1) |
            (df["haversine_distance_km"] * 60 <= MAX_SPEED_KMH * df["trip_duration"])]
    steps["speed"] = len(df)

    # ---- 8. Έξοδος ----
    output_cols = [
        "rideable_type", "started_at",
        "start_station_id", "end_station_id", "start_station_key", "end_station_key",
        "member_casual", "city", "trip_duration", "same_station",
        "hour", "day_of_week", "month", "is_weekend", "part_of_day",
        "hour_sin", "hour_cos",
        "haversine_distance_km", "bearing_sin", "bearing_cos",
    ]
    df = (df[output_cols]
          .sort_values(["started_at", "start_station_id", "end_station_id", "trip_duration"])
          .reset_index(drop=True))

   
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    print("\n--- Cleaning flow (sequential) ---")
    print(pd.Series(steps).to_string())
    print(f"\nSaved {len(df):,} rows to {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    build_london_featured_parquet()
