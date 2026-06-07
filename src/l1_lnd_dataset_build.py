from pathlib import Path
import numpy as np
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CONTEXT_DIR = Path("data/context")
WEATHER_DIR = CONTEXT_DIR / "weather"

LONDON_PARQUET = RAW_DIR / "london" / "london.parquet"
STATIONS_PARQUET = CONTEXT_DIR / "london_stations.parquet"
OUTPUT_PATH = PROCESSED_DIR / "london_featured.parquet"
OUTPUT_SAMPLED_PATH = PROCESSED_DIR / "london_sampled.parquet"
OUTPUT_WITH_WEATHER_PATH = PROCESSED_DIR / "london_sampled_with_weather.parquet"

STRATA_COLS = ["month", "rideable_type", "part_of_day", "is_weekend"]
TARGET_ROWS = 1_000_000

TFL_BIKEPOINT_URL = "https://api.tfl.gov.uk/BikePoint"

WEATHER_COLUMNS = [
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kph",
    "snow_mm",
    "rain_flag",
    "snow_flag",
]


def load_or_fetch_stations() -> pd.DataFrame:
    if STATIONS_PARQUET.exists():
        print(f"Using cached stations from {STATIONS_PARQUET}")
        return pd.read_parquet(STATIONS_PARQUET)

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


def build_london_featured_parquet() -> pd.DataFrame:
    stations = load_or_fetch_stations().set_index("station_id")

    print("Reading London parquet...")
    df = pd.read_parquet(LONDON_PARQUET, columns=[
        "Start date", "End date",
        "Start station number", "End station number",
        "Bike model", "Total duration (ms)",
    ])

    df = df.rename(columns={
        "Start date":           "started_at",
        "End date":             "ended_at",
        "Start station number": "start_station_id",
        "End station number":   "end_station_id",
        "Bike model":           "rideable_type",
        "Total duration (ms)":  "duration_ms",
    })

    df["started_at"]    = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"]      = pd.to_datetime(df["ended_at"],   errors="coerce")
    df["trip_duration"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 60000.0
    df["rideable_type"] = df["rideable_type"].map({"CLASSIC": "classic_bike", "PBSC_EBIKE": "electric_bike"}).fillna(df["rideable_type"])
    df["member_casual"] = "unknown"
    df["city"]          = "london"
    df = df.drop(columns=["duration_ms", "ended_at"])

    df = df.dropna(subset=["started_at", "trip_duration", "start_station_id", "end_station_id"])
    df = df[df["trip_duration"] > 0]
    print(f"Rows after basic cleaning: {len(df):,}")

    df = df.join(stations[["lat", "lng"]].rename(columns={"lat": "start_lat", "lng": "start_lng"}), on="start_station_id", how="inner")
    df = df.join(stations[["lat", "lng"]].rename(columns={"lat": "end_lat",   "lng": "end_lng"}),   on="end_station_id",   how="inner")
    df = df.reset_index(drop=True)
    print(f"Rows after station join: {len(df):,}")

    df["same_station"] = (df["start_station_id"] == df["end_station_id"]).astype(int)
    df["hour"]         = df["started_at"].dt.hour
    df["day_of_week"]  = df["started_at"].dt.dayofweek
    df["month"]        = df["started_at"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

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

    df["haversine_distance_km"] = haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])

    bearing = bearing_deg(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])
    df["bearing_sin"] = np.sin(np.radians(bearing))
    df["bearing_cos"] = np.cos(np.radians(bearing))

    df = df.drop(columns=["start_station_id", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng"])
    df = df.reset_index(drop=True)

    before = len(df)
    df = df.query(
        "trip_duration > 1.0 and trip_duration < 100.0"
        " and haversine_distance_km < 12.0"
        " and (haversine_distance_km > 0.05 or same_station == 1)"
    ).reset_index(drop=True)
    print(f"Rows after filtering: {len(df):,} (removed {before - len(df):,})")

    output_cols = [
        "rideable_type", "started_at",
        "member_casual", "city", "trip_duration", "same_station",
        "hour", "day_of_week", "month", "is_weekend", "part_of_day",
        "hour_sin", "hour_cos",
        "haversine_distance_km", "bearing_sin", "bearing_cos",
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    return df[output_cols]


def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    effective_target = min(TARGET_ROWS, total)

    # Proportional allocation — exact count via largest-remainder method
    counts = df.groupby(STRATA_COLS, observed=True).size()
    raw = counts * effective_target / total
    allocation = np.floor(raw).astype(int)
    leftover = effective_target - int(allocation.sum())
    if leftover > 0:
        top = (raw - allocation).nlargest(leftover).index
        allocation[top] += 1

    allocation = allocation.reset_index(name="target_n")

    df_m = df.merge(allocation, on=STRATA_COLS, how="left")
    df_m["_rand"] = np.random.default_rng(42).random(len(df_m))
    df_m["_rank"] = df_m.groupby(STRATA_COLS, observed=True)["_rand"].rank(method="first")

    sampled = (
        df_m[df_m["_rank"] <= df_m["target_n"]]
        .drop(columns=["target_n", "_rand", "_rank"])
        .reset_index(drop=True)
    )

    print(f"Sampled {len(sampled):,} rows from {total:,} ({len(sampled) / total:.1%})")
    sampled.to_parquet(OUTPUT_SAMPLED_PATH, index=False, compression="zstd")
    print(f"Saved to {OUTPUT_SAMPLED_PATH}")
    return sampled


def add_weather(df: pd.DataFrame) -> None:
    weather_path = WEATHER_DIR / "london_hourly_weather.parquet"
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {weather_path}\n"
            "Run src/96_build_weather_context.py first."
        )

    weather = pd.read_parquet(weather_path)
    weather["weather_hour"] = pd.to_datetime(weather["weather_hour"]).dt.floor("h")
    weather = weather.drop_duplicates(subset=["city", "weather_hour"], keep="first")
    weather = weather[["city", "weather_hour"] + WEATHER_COLUMNS]

    df = df.copy()
    df["weather_hour"] = df["started_at"].dt.floor("h")

    merged = df.merge(weather, on=["city", "weather_hour"], how="left", validate="many_to_one")

    missing = merged["temperature_c"].isna().sum()
    print(f"Rows without matched weather: {missing:,} ({100 * missing / len(merged):.2f}%)")

    merged = merged.drop(columns=["weather_hour"])
    merged.to_parquet(OUTPUT_WITH_WEATHER_PATH, index=False, compression="zstd")
    print(f"Saved {len(merged):,} rows to {OUTPUT_WITH_WEATHER_PATH}")


if __name__ == "__main__":
    featured = build_london_featured_parquet()
    sampled = stratified_sample(featured)
    add_weather(sampled)
