from __future__ import annotations

from pathlib import Path

import pandas as pd

"""
Add hourly Meteostat weather features to each sampled training dataset.

This script runs after stratified sampling and before model-input preparation.
It joins each trip to city-level hourly weather using:

- city
- started_at rounded down to the hour

Inputs:
- data/processed/{name}.parquet  for each name in DATASET_NAMES
- data/context/weather/chicago_hourly_weather.parquet
- data/context/weather/nyc_hourly_weather.parquet

Outputs:
- data/processed/chicago_sampled_with_weather.parquet
- data/processed/nyc_sampled_with_weather.parquet
- data/processed/chicago_nyc_sampled_with_weather.parquet
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WEATHER_DIR = PROJECT_ROOT / "data" / "context" / "weather"

DATASET_NAMES = ["chicago_sampled", "nyc_sampled", "chicago_nyc_sampled"]

CITY_NAMES = ["chicago", "nyc"]

WEATHER_COLUMNS = [
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kph",
    "snow_mm",
    "rain_flag",
    "snow_flag",
]


def load_trips(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading sampled dataset from: {input_path}")
    trips = pd.read_parquet(input_path)
    print(f"Trip rows: {len(trips):,}")

    required_cols = ["city", "started_at"]
    missing = [col for col in required_cols if col not in trips.columns]
    if missing:
        raise ValueError(
            "Cannot join weather because the dataset is missing: "
            + ", ".join(missing)
        )

    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips["weather_hour"] = trips["started_at"].dt.floor("h")
    return trips


def load_weather() -> pd.DataFrame:
    weather_frames = []

    for city_name in CITY_NAMES:
        weather_path = WEATHER_DIR / f"{city_name}_hourly_weather.parquet"
        if not weather_path.exists():
            raise FileNotFoundError(
                f"Weather file not found: {weather_path}. "
                "Run src/96_build_weather_context.py first."
            )

        city_weather = pd.read_parquet(weather_path)
        weather_frames.append(city_weather)

    weather = pd.concat(weather_frames, ignore_index=True)

    required_cols = ["city", "weather_hour"] + WEATHER_COLUMNS
    missing = [col for col in required_cols if col not in weather.columns]
    if missing:
        raise ValueError(
            "Weather data is missing expected columns: " + ", ".join(missing)
        )

    weather["weather_hour"] = pd.to_datetime(weather["weather_hour"]).dt.floor("h")
    weather = weather.drop_duplicates(subset=["city", "weather_hour"], keep="first")
    return weather.loc[:, required_cols]


def add_weather_features(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = trips.merge(
        weather,
        on=["city", "weather_hour"],
        how="left",
        validate="many_to_one",
    )

    missing_weather_rows = merged["temperature_c"].isna().sum()
    missing_weather_pct = (
        100.0 * missing_weather_rows / len(merged) if len(merged) else 0.0
    )

    print(
        "Rows without matched temperature: "
        f"{missing_weather_rows:,} ({missing_weather_pct:.2f}%)"
    )

    return merged.drop(columns=["weather_hour"])


def process_dataset(name: str, weather: pd.DataFrame) -> None:
    input_path = PROCESSED_DIR / f"{name}.parquet"
    output_path = PROCESSED_DIR / f"{name}_with_weather.parquet"

    trips = load_trips(input_path)
    final_df = add_weather_features(trips, weather)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print(f"Saved {name}_with_weather: {final_df.shape}")
    print(final_df[["started_at"] + WEATHER_COLUMNS].head())


def main() -> None:
    weather = load_weather()

    for name in DATASET_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Processing {name}")
        print(f"{'=' * 60}")
        process_dataset(name, weather)


if __name__ == "__main__":
    main()
