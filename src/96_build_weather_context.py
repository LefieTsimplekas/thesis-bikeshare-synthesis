from pathlib import Path
import duckdb


"""
Build hourly weather context per city.

This script fetches hourly weather aligned to the trip time range using Meteostat
and writes one parquet file per city.

Outputs:
- data/context/weather/chicago_hourly_weather.parquet
- data/context/weather/nyc_hourly_weather.parquet
- data/context/weather/london_hourly_weather.parquet

Requirements:
- meteostat
- pandas
"""


PROCESSED_DIR = Path("data/processed")
WEATHER_DIR = Path("data/context/weather")


def sql_string(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def require_weather_dependencies():
    try:
        import pandas as pd
        from meteostat import Point, stations
    except ImportError as exc:
        raise ImportError(
            "This script requires meteostat and pandas. Install them first."
        ) from exc

    try:
        from meteostat import Stations  # type: ignore[attr-defined]
    except ImportError:
        Stations = None

    try:
        from meteostat import Hourly as hourly_api
    except ImportError:
        from meteostat import hourly as hourly_api

    stations_api = Stations() if Stations is not None else stations
    return pd, hourly_api, Point, stations_api


def score_weather_coverage(pd, weather_df, start, end) -> int:
    if weather_df is None or weather_df.empty:
        return 0

    weather_hours = pd.to_datetime(weather_df.index).floor("h")
    if "temp" in weather_df.columns:
        weather_hours = weather_hours[weather_df["temp"].notna()]

    expected_hours = pd.date_range(
        pd.to_datetime(start).floor("h"),
        pd.to_datetime(end).floor("h"),
        freq="h",
    )
    return weather_hours.intersection(expected_hours).nunique()


def fetch_hourly_weather(pd, hourly_api, stations_api, location, start, end):
    # Try the point interpolation first, then nearby real stations. We choose the
    # candidate with the widest hourly coverage, not merely the first non-empty one.
    candidates = []

    point_weather = fetch_hourly_candidate(hourly_api, location, start, end)
    candidates.append(
        {
            "label": "reference point",
            "distance_m": 0.0,
            "weather_df": point_weather,
        }
    )

    if stations_api is None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["weather_df"] is not None and not candidate["weather_df"].empty
        ]
        if not candidates:
            raise RuntimeError(
                "Meteostat returned no weather for the reference point and this "
                "installed version does not expose a stations API."
            )

    else:
        nearby_stations = stations_api.nearby(location, radius=100000, limit=25)
        if nearby_stations.empty:
            raise RuntimeError("Meteostat returned no nearby weather stations.")

        for station_id, station in nearby_stations.iterrows():
            weather_df = fetch_hourly_candidate(hourly_api, str(station_id), start, end)
            candidates.append(
                {
                    "label": f"{station_id} ({station['name']})",
                    "distance_m": station["distance"],
                    "weather_df": weather_df,
                }
            )

    scored_candidates = []
    for candidate in candidates:
        coverage_hours = score_weather_coverage(
            pd, candidate["weather_df"], start, end
        )
        if coverage_hours > 0:
            scored_candidates.append({**candidate, "coverage_hours": coverage_hours})

    if not scored_candidates:
        raise RuntimeError(
            "Meteostat returned no hourly weather rows for the reference point "
            "or nearby stations."
        )

    best_candidate = max(
        scored_candidates,
        key=lambda candidate: (
            candidate["coverage_hours"],
            -candidate["distance_m"],
        ),
    )
    distance_km = best_candidate["distance_m"] / 1000
    print(
        "Using weather source "
        f"{best_candidate['label']} ({best_candidate['coverage_hours']} hours, "
        f"{distance_km:.1f} km away)"
    )
    return best_candidate["weather_df"].reset_index()


def fetch_hourly_candidate(hourly_api, station_or_location, start, end):
    result = hourly_api(station_or_location, start, end)
    if hasattr(result, "fetch"):
        return result.fetch()

    return result


def build_city_weather(city_name: str) -> None:
    pd, hourly_api, Point, stations_api = require_weather_dependencies()

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    featured_path = PROCESSED_DIR / f"{city_name}_featured.parquet"
    output_path = WEATHER_DIR / f"{city_name}_hourly_weather.parquet"

    if not featured_path.exists():
        raise FileNotFoundError(f"Featured data not found: {featured_path}")

    con = duckdb.connect()
    query = f"""
    SELECT
        AVG(start_lat) AS mean_lat,
        AVG(start_lng) AS mean_lng,
        MIN(started_at) AS min_started_at,
        MAX(started_at) AS max_started_at
    FROM read_parquet('{sql_string(featured_path)}')
    """
    try:
        mean_lat, mean_lng, min_started_at, max_started_at = con.execute(
            query
        ).fetchone()
    finally:
        con.close()

    if mean_lat is None or mean_lng is None:
        raise ValueError(f"Cannot build weather context from empty file: {featured_path}")

    print(f"\n{'=' * 60}")
    print(f"Building hourly weather context: {city_name}")
    print(f"{'=' * 60}")
    print(f"Reference point: ({mean_lat}, {mean_lng})")
    print(f"Date range: {min_started_at} -> {max_started_at}")

    location = Point(mean_lat, mean_lng)
    weather_df = fetch_hourly_weather(
        pd,
        hourly_api,
        stations_api,
        location,
        min_started_at,
        max_started_at,
    )

    weather_df = weather_df.rename(
        columns={
            "time": "weather_hour",
            "temp": "temperature_c",
            "prcp": "precipitation_mm",
            "wspd": "wind_speed_kph",
            "snow": "snow_mm",
            "snwd": "snow_mm",
        }
    )

    if "weather_hour" in weather_df.columns:
        weather_df["weather_hour"] = pd.to_datetime(weather_df["weather_hour"]).dt.floor(
            "h"
        )
    if "precipitation_mm" not in weather_df.columns:
        weather_df["precipitation_mm"] = 0.0
    else:
        weather_df["precipitation_mm"] = weather_df["precipitation_mm"].fillna(0.0)
    if "wind_speed_kph" not in weather_df.columns:
        weather_df["wind_speed_kph"] = pd.NA
    if "snow_mm" not in weather_df.columns:
        weather_df["snow_mm"] = 0.0
    else:
        weather_df["snow_mm"] = weather_df["snow_mm"].fillna(0.0)
    if "temperature_c" not in weather_df.columns:
        weather_df["temperature_c"] = pd.NA

    weather_df["rain_flag"] = (weather_df["precipitation_mm"].fillna(0) > 0).astype(int)
    weather_df["snow_flag"] = (weather_df["snow_mm"].fillna(0) > 0).astype(int)
    weather_df["city"] = city_name

    output_columns = [
        "city",
        "weather_hour",
        "temperature_c",
        "precipitation_mm",
        "wind_speed_kph",
        "snow_mm",
        "rain_flag",
        "snow_flag",
    ]

    weather_df[output_columns].to_parquet(output_path, index=False)
    print(f"Saved weather context to: {output_path}")
    print(weather_df[output_columns].head())


def main() -> None:
    for city_name in ["chicago", "nyc", "london"]:
        build_city_weather(city_name)


if __name__ == "__main__":
    main()
