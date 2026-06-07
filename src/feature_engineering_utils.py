import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

KNOWN_CATEGORIES = {
    "rideable_type": ["classic_bike", "electric_bike"],
    "part_of_day":   ["morning_peak", "midday", "evening_peak", "night"],
}

# --- Model input schema (single source of truth for 06 and 08) ---
CATEGORICAL_COLS = [
    "rideable_type",
    "part_of_day",
]

DISCRETE_COLS = [
    "day_of_week",
    "month",
]

BINARY_COLS = [
    "is_weekend",
    "same_station",
    "rain_flag",
    "snow_flag",
]

CYCLICAL_COLS = [
    "hour_sin",
    "hour_cos",
    "bearing_sin",
    "bearing_cos",
]

LOG1P_CONTINUOUS_COLS = [
    "trip_duration",
    "haversine_distance_km",
    "precipitation_mm",
    "wind_speed_kph",
    "snow_mm",
]

STANDARD_CONTINUOUS_COLS = [
    "temperature_c",
]

CONTINUOUS_COLS = LOG1P_CONTINUOUS_COLS + STANDARD_CONTINUOUS_COLS

DROP_COLS = [
    "hour",
]

SELECTED_INPUT_COLS = (
    CATEGORICAL_COLS
    + DISCRETE_COLS
    + BINARY_COLS
    + CYCLICAL_COLS
    + CONTINUOUS_COLS
    + DROP_COLS
)


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points on Earth in kilometers.
    """
    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * earth_radius_km


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Compute bearing from point 1 to point 2 in degrees.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (
        np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def get_part_of_day(hour):
    if 5 <= hour < 11:
        return "morning_peak"
    if 11 <= hour < 16:
        return "midday"
    if 16 <= hour < 20:
        return "evening_peak"
    return "night"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the bike trips dataset.
    """
    print(f"Initial shape: {df.shape}")

    df = df.copy()

    df["trip_duration"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    df["same_station"] = (
        df["start_station_id"].notna()
        & df["end_station_id"].notna()
        & (df["start_station_id"] == df["end_station_id"])
    ).astype(int)

    df["hour"] = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.dayofweek
    df["month"] = df["started_at"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["part_of_day"] = df["hour"].apply(get_part_of_day)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["delta_lat"] = df["end_lat"] - df["start_lat"]
    df["delta_lng"] = df["end_lng"] - df["start_lng"]

    df["haversine_distance_km"] = haversine(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )

    df["bearing"] = calculate_bearing(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )

    bearing_rad = np.radians(df["bearing"])
    df["bearing_sin"] = np.sin(bearing_rad)
    df["bearing_cos"] = np.cos(bearing_rad)

    df = df.drop(columns=["start_station_id", "end_station_id"], errors="ignore")

    print(f"Final shape after feature engineering: {df.shape}")
    return df


# --- Model input transforms (shared between 06 and 08) ---

def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in SELECTED_INPUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "The training dataset is missing expected columns: "
            + ", ".join(missing)
        )


def transform_continuous_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    for col in CONTINUOUS_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        fill_value = df[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        df[col] = df[col].fillna(fill_value)

    # These variables are nonnegative and right-skewed, so we first compress the tail.
    for col in LOG1P_CONTINUOUS_COLS:
        df[col] = df[col].clip(lower=0)
        df[col] = df[col].map(lambda value: np.log1p(value))

    scaler = StandardScaler()
    df[CONTINUOUS_COLS] = scaler.fit_transform(df[CONTINUOUS_COLS])
    return df, scaler


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLS)

    for col in BINARY_COLS:
        df[col] = df[col].fillna(0)

    # Fix category ordering before get_dummies so all datasets produce identical columns.
    for col in CATEGORICAL_COLS:
        if col in KNOWN_CATEGORIES:
            df[col] = pd.Categorical(df[col], categories=KNOWN_CATEGORIES[col])

    encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        dtype="int8",
    )

    for col in BINARY_COLS + DISCRETE_COLS:
        encoded[col] = encoded[col].astype("int8")

    for col in CYCLICAL_COLS + CONTINUOUS_COLS:
        encoded[col] = encoded[col].astype("float32")

    return encoded
