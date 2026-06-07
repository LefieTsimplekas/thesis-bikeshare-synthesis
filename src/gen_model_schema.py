"""
Single source of truth for the raw feature table fed to generative models.

We pass raw "hour" (0-23) as categorical rather than hour_sin/hour_cos +
part_of_day from step 06: TabDDPM handles discrete hour with multinomial
diffusion and part_of_day can be recovered post-hoc for diversity metrics.
To change the feature set, edit only MODEL_CATEGORICAL_COLS / MODEL_CONTINUOUS_COLS.
"""
import pandas as pd

MODEL_CATEGORICAL_COLS = [
    "rideable_type",
    "day_of_week",
    "month",
    "hour",
    "is_weekend",
    "same_station",
    "rain_flag",
    "snow_flag",
]

MODEL_CONTINUOUS_COLS = [
    "bearing_sin",
    "bearing_cos",
    "trip_duration",
    "haversine_distance_km",
    "precipitation_mm",
    "wind_speed_kph",
    "snow_mm",
    "temperature_c",
]

MODEL_COLS = MODEL_CATEGORICAL_COLS + MODEL_CONTINUOUS_COLS


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MODEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"build_model_table: missing columns: {missing}")
    out = df[MODEL_COLS].copy()
    for col in MODEL_CATEGORICAL_COLS:
        out[col] = out[col].astype("category")
    for col in MODEL_CONTINUOUS_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out
