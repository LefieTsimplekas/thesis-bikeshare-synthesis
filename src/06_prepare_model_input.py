from __future__ import annotations

import sys
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering_utils import (
    KNOWN_CATEGORIES,
    SELECTED_INPUT_COLS,
    validate_columns,
    transform_continuous_features,
    encode_categorical_features,
)

"""
Prepare weather-enriched sampled datasets for tabular diffusion modeling.

Loops over all three Level-1 regime datasets and produces per-dataset artifacts.
The `city` and `member_casual` columns are excluded from model features so all
three datasets (and London) share the exact same feature schema.

Outputs per dataset (name in chicago_sampled / nyc_sampled / chicago_nyc_sampled):
- data/model_input/model_input_{name}.parquet
- data/model_input/model_input_feature_info_{name}.json
- artifacts/model_input/continuous_scaler_{name}.joblib
- artifacts/model_input/final_feature_columns_{name}.json
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_INPUT_DIR = PROJECT_ROOT / "data" / "model_input"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "model_input"

DATASET_NAMES = ["chicago_sampled", "nyc_sampled", "chicago_nyc_sampled"]


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading dataset from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Input shape: {df.shape}")

    validate_columns(df)
    return df.loc[:, SELECTED_INPUT_COLS].copy()


def save_artifacts(
    name: str,
    encoded_df: pd.DataFrame,
    scaler: StandardScaler,
    input_path: Path,
) -> None:
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = MODEL_INPUT_DIR / f"model_input_{name}.parquet"
    scaler_path = ARTIFACTS_DIR / f"continuous_scaler_{name}.joblib"
    feature_columns_path = ARTIFACTS_DIR / f"final_feature_columns_{name}.json"
    feature_info_path = MODEL_INPUT_DIR / f"model_input_feature_info_{name}.json"

    encoded_df.to_parquet(output_path, index=False)
    joblib.dump(scaler, scaler_path)

    with feature_columns_path.open("w", encoding="utf-8") as f:
        json.dump(encoded_df.columns.tolist(), f, indent=2)

    feature_info = {
        "dataset_name": name,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "categorical_columns": KNOWN_CATEGORIES,
        "continuous_transform": "log1p_for_nonnegative_skewed_columns_then_standard_scaler",
        "categorical_encoding": "one_hot",
        "final_feature_count": len(encoded_df.columns),
    }

    with feature_info_path.open("w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  Saved: {scaler_path}")
    print(f"  Saved: {feature_columns_path}")
    print(f"  Saved: {feature_info_path}")


def process_dataset(name: str) -> None:
    input_path = PROCESSED_DIR / f"{name}_with_weather.parquet"

    df = load_input_dataframe(input_path)

    print("\nSelected columns for model input:")
    print(SELECTED_INPUT_COLS)

    df, scaler = transform_continuous_features(df)
    encoded_df = encode_categorical_features(df)

    print(f"\nProcessed shape: {encoded_df.shape}")
    save_artifacts(name, encoded_df, scaler, input_path)


def main() -> None:
    for name in DATASET_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {name}")
        print(f"{'=' * 60}")
        process_dataset(name)

    col_sets = []
    for name in DATASET_NAMES:
        path = ARTIFACTS_DIR / f"final_feature_columns_{name}.json"
        with path.open(encoding="utf-8") as f:
            col_sets.append(json.load(f))

    ref = col_sets[0]
    for name, cols in zip(DATASET_NAMES[1:], col_sets[1:]):
        if cols != ref:
            raise RuntimeError(
                f"Feature schema mismatch: {DATASET_NAMES[0]} vs {name}\n"
                f"Symmetric diff: {set(ref) ^ set(cols)}"
            )
    print("\nSanity check PASSED: all three datasets have identical feature schemas.")


if __name__ == "__main__":
    main()
