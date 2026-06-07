from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
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
Split london_sampled_with_weather into 80/20 train/test and prepare model input
for the train split (oracle experiment, Experiment D: London->London).

Split rules:
- Stratified on month + rideable_type + part_of_day + is_weekend, seed=42.
- london_test.parquet is LOCKED: never overwritten unless --force is passed.
  It is the held-out ground truth for ALL experiments (A/B/C/D).
- london_train.parquet is the oracle training set (Experiment D only).

Model input is prepared ONLY for london_train (london_test stays as raw parquet).

Outputs:
- data/processed/london_train.parquet          (80%, raw)
- data/processed/london_test.parquet           (20%, raw, LOCKED)
- data/model_input/model_input_london_train.parquet
- data/model_input/model_input_feature_info_london_train.json
- artifacts/model_input/continuous_scaler_london_train.joblib
- artifacts/model_input/final_feature_columns_london_train.json
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_INPUT_DIR = PROJECT_ROOT / "data" / "model_input"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "model_input"

TRAIN_PATH = PROCESSED_DIR / "london_train.parquet"
TEST_PATH = PROCESSED_DIR / "london_test.parquet"

STRATA_COLS = ["month", "rideable_type", "part_of_day", "is_weekend"]
TEST_SIZE = 0.20
SEED = 42

ALL_NAMES = ["chicago_sampled", "nyc_sampled", "chicago_nyc_sampled", "london_train"]


def load_and_split(force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if TEST_PATH.exists() and not force:
        print("london_test.parquet is LOCKED — loading existing split (--force to overwrite).")
        train_df = pd.read_parquet(TRAIN_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        print(f"Loaded: train {len(train_df):,} rows | test {len(test_df):,} rows")
        return train_df, test_df

    input_path = PROCESSED_DIR / "london_sampled_with_weather.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Total rows: {len(df):,}")

    df["_strata_key"] = df[STRATA_COLS[0]].astype(str)
    for col in STRATA_COLS[1:]:
        df["_strata_key"] += "_" + df[col].astype(str)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["_strata_key"],
    )

    overlap = set(train_df.index) & set(test_df.index)
    assert len(overlap) == 0, f"Train/test index overlap: {len(overlap)} rows"

    total = df.groupby("_strata_key").size().rename("total")
    n_test = test_df.groupby("_strata_key").size().rename("n_test")
    report = total.to_frame().join(n_test)
    report["test_ratio"] = report["n_test"] / report["total"]
    print(f"\nPer-stratum test_ratio (expect ~{TEST_SIZE}):")
    print(report["test_ratio"].describe().round(4).to_string())

    train_df = train_df.drop(columns=["_strata_key"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["_strata_key"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"\nSaved london_train: {len(train_df):,} rows  -> {TRAIN_PATH}")
    print(f"Saved london_test:  {len(test_df):,} rows  -> {TEST_PATH}  [LOCKED]")

    return train_df, test_df


def save_artifacts(encoded_df: pd.DataFrame, scaler: StandardScaler) -> None:
    name = "london_train"
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
        "input_file": str(TRAIN_PATH),
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


def sanity_check_all() -> None:
    col_sets: dict[str, list[str]] = {}
    for name in ALL_NAMES:
        path = ARTIFACTS_DIR / f"final_feature_columns_{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing feature columns file for {name}: {path}\n"
                "Run 06_prepare_model_input.py first."
            )
        with path.open(encoding="utf-8") as f:
            col_sets[name] = json.load(f)

    ref_name = ALL_NAMES[0]
    ref = col_sets[ref_name]
    for name in ALL_NAMES[1:]:
        if col_sets[name] != ref:
            raise RuntimeError(
                f"Feature schema mismatch: {ref_name} vs {name}\n"
                f"Symmetric diff: {set(ref) ^ set(col_sets[name])}"
            )

    print(f"\nSanity check PASSED: all {len(ALL_NAMES)} datasets have identical feature schemas.")
    print(f"  {ALL_NAMES}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split London dataset and prepare oracle model input."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite locked london_test.parquet and redo the split.",
    )
    args = parser.parse_args()

    train_df, test_df = load_and_split(args.force)
    print(f"\nTrain: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    print(f"\n{'=' * 60}")
    print("Preparing model input for london_train")
    print(f"{'=' * 60}")

    validate_columns(train_df)
    df = train_df.loc[:, SELECTED_INPUT_COLS].copy()
    df, scaler = transform_continuous_features(df)
    encoded_df = encode_categorical_features(df)

    print(f"\nProcessed shape: {encoded_df.shape}")
    save_artifacts(encoded_df, scaler)

    sanity_check_all()


if __name__ == "__main__":
    main()
