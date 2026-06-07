from pathlib import Path
import pandas as pd
from feature_engineering_utils import create_features


PROCESSED_DIR = Path("data/processed")

"""
Standalone feature engineering helper for already-cleaned parquet datasets.

Use this only if you already have `*_cleaned.parquet` files and want to build
`*_featured.parquet` separately from the main pipeline.
"""


def process_city(city_name: str) -> None:
    input_path = PROCESSED_DIR / f"{city_name}_cleaned.parquet"
    output_path = PROCESSED_DIR / f"{city_name}_featured.parquet"

    print(f"\nProcessing city: {city_name}")
    print(f"Loading: {input_path}")

    df = pd.read_parquet(input_path)
    df_featured = create_features(df)

    df_featured.to_parquet(output_path, index=False)
    print(f"Featured data saved to: {output_path}")


if __name__ == "__main__":
    process_city("chicago")
    process_city("nyc")
