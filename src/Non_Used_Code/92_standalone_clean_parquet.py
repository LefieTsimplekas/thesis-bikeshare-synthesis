from pathlib import Path
import pandas as pd
from Non_Used_Code.data_cleaning import clean_data

"""
Standalone cleaner for parquet datasets.

This script:
1. Loads one parquet dataset per city
2. Removes rows with invalid timestamps
3. Removes rows with missing coordinates
4. Keeps only valid coordinate ranges
5. Optionally removes duplicates
6. Saves one cleaned parquet file per city

Use this only if you already have parquet files that still need cleaning.
The main pipeline performs loading, cleaning, and feature engineering directly in
`01_build_featured_data.py`, so this helper script is optional.

Default input:
- data/processed/chicago_raw.parquet
- data/processed/nyc_raw.parquet

Default output:
- data/processed/chicago_cleaned.parquet
- data/processed/nyc_cleaned.parquet
"""

PROCESSED_DIR = Path("data/processed")


def process_city(
    city_name: str,
    remove_duplicates: bool = False,
    input_suffix: str = "raw",
    output_suffix: str = "cleaned",
) -> None:
    input_path = PROCESSED_DIR / f"{city_name}_{input_suffix}.parquet"
    output_path = PROCESSED_DIR / f"{city_name}_{output_suffix}.parquet"

    print(f"\nProcessing city: {city_name}")
    print(f"Loading: {input_path}")

    df = pd.read_parquet(input_path)
    df_clean = clean_data(df, remove_duplicates=remove_duplicates)

    df_clean.to_parquet(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    process_city("chicago", remove_duplicates=False)
    process_city("nyc", remove_duplicates=False)
