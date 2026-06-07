from pathlib import Path
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONDON_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "london"
OUTPUT_PARQUET = LONDON_RAW_DIR / "london.parquet"
CHUNK_SIZE = 250_000


def convert_london_csvs_to_one_parquet(
    output_path: Path = OUTPUT_PARQUET,
    overwrite: bool = False,
) -> bool:
    if not LONDON_RAW_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {LONDON_RAW_DIR}")

    csv_files = sorted(LONDON_RAW_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {LONDON_RAW_DIR}")
        return False

    if output_path.exists() and not overwrite:
        print(f"Output already exists: {output_path}")
        return True

    temp_path = output_path.with_suffix(".tmp.parquet")

    if temp_path.exists():
        temp_path.unlink()

    writer = None
    columns = None
    total_rows = 0
    failed = False

    try:
        for csv_path in csv_files:
            print(f"Reading: {csv_path.name}")

            for chunk in pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):
                if columns is None:
                    columns = list(chunk.columns)
                else:
                    missing_columns = set(columns) - set(chunk.columns)
                    extra_columns = set(chunk.columns) - set(columns)

                    if missing_columns or extra_columns:
                        raise ValueError(
                            f"{csv_path.name} has different columns. "
                            f"Missing: {sorted(missing_columns)}. "
                            f"Extra: {sorted(extra_columns)}."
                        )

                    chunk = chunk[columns]

                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(temp_path, table.schema)

                writer.write_table(table)
                total_rows += len(chunk)
    except Exception as exc:
        failed = True
        print(f"Failed: {exc}")
    finally:
        if writer is not None:
            writer.close()

    if failed:
        if temp_path.exists():
            temp_path.unlink()
        return False

    temp_path.replace(output_path)
    print(f"Saved: {output_path}")
    print(f"Rows written: {total_rows}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert London raw CSV files to parquet files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite parquet files that already exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PARQUET,
        help="Output parquet path.",
    )
    args = parser.parse_args()

    convert_london_csvs_to_one_parquet(
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
