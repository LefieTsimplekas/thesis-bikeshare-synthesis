from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import duckdb


DEFAULT_INPUT_DIR = Path("data/raw/nyc")
DEFAULT_OUTPUT_DIR = Path("data/raw/nyc_monthly")
DEFAULT_ALL_OUTPUT_PATH = Path("data/raw/nyc_tripdata.parquet")
FILENAME_PATTERN = re.compile(
    r"^(?P<month>\d{6}-citibike-tripdata)_(?P<part>\d+)\.parquet$",
    re.IGNORECASE,
)


def sql_string(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def parquet_sql_list(paths: list[Path]) -> str:
    return ", ".join(f"'{sql_string(path)}'" for path in paths)


def group_monthly_parts(input_dir: Path) -> dict[str, list[Path]]:
    monthly_files: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for parquet_path in input_dir.glob("*.parquet"):
        match = FILENAME_PATTERN.match(parquet_path.name)
        if not match:
            continue

        month_key = match.group("month")
        part_number = int(match.group("part"))
        monthly_files[month_key].append((part_number, parquet_path))

    return {
        month_key: [path for _, path in sorted(parts)]
        for month_key, parts in sorted(monthly_files.items())
    }


def merge_parquet_files(
    input_files: list[Path],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    if not input_files:
        raise ValueError("No input parquet files were provided for merging.")

    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(".tmp.parquet")

    if temp_output_path.exists():
        temp_output_path.unlink()

    escaped_files = parquet_sql_list(input_files)
    query = f"""
    COPY (
        SELECT *
        FROM read_parquet(
            [{escaped_files}],
            union_by_name=True
        )
    )
    TO '{sql_string(temp_output_path)}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con = duckdb.connect()
    try:
        con.execute(query)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise
    finally:
        con.close()

    temp_output_path.replace(output_path)
    print(f"Saved: {output_path}")


def merge_nyc_parts_by_month(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    grouped_files = group_monthly_parts(input_dir)
    if not grouped_files:
        raise ValueError(f"No split CitiBike parquet files found in: {input_dir}")

    print(f"Found {len(grouped_files)} monthly groups in {input_dir}")

    for month_key, input_files in grouped_files.items():
        output_path = output_dir / f"{month_key}.parquet"
        print(f"\nMerging {month_key}: {len(input_files)} files")
        for input_file in input_files:
            print(f"  - {input_file.name}")

        merge_parquet_files(input_files, output_path, overwrite=overwrite)


def merge_all_nyc_parts(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_ALL_OUTPUT_PATH,
    overwrite: bool = False,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    grouped_files = group_monthly_parts(input_dir)
    if not grouped_files:
        raise ValueError(f"No split CitiBike parquet files found in: {input_dir}")

    input_files = [
        input_file
        for monthly_files in grouped_files.values()
        for input_file in monthly_files
    ]

    print(f"Found {len(input_files)} split files in {input_dir}")
    print(f"Merging all NYC files into: {output_path}")
    merge_parquet_files(input_files, output_path, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge NYC CitiBike split monthly parquet files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder with split NYC parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where merged monthly parquet files will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite merged files if they already exist.",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=None,
        help=(
            "Optional path for one parquet file containing all split NYC files. "
            f"Example: {DEFAULT_ALL_OUTPUT_PATH}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.single_output:
        merge_all_nyc_parts(
            input_dir=args.input_dir,
            output_path=args.single_output,
            overwrite=args.overwrite,
        )
    else:
        merge_nyc_parts_by_month(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
