# What `01_build_featured_data.py` does

This file builds the first processed/featured dataset from the raw trip parquet files.

It reads raw bike trip data for each city, cleans invalid rows, creates useful time and spatial features, and saves the result as a compressed parquet file in `data/processed`.

## Main purpose

The script converts raw city trip files into model-ready featured files:

- `data/raw/chicago/*.parquet` becomes `data/processed/chicago_featured.parquet`
- `data/raw/nyc/*.parquet` becomes `data/processed/nyc_featured.parquet`

It uses DuckDB instead of pandas, so the processing can happen efficiently even when the datasets are large.

## Important paths

```python
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
```

The script expects the raw data to be organized by city:

```text
data/raw/chicago/
data/raw/nyc/
```

Each city folder should contain parquet files.

## Main function

```python
build_city_featured_parquet(city_name: str, remove_duplicates: bool = False)
```

This function processes one city at a time.

Example:

```python
build_city_featured_parquet("chicago", remove_duplicates=False)
```

It looks for all parquet files inside:

```text
data/raw/chicago/
```

If no files exist for that city, it stops with an error.

## Step 1: Read all raw parquet files

The script collects all `.parquet` files for a city:

```python
files = sorted(glob.glob(str(city_dir / "*.parquet")))
```

Then DuckDB reads them together using:

```sql
read_parquet([...], union_by_name=True)
```

`union_by_name=True` is useful because it matches columns by name, which helps if the parquet files do not have the exact same column order.

## Step 2: Standardize column types

Inside `typed_data`, the script converts important columns into consistent types:

- `rideable_type` becomes text
- `started_at` and `ended_at` become timestamps
- station IDs become text
- latitude and longitude columns become numbers
- `member_casual` becomes text
- a new `city` column is added

The `city` column is important because later Chicago and NYC data can be combined while still keeping track of where each trip came from.

## Step 3: Clean invalid rows

Inside `cleaned_data`, the script removes rows where important values are missing or invalid.

It keeps only rows where:

- `started_at` is valid
- `ended_at` is valid
- start and end coordinates are present
- latitude is between `-90` and `90`
- longitude is between `-180` and `180`

This removes rows that cannot be used reliably for trip duration or spatial calculations.

## Step 4: Optional duplicate removal

The argument `remove_duplicates` controls duplicate handling.

If it is `False`, the script keeps all cleaned rows:

```sql
SELECT * FROM cleaned_data
```

If it is `True`, it removes exact duplicate rows:

```sql
SELECT DISTINCT * FROM cleaned_data
```

In the current script, duplicate removal is disabled when the file is run directly.

## Step 5: Create base features

Inside `base_features`, the script creates the first set of engineered columns.

### `trip_duration`

```sql
date_diff('second', started_at, ended_at) / 60.0
```

This is the trip duration in minutes.

### `same_station`

This is `1` when the trip starts and ends at the same station.

It is `0` otherwise.

This can capture circular trips or trips where the bike was returned to the same place.

### Time features

The script extracts:

- `hour`
- `day_of_week`
- `month`

`day_of_week` uses this format:

```text
0 = Monday
1 = Tuesday
2 = Wednesday
3 = Thursday
4 = Friday
5 = Saturday
6 = Sunday
```

### Coordinate movement

The script creates:

- `delta_lat = end_lat - start_lat`
- `delta_lng = end_lng - start_lng`

These describe the rough direction of the trip.

For example:

- positive `delta_lat` means movement north
- negative `delta_lat` means movement south
- positive `delta_lng` means movement east
- negative `delta_lng` means movement west

### `haversine_distance_km`

This calculates the straight-line distance between the start and end coordinates in kilometers.

It uses the haversine formula, which accounts for the curvature of the Earth.

### `bearing_raw`

This calculates the direction angle from the start point to the end point.

The bearing is later normalized so it is always between `0` and `360` degrees.

## Step 6: Create final engineered features

Inside `engineered_data`, the script adds higher-level features.

### `is_weekend`

This is `1` for Saturday and Sunday.

It is `0` for Monday to Friday.

### `part_of_day`

The hour is grouped into mobility-related time periods:

- `morning_peak`: 05:00 to 10:59
- `midday`: 11:00 to 15:59
- `evening_peak`: 16:00 to 19:59
- `night`: 20:00 to 04:59

This helps the model capture commuting and daily-use patterns.

### `hour_sin` and `hour_cos`

These encode hour as a circular feature.

This is better than using only the raw hour because time wraps around:

```text
23:00 is close to 00:00
```

A model may not understand that from the number `23` alone, so sine/cosine encoding helps.

### `bearing`

The script converts negative bearing values into the `0` to `360` range.

### `bearing_sin` and `bearing_cos`

These encode trip direction as circular features.

This solves the same problem as hour encoding:

```text
0 degrees and 360 degrees represent the same direction
```

## Final output columns

The saved parquet file contains these columns:

- `rideable_type`
- `started_at`
- `ended_at`
- `start_lat`
- `start_lng`
- `end_lat`
- `end_lng`
- `member_casual`
- `city`
- `trip_duration`
- `same_station`
- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `part_of_day`
- `hour_sin`
- `hour_cos`
- `delta_lat`
- `delta_lng`
- `haversine_distance_km`
- `bearing`
- `bearing_sin`
- `bearing_cos`

The station ID columns are not included in the final output. They are only used temporarily to calculate `same_station`.

## What happens when you run the file

At the bottom:

```python
if __name__ == "__main__":
    build_city_featured_parquet("chicago", remove_duplicates=False)
    build_city_featured_parquet("nyc", remove_duplicates=False)
```

So, when you run:

```bash
python src/01_build_featured_data.py
```

the script creates two processed files:

```text
data/processed/chicago_featured.parquet
data/processed/nyc_featured.parquet
```

## Why this file matters for the thesis

This script is the bridge between raw trip records and machine-learning-ready data.

It does three important things:

1. Cleans unusable rows.
2. Adds temporal features that describe when trips happen.
3. Adds spatial features that describe how far and in what direction trips move.

These features are useful for urban mobility analysis, cross-city comparison, and later modelling steps.
