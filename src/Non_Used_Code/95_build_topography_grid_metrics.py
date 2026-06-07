from pathlib import Path


"""
Build rectangular-grid topography metrics for each city.

This script expects:
- grid definitions from 03_build_urban_grid_metrics.py
- a DEM raster per city in data/context/topography/raw/

Expected raw inputs:
- data/context/topography/raw/chicago_dem.tif
- data/context/topography/raw/nyc_dem.tif

Outputs:
- data/context/topography/chicago_grid_topography.parquet
- data/context/topography/nyc_grid_topography.parquet

Requirements:
- geopandas
- rasterio
- shapely
- pyproj
"""


URBAN_FORM_DIR = Path("data/context/urban_form")
TOPOGRAPHY_RAW_DIR = Path("data/context/topography/raw")
TOPOGRAPHY_OUTPUT_DIR = Path("data/context/topography")

CITY_FILES = {
    "chicago": "chicago_dem.tif",
    "nyc": "nyc_dem.tif",
}


def require_topography_dependencies():
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import box
    except ImportError as exc:
        raise ImportError(
            "This script requires geopandas, rasterio, shapely, pyproj, numpy, and pandas."
        ) from exc

    return gpd, np, pd, rasterio, mask, box


def build_city_topography(city_name: str, dem_filename: str) -> None:
    gpd, np, pd, rasterio, mask, box = require_topography_dependencies()

    TOPOGRAPHY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_path = URBAN_FORM_DIR / f"{city_name}_grid_metrics.parquet"
    dem_path = TOPOGRAPHY_RAW_DIR / dem_filename
    output_path = TOPOGRAPHY_OUTPUT_DIR / f"{city_name}_grid_topography.parquet"

    if not grid_path.exists():
        raise FileNotFoundError(f"Urban grid metrics not found: {grid_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    print(f"\n{'=' * 60}")
    print(f"Building topography grid metrics: {city_name}")
    print(f"{'=' * 60}")
    print(f"Grid source: {grid_path}")
    print(f"DEM source: {dem_path}")

    grid_df = pd.read_parquet(grid_path)
    projected_crs = grid_df["projected_crs"].dropna().iloc[0]
    grid_df["geometry"] = grid_df.apply(
        lambda row: box(
            row["cell_min_x"],
            row["cell_min_y"],
            row["cell_max_x"],
            row["cell_max_y"],
        ),
        axis=1,
    )

    with rasterio.open(dem_path) as src:
        grid_gdf = gpd.GeoDataFrame(grid_df, geometry="geometry", crs=projected_crs)
        if src.crs is not None and str(src.crs) != str(projected_crs):
            grid_gdf = grid_gdf.to_crs(src.crs)
        results = []

        for _, row in grid_gdf.iterrows():
            try:
                clipped, transform = mask(src, [row.geometry], crop=True, filled=False)
                elevation = clipped[0]
                elevation = elevation.compressed()
            except ValueError:
                elevation = np.array([])

            if elevation.size == 0:
                mean_elevation_m = float("nan")
                mean_slope_deg = float("nan")
            else:
                mean_elevation_m = float(elevation.mean())

                try:
                    slope_y, slope_x = np.gradient(
                        clipped[0].filled(np.nan),
                        src.res[1],
                        src.res[0],
                    )
                    slope_rad = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
                    slope_deg = np.degrees(slope_rad)
                    mean_slope_deg = float(np.nanmean(slope_deg))
                except Exception:
                    mean_slope_deg = float("nan")

            results.append(
                {
                    "cell_id": row["cell_id"],
                    "city": city_name,
                    "mean_elevation_m": mean_elevation_m,
                    "mean_slope_deg": mean_slope_deg,
                }
            )

    pd.DataFrame(results).to_parquet(output_path, index=False)
    print(f"Saved topography metrics to: {output_path}")


def main() -> None:
    for city_name, dem_filename in CITY_FILES.items():
        build_city_topography(city_name, dem_filename)


if __name__ == "__main__":
    main()
