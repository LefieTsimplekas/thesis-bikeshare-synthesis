from pathlib import Path


"""
Build rectangular-grid urban-form metrics for each city.

This script downloads the street network for each city from OpenStreetMap
using OSMnx, creates a rectangular grid, and computes per-cell metrics such as:
- intersection_density
- road_density

Outputs:
- data/context/urban_form/chicago_grid_metrics.parquet
- data/context/urban_form/nyc_grid_metrics.parquet

Requirements:
- osmnx
- geopandas
- shapely
- pyproj

Install example:
    pip install osmnx geopandas shapely pyproj
"""


CONTEXT_DIR = Path("data/context/urban_form")
GRID_SIZE_METERS = 500

CITY_QUERIES = {
    "chicago": "Chicago, Illinois, USA",
    "nyc": "New York City, New York, USA",
}


def require_geospatial_dependencies():
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError as exc:
        raise ImportError(
            "This script requires osmnx, geopandas, shapely, and pyproj. "
            "Install them first, then rerun the script."
        ) from exc

    return ox, gpd, box


def build_rectangular_grid(area_gdf, grid_size_meters: int, box):
    minx, miny, maxx, maxy = area_gdf.total_bounds

    cells = []
    x = minx
    x_idx = 0

    while x < maxx:
        y = miny
        y_idx = 0
        while y < maxy:
            cells.append(
                {
                    "grid_x": x_idx,
                    "grid_y": y_idx,
                    "geometry": box(x, y, x + grid_size_meters, y + grid_size_meters),
                }
            )
            y += grid_size_meters
            y_idx += 1
        x += grid_size_meters
        x_idx += 1

    return cells


def build_city_metrics(city_key: str, city_query: str) -> None:
    ox, gpd, box = require_geospatial_dependencies()

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONTEXT_DIR / f"{city_key}_grid_metrics.parquet"

    print(f"\n{'=' * 60}")
    print(f"Building urban grid metrics: {city_key}")
    print(f"{'=' * 60}")
    print(f"Place query: {city_query}")
    print(f"Grid size: {GRID_SIZE_METERS}m x {GRID_SIZE_METERS}m")

    area_gdf = ox.geocode_to_gdf(city_query)
    projected_crs = area_gdf.estimate_utm_crs()
    area_gdf = area_gdf.to_crs(projected_crs)

    graph = ox.graph_from_place(city_query, network_type="bike", simplify=True)
    graph = ox.project_graph(graph, to_crs=projected_crs)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    grid_records = build_rectangular_grid(area_gdf, GRID_SIZE_METERS, box)
    grid_gdf = gpd.GeoDataFrame(grid_records, crs=projected_crs)
    grid_gdf = gpd.overlay(grid_gdf, area_gdf[["geometry"]], how="intersection")

    intersections = nodes_gdf[nodes_gdf["street_count"].fillna(0) >= 3].copy()

    edges_join = gpd.sjoin(edges_gdf, grid_gdf, predicate="intersects", how="inner")
    nodes_join = gpd.sjoin(intersections, grid_gdf, predicate="within", how="inner")

    edge_lengths = (
        edges_join.groupby(["grid_x", "grid_y"])["length"]
        .sum()
        .reset_index(name="road_length_m")
    )
    node_counts = (
        nodes_join.groupby(["grid_x", "grid_y"])
        .size()
        .reset_index(name="intersection_count")
    )

    metrics_gdf = grid_gdf.copy()
    metrics_gdf["cell_area_km2"] = metrics_gdf.geometry.area / 1_000_000

    metrics_gdf = metrics_gdf.merge(edge_lengths, on=["grid_x", "grid_y"], how="left")
    metrics_gdf = metrics_gdf.merge(node_counts, on=["grid_x", "grid_y"], how="left")

    metrics_gdf["road_length_m"] = metrics_gdf["road_length_m"].fillna(0.0)
    metrics_gdf["intersection_count"] = metrics_gdf["intersection_count"].fillna(0)
    metrics_gdf["intersection_density"] = (
        metrics_gdf["intersection_count"] / metrics_gdf["cell_area_km2"]
    )
    metrics_gdf["road_density"] = (
        metrics_gdf["road_length_m"] / metrics_gdf["cell_area_km2"]
    )
    bounds_df = metrics_gdf.geometry.bounds.rename(
        columns={
            "minx": "cell_min_x",
            "miny": "cell_min_y",
            "maxx": "cell_max_x",
            "maxy": "cell_max_y",
        }
    )
    metrics_gdf = metrics_gdf.join(bounds_df)
    metrics_gdf["cell_id"] = (
        city_key
        + "_x"
        + metrics_gdf["grid_x"].astype(str)
        + "_y"
        + metrics_gdf["grid_y"].astype(str)
    )
    metrics_gdf["city"] = city_key
    metrics_gdf["grid_size_meters"] = GRID_SIZE_METERS
    metrics_gdf["projected_crs"] = str(projected_crs)

    output_columns = [
        "cell_id",
        "city",
        "grid_x",
        "grid_y",
        "grid_size_meters",
        "projected_crs",
        "cell_min_x",
        "cell_min_y",
        "cell_max_x",
        "cell_max_y",
        "cell_area_km2",
        "intersection_count",
        "road_length_m",
        "intersection_density",
        "road_density",
    ]

    metrics_gdf[output_columns].to_parquet(output_path, index=False)

    print(f"Saved urban metrics to: {output_path}")
    print(metrics_gdf[output_columns].head())


def main() -> None:
    for city_key, city_query in CITY_QUERIES.items():
        build_city_metrics(city_key, city_query)


if __name__ == "__main__":
    main()
