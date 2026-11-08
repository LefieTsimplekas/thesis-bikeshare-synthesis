"""
05a_osm_station_profiles.py

Προφίλ αστικού περιβάλλοντος για κάθε φυσικό σταθμό των τριών πόλεων (§3.1).

Χαρακτηριστικά (ακτίνα PROFILE_RADIUS_M γύρω από κάθε σταθμό):
  dist_center_norm      : απόσταση από το κέντρο / διάμεση απόσταση των σταθμών της πόλης
  bike_station_density  : άλλοι σταθμοί ποδηλάτου ανά km² (από τον πίνακα σταθμών)
  poi_density           : σημεία ενδιαφέροντος OSM ανά km²
  transit_density       : σταθμοί μετρό/τρένου OSM ανά km²
  cycle_infra_km        : μήκος ποδηλατικής υποδομής (km)
Κρατά επίσης το dist_center_km (ακατέργαστο) για έλεγχο.

Μόνο σταθμοί με διαδρομές το 2025 (n_trips > 0).
Δεν γίνεται εδώ καμία κανονικοποίηση μεταξύ πόλεων: η τυποποίηση και οι τύποι
σταθμών ορίζονται στο 05b, μόνο από τις πόλεις-πηγές.

Είσοδος : data/processed/stations_{city}.parquet          (από 02b / l1)
Έξοδοι  : data/context/osm/station_profiles_{city}.parquet
          data/context/osm/bike_graph_{city}.graphml       (ξαναχρησιμοποιείται στο 05c)
          data/context/osm/osm_{poi,transit}_{city}.parquet (cache)
          data/context/osm/osm_meta_{city}.json            (ημερομηνία λήψης κ.λπ.)
"""

import json
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from config import (
    PROCESSED_DIR, OSM_DIR, CITY_CENTERS, PROFILE_RADIUS_M, OSM_BBOX_PAD_DEG,
)

CITIES = ["chicago", "nyc", "london"]

POI_TAGS = {"amenity": True, "shop": True, "office": True, "tourism": True}
TRANSIT_TAGS = {"railway": ["station", "halt"]}   # μετρό, προαστιακός, τρένο

PROFILE_COLS = [
    "dist_center_norm", "bike_station_density", "poi_density",
    "transit_density", "cycle_infra_km",
]

ox.settings.use_cache = True
ox.settings.cache_folder = str(OSM_DIR / "cache")
ox.settings.requests_timeout = 600
ox.settings.useful_tags_way = list(
    set(ox.settings.useful_tags_way)
    | {"cycleway", "cycleway:left", "cycleway:right", "cycleway:both", "bicycle"}
)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Είσοδος: πίνακας φυσικών σταθμών
# ---------------------------------------------------------------------------
def load_stations(city: str) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED_DIR / f"stations_{city}.parquet")
    n_all = len(st)
    st = st[st["n_trips"] > 0].reset_index(drop=True)
    print(f"Stations: {len(st):,} with 2025 trips (of {n_all:,} in table)")
    return st[["station_key", "lat", "lng", "n_trips"]]


def city_bbox(st: pd.DataFrame) -> tuple:
    # OSMnx 2.x: (left, bottom, right, top)
    p = OSM_BBOX_PAD_DEG
    return (st.lng.min() - p, st.lat.min() - p, st.lng.max() + p, st.lat.max() + p)


# ---------------------------------------------------------------------------
# Λήψη OSM (με cache στον δίσκο)
# ---------------------------------------------------------------------------
def load_bike_graph(city: str, bbox: tuple):
    path = OSM_DIR / f"bike_graph_{city}.graphml"
    if path.exists():
        print(f"Using cached bike graph: {path}")
        return ox.load_graphml(path)
    print("Downloading OSM bike network (this can take a while)...")
    G = ox.graph_from_bbox(bbox, network_type="bike", simplify=True)
    ox.save_graphml(G, path)
    return G


def fetch_features(city: str, name: str, tags: dict, bbox: tuple) -> gpd.GeoDataFrame:
    path = OSM_DIR / f"osm_{name}_{city}.parquet"
    if path.exists():
        print(f"Using cached OSM {name}: {path}")
        return gpd.read_parquet(path)
    print(f"Downloading OSM {name}...")
    try:
        g = ox.features_from_bbox(bbox, tags)
        g = g[["geometry"]].reset_index(drop=True)
    except ox._errors.InsufficientResponseError:
        g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g.to_parquet(path)
    return g


def write_meta(city: str, bbox: tuple, G, pois, transit) -> None:
    path = OSM_DIR / f"osm_meta_{city}.json"
    if path.exists():
        return   # η αρχική ημερομηνία λήψης μένει
    meta = {
        "city": city,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "osmnx_version": ox.__version__,
        "bbox_left_bottom_right_top": bbox,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "n_poi": len(pois),
        "n_transit": len(transit),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Χαρακτηριστικά
# ---------------------------------------------------------------------------
def count_within(buffers: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> pd.Series:
    if points.empty:
        return pd.Series(0, index=buffers["station_key"])
    j = gpd.sjoin(points[["geometry"]], buffers[["station_key", "geometry"]], predicate="within")
    return j.groupby("station_key").size().reindex(buffers["station_key"], fill_value=0)


def cycle_infra_km(buffers: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> pd.Series:
    hw = edges["highway"].astype(str)
    mask = hw.str.contains("cycleway")
    for c in ["cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]:
        if c in edges:
            mask |= edges[c].astype(str).str.contains("lane|track|shared_lane|separate", regex=True)
    if "bicycle" in edges:
        mask |= edges["bicycle"].astype(str).str.contains("designated")
    cyc = edges.loc[mask, ["geometry"]].reset_index(drop=True)
    if cyc.empty:
        return pd.Series(0.0, index=buffers["station_key"])
    inter = gpd.overlay(cyc, buffers[["station_key", "geometry"]], how="intersection", keep_geom_type=True)
    inter["len_m"] = inter.length
    return (inter.groupby("station_key")["len_m"].sum() / 1000).reindex(buffers["station_key"], fill_value=0.0)


def compute_profiles(city: str, st: pd.DataFrame, G, pois: gpd.GeoDataFrame,
                     transit: gpd.GeoDataFrame) -> pd.DataFrame:
    gst = gpd.GeoDataFrame(st.copy(), geometry=gpd.points_from_xy(st.lng, st.lat), crs="EPSG:4326")
    crs = gst.estimate_utm_crs()
    gst = gst.to_crs(crs)
    buffers = gst[["station_key"]].copy()
    buffers = gpd.GeoDataFrame(buffers, geometry=gst.buffer(PROFILE_RADIUS_M), crs=crs)
    area_km2 = np.pi * PROFILE_RADIUS_M ** 2 / 1e6

    # POI: πολύγωνα -> κεντροειδές, σε προβολή
    poi_pts = pois.to_crs(crs)
    poi_pts = gpd.GeoDataFrame(geometry=poi_pts.geometry.centroid, crs=crs)

    # Σταθμοί τρένου/μετρό: μόνο σημεία, για να μη μετρηθεί δύο φορές ο ίδιος σταθμός
    # (χαρτογραφημένος και ως σημείο και ως περίγραμμα)
    tr = transit.to_crs(crs)
    tr = tr[tr.geom_type == "Point"]

    _, edges = ox.graph_to_gdfs(ox.convert.to_undirected(G))
    edges = edges.to_crs(crs)

    out = st.set_index("station_key").copy()
    out["dist_center_km"] = haversine_km(out.lat, out.lng, *CITY_CENTERS[city])
    out["dist_center_norm"] = out["dist_center_km"] / out["dist_center_km"].median()
    # άλλοι σταθμοί μέσα στην ακτίνα (χωρίς τον ίδιο)
    out["bike_station_density"] = (count_within(buffers, gst[["geometry"]]) - 1) / area_km2
    out["poi_density"] = count_within(buffers, poi_pts) / area_km2
    out["transit_density"] = count_within(buffers, tr) / area_km2
    out["cycle_infra_km"] = cycle_infra_km(buffers, edges)

    out = out.reset_index()
    out.insert(1, "city", city)
    return out


def summary(city: str, prof: pd.DataFrame) -> None:
    print(f"\n--- Profile summary: {city} ({len(prof):,} stations) ---")
    desc = prof[["dist_center_km"] + PROFILE_COLS].describe(percentiles=[0.05, 0.5, 0.95]).T
    print(desc[["5%", "50%", "95%", "max"]].round(2).to_string())
    print("Share of stations with zero value:")
    for c in ["poi_density", "transit_density", "cycle_infra_km", "bike_station_density"]:
        print(f"  {c:<22} {100 * (prof[c] == 0).mean():.1f}%")


# ---------------------------------------------------------------------------
def build_city(city: str) -> None:
    print(f"\n===== {city} =====")
    st = load_stations(city)
    bbox = city_bbox(st)

    G = load_bike_graph(city, bbox)
    pois = fetch_features(city, "poi", POI_TAGS, bbox)
    transit = fetch_features(city, "transit", TRANSIT_TAGS, bbox)
    write_meta(city, bbox, G, pois, transit)

    prof = compute_profiles(city, st, G, pois, transit)
    out_path = OSM_DIR / f"station_profiles_{city}.parquet"
    prof.to_parquet(out_path, index=False)
    summary(city, prof)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        build_city(city)
