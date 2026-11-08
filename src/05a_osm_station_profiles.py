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

Πηγές OSM:
  - Σημεία ενδιαφέροντος και σταθμοί τρένου: τοπικά αρχεία .osm.pbf του Geofabrik
    (κατεβαίνουν αυτόματα μία φορά στο data/raw/osm/). Διαβάζονται κόμβοι (nodes)
    και γραμμές/περιγράμματα (ways, ως το κέντρο των κορυφών τους). Τα σύνθετα
    πολύγωνα (relations) δεν διαβάζονται· το ποσοστό τους καταγράφεται στο meta.
  - Οδικό δίκτυο για ποδήλατο: Overpass API μέσω OSMnx (ελαφρύ αίτημα).

Είσοδος : data/processed/stations_{city}.parquet          (από 02b / l1)
Έξοδοι  : data/context/osm/station_profiles_{city}.parquet
          data/context/osm/bike_graph_{city}.graphml       (ξαναχρησιμοποιείται στο 05c)
          data/context/osm/osm_{poi,transit}_{city}.parquet (cache)
          data/context/osm/osm_meta_{city}.json            (ημερομηνίες και αρχεία πηγής)
"""

import hashlib
import json
import time
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import osmium
import osmnx as ox
import pandas as pd
import requests

from config import (
    PROCESSED_DIR, OSM_DIR, PBF_DIR, CITY_CENTERS, PROFILE_RADIUS_M, OSM_BBOX_PAD_DEG,
)

CITIES = ["chicago", "nyc", "london"]

POI_KEYS = ("amenity", "shop", "office", "tourism")   # οποιαδήποτε τιμή
TRANSIT_VALUES = ("station", "halt")                  # railway=*: μετρό, προαστιακός, τρένο

# Τοπικά αρχεία OSM (Geofabrik). Κάθε πόλη διαβάζει τις πολιτείες που καλύπτουν
# τους σταθμούς της και τον κύκλο των 500 m γύρω τους, μαζί με τις γειτονικές.
GEOFABRIK = "https://download.geofabrik.de"
OSM_EXTRACTS = {
    "chicago": ["north-america/us/illinois", "north-america/us/indiana"],
    "nyc":     ["north-america/us/new-york", "north-america/us/new-jersey"],
    "london":  ["europe/united-kingdom/england/greater-london"],
}

PROFILE_COLS = [
    "dist_center_norm", "bike_station_density", "poi_density",
    "transit_density", "cycle_infra_km",
]

ox.settings.use_cache = True
ox.settings.cache_folder = str(OSM_DIR / "cache")
ox.settings.requests_timeout = 600
# Μικρότερα αιτήματα (100 km² το καθένα): ελαφρύτερα για τον server, και όσα
# πετύχουν μένουν στο cache, οπότε μια αποτυχία στη μέση δεν χάνει τα προηγούμενα.
ox.settings.max_query_area_size = 100 * 1000 * 1000

# Δημόσιοι servers Overpass (ίδια δεδομένα OSM). Σε κάθε αποτυχία δοκιμάζεται ο επόμενος.
OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
MAX_ATTEMPTS = 8
FIRST_WAIT_S = 30
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
def with_retry(func, *args):
    """Επανάληψη με αυξανόμενη αναμονή (30 s, 60 s, 120 s, ...) και εναλλαγή server,
    όπως ζητά το Overpass όταν είναι υπερφορτωμένο."""
    for attempt in range(MAX_ATTEMPTS):
        ox.settings.overpass_url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            return func(*args)
        except (ox._errors.ResponseStatusCodeError, requests.exceptions.RequestException) as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            wait = FIRST_WAIT_S * 2 ** attempt
            first_line = str(e).splitlines()[0][:120]
            print(f"  Overpass error at {ox.settings.overpass_url}: {first_line}")
            print(f"  Retrying in {wait} s (attempt {attempt + 2}/{MAX_ATTEMPTS})...")
            time.sleep(wait)


def load_bike_graph(city: str, bbox: tuple):
    path = OSM_DIR / f"bike_graph_{city}.graphml"
    if path.exists():
        print(f"Using cached bike graph: {path}")
        return ox.load_graphml(path)
    print("Downloading OSM bike network (this can take a while)...")
    G = with_retry(lambda: ox.graph_from_bbox(bbox, network_type="bike", simplify=True))
    ox.save_graphml(G, path)
    return G


def pbf_path(region: str):
    return PBF_DIR / f"{region.split('/')[-1]}-latest.osm.pbf"


def download_pbf(region: str):
    """Κατεβάζει το αρχείο του Geofabrik μία φορά (αν δεν υπάρχει ήδη)."""
    path = pbf_path(region)
    if path.exists():
        return path
    url = f"{GEOFABRIK}/{region}-latest.osm.pbf"
    print(f"Downloading {url}")
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, next_report = 0, 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done >= next_report:
                    print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                    next_report += max(total // 10, 1)
    tmp.replace(path)
    return path


def pbf_info(path) -> dict:
    reader = osmium.io.Reader(str(path), osmium.osm.osm_entity_bits.NOTHING)
    header = reader.header()
    snapshot = header.get("osmosis_replication_timestamp", "")
    reader.close()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"file": path.name, "osm_snapshot_utc": snapshot,
            "size_mb": round(path.stat().st_size / 1e6, 1), "sha256": h.hexdigest()}


def read_pbf_points(paths, bbox: tuple):
    """POI και σταθμοί τρένου μέσα στο bbox, ως σημεία (lon, lat).
    Nodes: η θέση τους. Ways: το κέντρο των κορυφών τους. Relations: παραλείπονται.
    Τα αρχεία γειτονικών πολιτειών επικαλύπτονται στα όρια, οπότε κάθε αντικείμενο
    μετρά μία φορά (με βάση τύπο και id)."""
    left, bottom, right, top = bbox
    poi, transit, seen = [], [], set()
    n_poi_file, n_rel_poi_file = 0, 0
    for path in paths:
        print(f"Reading {path.name} ...")
        fp = (osmium.FileProcessor(str(path))
              .with_locations()
              .with_filter(osmium.filter.KeyFilter(*POI_KEYS, "railway")))
        for obj in fp:
            tags = obj.tags
            is_poi = any(k in tags for k in POI_KEYS)
            if obj.is_relation():
                if is_poi:
                    n_rel_poi_file += 1
                continue
            key = ("n" if obj.is_node() else "w", obj.id)
            if key in seen:
                continue
            seen.add(key)
            if obj.is_node():
                if not obj.location.valid():
                    continue
                lon, lat = obj.location.lon, obj.location.lat
            else:
                locs = [n.location for n in obj.nodes if n.location.valid()]
                if not locs:
                    continue
                lon = sum(l.lon for l in locs) / len(locs)
                lat = sum(l.lat for l in locs) / len(locs)
            if is_poi:
                n_poi_file += 1
            if not (left <= lon <= right and bottom <= lat <= top):
                continue
            if is_poi:
                poi.append((lon, lat))
            if obj.is_node() and tags.get("railway") in TRANSIT_VALUES:
                transit.append((lon, lat))

    def to_gdf(pts):
        if not pts:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        a = np.array(pts)
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(a[:, 0], a[:, 1]), crs="EPSG:4326")

    rel_share = n_rel_poi_file / max(n_poi_file + n_rel_poi_file, 1)
    return to_gdf(poi), to_gdf(transit), rel_share


def load_points(city: str, bbox: tuple):
    """POI και σταθμοί τρένου για την πόλη: από το cache, ή από τα αρχεία του Geofabrik."""
    poi_path = OSM_DIR / f"osm_poi_{city}.parquet"
    tr_path = OSM_DIR / f"osm_transit_{city}.parquet"
    if poi_path.exists() and tr_path.exists():
        print(f"Using cached OSM points: {poi_path.name}, {tr_path.name}")
        return gpd.read_parquet(poi_path), gpd.read_parquet(tr_path), None

    paths = [download_pbf(r) for r in OSM_EXTRACTS[city]]
    pois, transit, rel_share = read_pbf_points(paths, bbox)
    pois.to_parquet(poi_path)
    transit.to_parquet(tr_path)
    source = {"pbf_files": [pbf_info(p) for p in paths],
              "poi_relations_skipped_share": round(rel_share, 4)}
    return pois, transit, source


def write_meta(city: str, bbox: tuple, G, pois, transit, source) -> None:
    path = OSM_DIR / f"osm_meta_{city}.json"
    if path.exists() or source is None:
        return   # οι αρχικές πληροφορίες λήψης μένουν
    meta = {
        "city": city,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "osmnx_version": ox.__version__,
        "bbox_left_bottom_right_top": bbox,
        "bike_graph_source": "Overpass API (OSMnx)",
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "points_source": "Geofabrik .osm.pbf",
        **source,
        "n_poi_in_bbox": len(pois),
        "n_transit_in_bbox": len(transit),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OSM snapshot: {[f['osm_snapshot_utc'] for f in source['pbf_files']]}, "
          f"POI relations skipped: {100 * source['poi_relations_skipped_share']:.2f}%")


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
    pois, transit, source = load_points(city, bbox)
    write_meta(city, bbox, G, pois, transit, source)

    prof = compute_profiles(city, st, G, pois, transit)
    out_path = OSM_DIR / f"station_profiles_{city}.parquet"
    prof.to_parquet(out_path, index=False)
    summary(city, prof)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    PBF_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        build_city(city)
