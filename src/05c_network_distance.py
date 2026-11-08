"""
05c_network_distance.py

Δικτυακή απόσταση (πάνω στο οδικό δίκτυο ποδηλάτου του OSM) για κάθε ζεύγος
φυσικών σταθμών αφετηρίας-προορισμού που εμφανίζεται στα δεδομένα (§3.3).

Γιατί: η ευθεία απόσταση (haversine) αγνοεί ποτάμια, γέφυρες, μονοδρόμους και
το σχήμα του οδικού δικτύου. Η δικτυακή απόσταση είναι πιο κοντά στη διαδρομή
που κάνει πραγματικά ο ποδηλάτης, και ο λόγος τους (circuity) περιγράφει πόσο
«ευθύ» είναι το δίκτυο κάθε πόλης.

Βήματα (ανά πόλη):
  1. Φόρτωση του bike_graph_{city}.graphml (05a). Κατευθυνόμενο γράφημα: τηρούνται
     οι μονόδρομοι όπως τους κωδικοποιεί το OSMnx για network_type="bike".
     Ανάμεσα σε παράλληλες ακμές κρατιέται η μικρότερη.
  2. Περιορισμός στη μεγαλύτερη ισχυρά συνεκτική συνιστώσα, ώστε από κάθε κόμβο
     να υπάρχει διαδρομή προς κάθε άλλον.
  3. Κάθε σταθμός «κουμπώνει» στον πλησιέστερο κόμβο της συνιστώσας (θέση από τον
     πίνακα σταθμών του 02b/l1). Η απόσταση σταθμού-κόμβου προστίθεται στα δύο άκρα.
  4. Ζεύγη: όλα τα (start_station_key, end_station_key) των train/val (και του
     test για το Λονδίνο). Μία φορά ανά ζεύγος, όχι ανά διαδρομή.
  5. Dijkstra (scipy, σε παρτίδες αφετηριών) από κάθε σταθμό αφετηρίας.
  6. Κυκλικές διαδρομές (ίδιος σταθμός): δεν έχουν δικτυακή απόσταση -> NaN.
  7. circuity = δικτυακή / ευθεία απόσταση σταθμών (θέσεις του πίνακα σταθμών).
  8. Προσθήκη των τύπων σταθμών του 05b (start_type, end_type).

Τα αρχεία των splits ΔΕΝ αλλάζουν (το london_test είναι κλειδωμένο με SHA-256).
Οι πίνακες ζευγών συνδέονται με τις διαδρομές στο 06.

Είσοδοι : data/context/osm/bike_graph_{city}.graphml         (05a)
          data/processed/stations_{city}.parquet             (02b / l1)
          data/processed/{city}_{train,val}.parquet, london_test.parquet
          data/context/osm/station_types.parquet             (05b)
Έξοδοι  : data/context/osm/od_network_{city}.parquet
            start_station_key, end_station_key, network_distance_km,
            station_distance_km, circuity, start_type, end_type
          artifacts/network_distance/network_summary.csv
          artifacts/network_distance/snap_{city}.csv         (απόσταση σταθμού-κόμβου)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.neighbors import BallTree

from config import PROCESSED_DIR, OSM_DIR, NETWORK_MAX_KM, MAX_SNAP_WARN_M

CITIES = ["chicago", "nyc", "london"]
SPLITS = {
    "chicago": ["train", "val"],
    "nyc": ["train", "val"],
    "london": ["train", "val", "test"],
}
EARTH_R_M = 6_371_008.8
BATCH = 64                      # αφετηρίες ανά κλήση Dijkstra (μνήμη: BATCH x κόμβοι)

ARTIFACT_DIR = Path("artifacts/network_distance")


def haversine_m(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Γράφημα
# ---------------------------------------------------------------------------
def load_graph(city: str):
    """Επιστρέφει πίνακα γειτνίασης (CSR, μέτρα) της μεγαλύτερης ισχυρά συνεκτικής
    συνιστώσας και τις συντεταγμένες των κόμβων της."""
    G = ox.load_graphml(OSM_DIR / f"bike_graph_{city}.graphml")
    nodes = np.fromiter(G.nodes, dtype=np.int64)
    pos = {n: i for i, n in enumerate(nodes)}
    lat = np.array([G.nodes[n]["y"] for n in nodes])
    lng = np.array([G.nodes[n]["x"] for n in nodes])

    e = pd.DataFrame([(pos[u], pos[v], float(d["length"]))
                      for u, v, d in G.edges(data=True) if u != v],
                     columns=["u", "v", "length"])
    e = e.groupby(["u", "v"], as_index=False)["length"].min()   # παράλληλες ακμές
    A = csr_matrix((e.length, (e.u, e.v)), shape=(len(nodes), len(nodes)))

    _, comp = connected_components(A, directed=True, connection="strong")
    main = np.flatnonzero(comp == np.bincount(comp).argmax())
    A = A[main][:, main]
    print(f"Graph: {len(nodes):,} nodes, {len(e):,} edges; "
          f"largest strongly connected component {len(main):,} nodes "
          f"({100 * len(main) / len(nodes):.1f}%)")
    return A, lat[main], lng[main]


def snap_stations(st: pd.DataFrame, lat: np.ndarray, lng: np.ndarray) -> pd.DataFrame:
    tree = BallTree(np.radians(np.c_[lat, lng]), metric="haversine")
    d, i = tree.query(np.radians(st[["lat", "lng"]].to_numpy()), k=1)
    st = st.copy()
    st["node"] = i[:, 0]
    st["snap_m"] = d[:, 0] * EARTH_R_M
    return st


# ---------------------------------------------------------------------------
# Ζεύγη και αποστάσεις
# ---------------------------------------------------------------------------
def observed_pairs(city: str) -> pd.DataFrame:
    cols = ["start_station_key", "end_station_key"]
    parts = [pd.read_parquet(PROCESSED_DIR / f"{city}_{s}.parquet", columns=cols)
             for s in SPLITS[city]]
    return pd.concat(parts).drop_duplicates().reset_index(drop=True)


def network_distances(A, pairs: pd.DataFrame, st: pd.DataFrame) -> np.ndarray:
    """Δικτυακή απόσταση (km) για κάθε ζεύγος, με τις αποστάσεις κουμπώματος."""
    node = st["node"].to_dict()
    snap = st["snap_m"].to_dict()
    o_node = pairs.start_station_key.map(node).to_numpy()
    d_node = pairs.end_station_key.map(node).to_numpy()
    out = np.full(len(pairs), np.nan)

    origins = np.unique(o_node)
    for b in range(0, len(origins), BATCH):
        batch = origins[b:b + BATCH]
        dist = dijkstra(A, directed=True, indices=batch, limit=NETWORK_MAX_KM * 1000)
        row_of = {n: r for r, n in enumerate(batch)}
        m = np.isin(o_node, batch)
        rows = np.array([row_of[n] for n in o_node[m]])
        out[m] = dist[rows, d_node[m]]
        print(f"  origins {min(b + BATCH, len(origins)):,}/{len(origins):,}", end="\r")
    print()

    out[np.isinf(out)] = np.nan
    snap_total = (pairs.start_station_key.map(snap) + pairs.end_station_key.map(snap)).to_numpy()
    return (out + snap_total) / 1000


# ---------------------------------------------------------------------------
def build_city(city: str, types: pd.DataFrame) -> dict:
    print(f"\n===== {city} =====")
    A, lat, lng = load_graph(city)

    st = pd.read_parquet(PROCESSED_DIR / f"stations_{city}.parquet",
                         columns=["station_key", "lat", "lng"])
    pairs = observed_pairs(city)
    used = pd.unique(pairs[["start_station_key", "end_station_key"]].to_numpy().ravel())
    st = snap_stations(st[st.station_key.isin(used)], lat, lng).set_index("station_key")

    missing = set(used) - set(st.index)
    if missing:
        raise ValueError(f"{len(missing)} station keys without position, e.g. {list(missing)[:5]}")

    far = st[st.snap_m > MAX_SNAP_WARN_M]
    print(f"Stations: {len(st):,}; snap distance median {st.snap_m.median():.0f} m, "
          f"p99 {st.snap_m.quantile(0.99):.0f} m; > {MAX_SNAP_WARN_M} m: {len(far)}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    st[["lat", "lng", "snap_m"]].sort_values("snap_m", ascending=False) \
        .to_csv(ARTIFACT_DIR / f"snap_{city}.csv")

    # Ευθεία απόσταση σταθμών, από τις θέσεις του πίνακα σταθμών
    o = st.loc[pairs.start_station_key, ["lat", "lng"]].to_numpy()
    d = st.loc[pairs.end_station_key, ["lat", "lng"]].to_numpy()
    pairs["station_distance_km"] = haversine_m(o[:, 0], o[:, 1], d[:, 0], d[:, 1]) / 1000

    round_trip = (pairs.start_station_key == pairs.end_station_key).to_numpy()
    nd = np.full(len(pairs), np.nan)
    nd[~round_trip] = network_distances(A, pairs[~round_trip], st)
    pairs["network_distance_km"] = nd
    pairs["circuity"] = np.where(pairs.station_distance_km > 0,
                                 pairs.network_distance_km / pairs.station_distance_km, np.nan)

    t = types[types.city == city].set_index("station_key")["origin_type"]
    pairs["start_type"] = pairs.start_station_key.map(t).astype("Int8")
    pairs["end_type"] = pairs.end_station_key.map(t).astype("Int8")

    out = OSM_DIR / f"od_network_{city}.parquet"
    pairs.to_parquet(out, index=False)

    nr = pairs[~round_trip]
    unreachable = int(nr.network_distance_km.isna().sum())
    summary = {
        "city": city,
        "stations": len(st),
        "od_pairs": len(pairs),
        "round_trip_pairs": int(round_trip.sum()),
        "unreachable_pairs": unreachable,
        "snap_median_m": round(float(st.snap_m.median()), 1),
        "snap_over_warn": len(far),
        "circuity_median": round(float(nr.circuity.median()), 3),
        "circuity_p05": round(float(nr.circuity.quantile(0.05)), 3),
        "circuity_p95": round(float(nr.circuity.quantile(0.95)), 3),
        "pairs_circuity_below_1": int((nr.circuity < 1).sum()),
        "types_missing": int(pairs.start_type.isna().sum() + pairs.end_type.isna().sum()),
    }

    # Έλεγχος χρησιμότητας: συσχέτιση με τη διάρκεια, στις διαδρομές train
    tr = pd.read_parquet(PROCESSED_DIR / f"{city}_train.parquet",
                         columns=["start_station_key", "end_station_key",
                                  "trip_duration", "haversine_distance_km"])
    tr = tr[tr.start_station_key != tr.end_station_key].merge(
        pairs[["start_station_key", "end_station_key", "network_distance_km"]],
        on=["start_station_key", "end_station_key"], how="left")
    summary["spearman_duration_haversine"] = round(float(
        tr.trip_duration.corr(tr.haversine_distance_km, method="spearman")), 4)
    summary["spearman_duration_network"] = round(float(
        tr.trip_duration.corr(tr.network_distance_km, method="spearman")), 4)

    print(f"OD pairs: {len(pairs):,} ({int(round_trip.sum()):,} round trip); "
          f"unreachable: {unreachable}")
    print(f"Circuity: median {summary['circuity_median']}, "
          f"p05 {summary['circuity_p05']}, p95 {summary['circuity_p95']}, "
          f"< 1: {summary['pairs_circuity_below_1']}")
    print(f"Spearman with duration (train, non-round trips): "
          f"haversine {summary['spearman_duration_haversine']}, "
          f"network {summary['spearman_duration_network']}")
    print(f"Saved to {out}")
    return summary


if __name__ == "__main__":
    types = pd.read_parquet(OSM_DIR / "station_types.parquet")
    rows = [build_city(c, types) for c in CITIES]
    s = pd.DataFrame(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(ARTIFACT_DIR / "network_summary.csv", index=False)
    print("\n" + s.T.to_string(header=False))
