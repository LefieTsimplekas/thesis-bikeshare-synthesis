"""
02d_merge_radius_check.py

Έλεγχος του ορίου ένωσης κωδικών σε φυσικούς σταθμούς (MERGE_RADIUS_M = 25 m, 02b).
Δεν αλλάζει κανένα αρχείο της ροής: μόνο τεκμηριώνει ότι το όριο δεν είναι αυθαίρετο.

Το όριο πρέπει να είναι
  - μεγαλύτερο από τις αποστάσεις ανάμεσα σε κωδικούς του ίδιου σταθμού
    (αλλαγή αρίθμησης, διαφορετική μορφοποίηση του ίδιου κωδικού),
  - μικρότερο από τις αποστάσεις ανάμεσα σε διαφορετικούς σταθμούς,
και το αποτέλεσμα να μην αλλάζει πολύ γύρω του.

Τι υπολογίζει, ανά πόλη:
  1. Ιστόγραμμα των αποστάσεων ανάμεσα σε κάθε ζεύγος κωδικών έως 100 m (κάδοι 5 m):
     ένα «κενό» ανάμεσα στις πολύ μικρές αποστάσεις (ίδιος σταθμός) και στις
     μεγαλύτερες (γειτονικοί σταθμοί) δείχνει πού πρέπει να πέσει το όριο.
  2. Ευαισθησία: πλήθος φυσικών σταθμών και ενωμένων κωδικών για όρια 5–100 m.
  3. Απόσταση κάθε φυσικού σταθμού (με 25 m) από τον πλησιέστερο ΑΛΛΟ σταθμό:
     ελάχιστη, 1ο, 5ο εκατοστημόριο και διάμεσος.
  Για σύγκριση, το (3) και για τους σταθμούς του Λονδίνου (BikePoint), όπου κάθε
  σταθμός έχει επίσημη θέση.

Είσοδοι : data/processed/station_ids_{chicago,nyc}.parquet   (02b)
          data/processed/stations_london.parquet              (l1)
Έξοδοι  : artifacts/station_table/merge_radius_pairs.csv        (ιστόγραμμα)
          artifacts/station_table/merge_radius_sensitivity.csv  (ευαισθησία)
          artifacts/station_table/nearest_station.csv           (πλησιέστερος σταθμός)

Χρήση:  python src/02d_merge_radius_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import MERGE_RADIUS_M, PROCESSED_DIR

OUT_DIR = Path("artifacts/station_table")
EARTH_M = 6_371_000.0
RADII = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
BINS = list(range(0, 105, 5))


def xyz(lat, lng) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lng, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def chord(m: float) -> float:
    """Απόσταση πάνω στη σφαίρα (m) -> ευθεία χορδή στη μοναδιαία σφαίρα."""
    return 2 * np.sin(m / EARTH_M / 2)


def arc_m(c: np.ndarray) -> np.ndarray:
    return 2 * np.arcsin(np.clip(c / 2, 0, 1)) * EARTH_M


def pairs_within(pts: np.ndarray, m: float) -> np.ndarray:
    return cKDTree(pts).query_pairs(chord(m), output_type="ndarray")


def n_groups(n: int, pairs: np.ndarray) -> int:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return len({find(i) for i in range(n)})


def nearest_other(pts: np.ndarray) -> np.ndarray:
    d, _ = cKDTree(pts).query(pts, k=2)
    return arc_m(d[:, 1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_rows, sens_rows, near_rows = [], [], []

    for city in ["chicago", "nyc"]:
        ids = pd.read_parquet(PROCESSED_DIR / f"station_ids_{city}.parquet")
        ids = ids.dropna(subset=["lat", "lng"]).reset_index(drop=True)
        pts = xyz(ids.lat, ids.lng)

        # 1. Αποστάσεις ζευγών κωδικών έως 100 m
        p = pairs_within(pts, BINS[-1])
        d = arc_m(np.linalg.norm(pts[p[:, 0]] - pts[p[:, 1]], axis=1)) if len(p) else np.array([])
        counts, _ = np.histogram(d, bins=BINS)
        for lo, c in zip(BINS[:-1], counts):
            hist_rows.append({"city": city, "from_m": lo, "to_m": lo + 5, "id_pairs": int(c)})

        # 2. Ευαισθησία στο όριο
        for r in RADII:
            g = n_groups(len(ids), pairs_within(pts, r))
            sens_rows.append({"city": city, "radius_m": r, "ids": len(ids),
                              "physical_stations": g, "ids_merged": len(ids) - g})

        # 3. Πλησιέστερος άλλος σταθμός, με τους σταθμούς του 02b (25 m)
        st = pd.read_parquet(PROCESSED_DIR / f"stations_{city}.parquet")
        near_rows.append({"city": city, "stations": len(st),
                          **describe(nearest_other(xyz(st.lat, st.lng)))})

    lon = pd.read_parquet(PROCESSED_DIR / "stations_london.parquet").dropna(subset=["lat", "lng"])
    near_rows.append({"city": "london (BikePoint)", "stations": len(lon),
                      **describe(nearest_other(xyz(lon.lat, lon.lng)))})

    hist, sens, near = pd.DataFrame(hist_rows), pd.DataFrame(sens_rows), pd.DataFrame(near_rows)
    hist.to_csv(OUT_DIR / "merge_radius_pairs.csv", index=False)
    sens.to_csv(OUT_DIR / "merge_radius_sensitivity.csv", index=False)
    near.to_csv(OUT_DIR / "nearest_station.csv", index=False)

    print(f"Όριο της ροής: {MERGE_RADIUS_M} m\n")
    print("1. Ζεύγη κωδικών ανά απόσταση (m):")
    print(hist.pivot(index=["from_m", "to_m"], columns="city", values="id_pairs").to_string())
    print("\n2. Ευαισθησία: φυσικοί σταθμοί (ενωμένοι κωδικοί) ανά όριο:")
    sens["cell"] = sens.physical_stations.map("{:,}".format) + " (" + sens.ids_merged.map("{:,}".format) + ")"
    print(sens.pivot(index="radius_m", columns="city", values="cell").to_string())
    print("\n3. Απόσταση από τον πλησιέστερο άλλο σταθμό (m):")
    print(near.round(1).to_string(index=False))
    print(f"\nSaved {OUT_DIR}/merge_radius_pairs.csv, merge_radius_sensitivity.csv, nearest_station.csv")


def describe(x: np.ndarray) -> dict:
    return {"min_m": float(np.min(x)), "p1_m": float(np.percentile(x, 1)),
            "p5_m": float(np.percentile(x, 5)), "median_m": float(np.median(x)),
            "under_25m": int((x < 25).sum()), "under_50m": int((x < 50).sum())}


if __name__ == "__main__":
    main()
