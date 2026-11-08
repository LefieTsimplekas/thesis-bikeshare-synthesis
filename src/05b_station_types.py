"""
05b_station_types.py

Τύποι σταθμών (origin_type) από τα προφίλ OSM του 05a (§3.2), και σύγκριση των
προφίλ των πόλεων (§3.4).

Βήματα:
  1. Μετασχηματισμός: log(1 + x) στις πυκνότητες και στην ποδηλατική υποδομή
     (πολύ ασύμμετρες κατανομές). Η κανονικοποιημένη απόσταση από το κέντρο μένει.
  2. Τυποποίηση με μέσο όρο και τυπική απόκλιση ΜΟΝΟ από τους σταθμούς Chicago και NYC.
  3. k-means ΜΟΝΟ στους σταθμούς των ΗΠΑ με τουλάχιστον MIN_TRIPS_FOR_FIT διαδρομές,
     για k στο STATION_TYPE_K_RANGE· επιλογή του k με το μεγαλύτερο silhouette.
  4. Κάθε σταθμός (και του Λονδίνου) παίρνει τον τύπο του πλησιέστερου κέντρου.
  5. Κάλυψη: μερίδιο των αφετηριών των διαδρομών εκπαίδευσης ανά τύπο, στο μείγμα
     50/50 Chicago-NYC (όπως το σενάριο C). Τύπος κάτω από MIN_TYPE_SHARE συγχωνεύεται
     με τον πλησιέστερο, και επαναλαμβάνεται μέχρι όλοι να το ξεπερνούν.
  6. Αρίθμηση τύπων κατά κεντρικότητα (0 = πιο κεντρικός), για σταθερή ερμηνεία.
  7. Πίνακας κάλυψης ανά πόλη, προφίλ ανά τύπο, και σύγκριση προφίλ πόλεων
     (MMD και Wasserstein ανά χαρακτηριστικό, πηγές έναντι Λονδίνου).

Κανένα δεδομένο του Λονδίνου δεν χρησιμοποιείται στα βήματα 1-6· το Λονδίνο μόνο
αντιστοιχίζεται και περιγράφεται.

Είσοδοι : data/context/osm/station_profiles_{city}.parquet   (από 05a)
          data/processed/{chicago,nyc}_train.parquet          (από 04b)
          data/processed/london_{train,test}.parquet          (από l2, μόνο για την περιγραφή)
Έξοδοι  : data/context/osm/station_types.parquet              (station_key, city, origin_type)
          artifacts/station_types/station_types_model.json    (k, silhouette, τυποποίηση, κέντρα)
          artifacts/station_types/type_coverage.csv
          artifacts/station_types/type_profiles.csv
          artifacts/station_types/city_profile_distance.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from config import (
    PROCESSED_DIR, OSM_DIR, SAMPLING_SEED,
    STATION_TYPE_K_RANGE, MIN_TYPE_SHARE, MIN_TRIPS_FOR_FIT,
)

SOURCE_CITIES = ["chicago", "nyc"]
TARGET_CITY = "london"
ALL_CITIES = SOURCE_CITIES + [TARGET_CITY]

PROFILE_COLS = [
    "dist_center_norm", "bike_station_density", "poi_density",
    "transit_density", "cycle_infra_km",
]
LOG_COLS = ["bike_station_density", "poi_density", "transit_density", "cycle_infra_km"]

ARTIFACT_DIR = Path("artifacts/station_types")
OUT_TYPES = OSM_DIR / "station_types.parquet"


# ---------------------------------------------------------------------------
# Δεδομένα
# ---------------------------------------------------------------------------
def load_profiles() -> pd.DataFrame:
    prof = pd.concat(
        [pd.read_parquet(OSM_DIR / f"station_profiles_{c}.parquet") for c in ALL_CITIES],
        ignore_index=True,
    )
    X = prof[PROFILE_COLS].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    prof[[f"x_{c}" for c in PROFILE_COLS]] = X.values
    return prof


def origin_counts(path: Path) -> pd.Series:
    """Πλήθος αφετηριών ανά φυσικό σταθμό σε ένα σύνολο διαδρομών."""
    return pd.read_parquet(path, columns=["start_station_key"])["start_station_key"].value_counts()


# ---------------------------------------------------------------------------
# Τυποποίηση και k-means (μόνο ΗΠΑ)
# ---------------------------------------------------------------------------
def standardize(prof: pd.DataFrame) -> tuple[np.ndarray, dict]:
    xcols = [f"x_{c}" for c in PROFILE_COLS]
    us = prof["city"].isin(SOURCE_CITIES)
    mean = prof.loc[us, xcols].mean()
    std = prof.loc[us, xcols].std(ddof=0)
    Z = ((prof[xcols] - mean) / std).to_numpy()
    params = {"mean": dict(zip(PROFILE_COLS, mean.round(6))),
              "std": dict(zip(PROFILE_COLS, std.round(6)))}
    return Z, params


def choose_k(Z_fit: np.ndarray) -> tuple[int, dict]:
    scores = {}
    for k in range(STATION_TYPE_K_RANGE[0], STATION_TYPE_K_RANGE[1] + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=SAMPLING_SEED).fit(Z_fit)
        scores[k] = float(silhouette_score(Z_fit, km.labels_))
    best = max(scores, key=scores.get)
    return best, scores


def type_shares(labels_by_city: dict, counts_by_city: dict, keys_by_city: dict) -> pd.DataFrame:
    """Μερίδιο αφετηριών ανά τύπο σε κάθε πόλη."""
    rows = {}
    for city, labels in labels_by_city.items():
        s = pd.Series(labels, index=keys_by_city[city])
        trips = counts_by_city[city].reindex(s.index, fill_value=0)
        rows[city] = trips.groupby(s.values).sum() / trips.sum()
    return pd.DataFrame(rows).fillna(0.0)


def merge_small_types(Z, centers, prof, counts_by_city):
    """Συγχωνεύει τύπους με μερίδιο < MIN_TYPE_SHARE στο μείγμα 50/50 Chicago-NYC:
    αφαιρεί το κέντρο τους και οι σταθμοί τους πάνε στο πλησιέστερο από τα υπόλοιπα."""
    merges = []
    keys_by_city = {c: prof.loc[prof.city == c, "station_key"].to_numpy() for c in SOURCE_CITIES}
    idx_by_city = {c: np.where(prof.city == c)[0] for c in SOURCE_CITIES}
    while True:
        labels = euclidean_distances(Z, centers).argmin(axis=1)
        shares = type_shares({c: labels[idx_by_city[c]] for c in SOURCE_CITIES},
                             counts_by_city, keys_by_city)
        pooled = shares.mean(axis=1).reindex(range(len(centers)), fill_value=0.0)
        smallest = int(pooled.idxmin())
        if pooled[smallest] >= MIN_TYPE_SHARE or len(centers) <= 2:
            return centers, labels, merges
        others = [i for i in range(len(centers)) if i != smallest]
        nearest = others[int(euclidean_distances(centers[[smallest]], centers[others]).argmin())]
        merges.append({"merged_type_share": round(float(pooled[smallest]), 4),
                       "into_center_index": int(nearest)})
        centers = centers[others]


# ---------------------------------------------------------------------------
# Σύγκριση προφίλ πόλεων (§3.4)
# ---------------------------------------------------------------------------
def rbf_mmd2(X, Y, max_n=2000, seed=SAMPLING_SEED):
    """Unbiased MMD² με RBF kernel και median heuristic."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    Zxy = np.vstack([X, Y])
    D = euclidean_distances(Zxy, Zxy, squared=True)
    K = np.exp(-D / np.median(D[D > 0]))
    n, m = len(X), len(Y)
    kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1))
            + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
            - 2 * kxy.mean())


def compare_cities(Z: np.ndarray, prof: pd.DataFrame) -> pd.DataFrame:
    tgt = Z[(prof.city == TARGET_CITY).to_numpy()]
    rows = []
    for name, cities in [("chicago", ["chicago"]), ("nyc", ["nyc"]), ("chicago+nyc", SOURCE_CITIES)]:
        src = Z[prof.city.isin(cities).to_numpy()]
        row = {"source": name, "mmd2_vs_london": rbf_mmd2(src, tgt)}
        for j, c in enumerate(PROFILE_COLS):
            row[f"W_{c}"] = wasserstein_distance(src[:, j], tgt[:, j])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prof = load_profiles()
    Z, scaler = standardize(prof)

    counts = {c: origin_counts(PROCESSED_DIR / f"{c}_train.parquet") for c in SOURCE_CITIES}
    counts["london_train"] = origin_counts(PROCESSED_DIR / "london_train.parquet")
    counts["london_test"] = origin_counts(PROCESSED_DIR / "london_test.parquet")

    # Σταθμοί που ορίζουν τους τύπους: ΗΠΑ, με τουλάχιστον MIN_TRIPS_FOR_FIT διαδρομές
    fit_mask = (prof.city.isin(SOURCE_CITIES) & (prof.n_trips >= MIN_TRIPS_FOR_FIT)).to_numpy()
    print(f"Stations used to define types: {fit_mask.sum():,} "
          f"(US stations with >= {MIN_TRIPS_FOR_FIT} trips; "
          f"{(prof.city.isin(SOURCE_CITIES) & ~fit_mask).sum()} excluded)")

    k, scores = choose_k(Z[fit_mask])
    print("\nSilhouette by k (US stations only):")
    for kk, s in scores.items():
        print(f"  k={kk}: {s:.4f}{'  <- chosen' if kk == k else ''}")
    km = KMeans(n_clusters=k, n_init=50, random_state=SAMPLING_SEED).fit(Z[fit_mask])

    # Συγχώνευση μικρών τύπων (μόνο με βάση ΗΠΑ)
    us_idx = prof.city.isin(SOURCE_CITIES).to_numpy()
    centers, labels_us, merges = merge_small_types(
        Z[us_idx], km.cluster_centers_, prof[us_idx].reset_index(drop=True),
        {c: counts[c] for c in SOURCE_CITIES})
    if merges:
        print(f"\nMerged {len(merges)} type(s) below {MIN_TYPE_SHARE:.0%}: {merges}")

    # Αρίθμηση κατά κεντρικότητα: 0 = μικρότερη κανονικοποιημένη απόσταση από το κέντρο
    order = np.argsort(centers[:, PROFILE_COLS.index("dist_center_norm")])
    centers = centers[order]
    prof["origin_type"] = euclidean_distances(Z, centers).argmin(axis=1)

    # Αποθήκευση τύπων
    prof[["station_key", "city", "origin_type"]].to_parquet(OUT_TYPES, index=False)

    # Κάλυψη ανά πόλη (μερίδιο αφετηριών ανά τύπο)
    cov = {}
    for name, city in [("chicago_train", "chicago"), ("nyc_train", "nyc"),
                       ("london_train", "london"), ("london_test", "london")]:
        cnt = counts[name if name.startswith("london") else city]
        s = prof.loc[prof.city == city].set_index("station_key")["origin_type"]
        trips = cnt.reindex(s.index, fill_value=0)
        cov[name] = trips.groupby(s.values).sum() / trips.sum()
    cov = pd.DataFrame(cov).fillna(0.0).sort_index()
    cov.insert(2, "pooled_50_50", cov[["chicago_train", "nyc_train"]].mean(axis=1))
    cov.index.name = "origin_type"
    cov.to_csv(ARTIFACT_DIR / "type_coverage.csv")
    print("\nShare of trip origins per type:")
    print((100 * cov).round(1).to_string())

    # Προφίλ ανά τύπο (ακατέργαστες τιμές, διάμεσος, μόνο ΗΠΑ)
    tp = (prof[prof.city.isin(SOURCE_CITIES)]
          .groupby("origin_type")[PROFILE_COLS + ["dist_center_km"]].median()
          .join(prof.groupby(["origin_type", "city"]).size().unstack(fill_value=0)))
    tp.to_csv(ARTIFACT_DIR / "type_profiles.csv")
    print("\nMedian profile per type (US stations) and number of stations per city:")
    print(tp.round(2).to_string())

    # Σύγκριση προφίλ πόλεων
    dist = compare_cities(Z, prof)
    dist.to_csv(ARTIFACT_DIR / "city_profile_distance.csv", index=False)
    print("\nDistance of station profiles to London (standardized with US statistics):")
    print(dist.round(4).to_string(index=False))

    model = {
        "profile_cols": PROFILE_COLS,
        "log1p_cols": LOG_COLS,
        "standardization_fitted_on": SOURCE_CITIES,
        "scaler": scaler,
        "min_trips_for_fit": MIN_TRIPS_FOR_FIT,
        "n_stations_fit": int(fit_mask.sum()),
        "k_range": list(STATION_TYPE_K_RANGE),
        "silhouette": {str(kk): round(s, 4) for kk, s in scores.items()},
        "k_chosen": int(k),
        "merges_below_min_share": merges,
        "min_type_share": MIN_TYPE_SHARE,
        "n_types_final": int(len(centers)),
        "centers_standardized": np.round(centers, 6).tolist(),
        "seed": SAMPLING_SEED,
    }
    (ARTIFACT_DIR / "station_types_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_TYPES} and artifacts in {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
